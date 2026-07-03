"""
createGraphDataset.py
=====================
Konvertiert VTK-Ergebnisse aus der CFD-Pipeline in PyTorch Geometric
Graph-Objekte fuer GNN-Training.

Ablauf:
    1. dataset_overview.csv einlesen (Simulationsparameter)
    2. Fuer jede gueltige Simulation (Converged / NotConverged):
       a. internal.vtu laden (Volumenfeld, cell-centered)
       b. boundary/*.vtp laden (Patch-Zuordnung, Wandflaechen)
       c. Zellzentren als Graph-Knoten extrahieren
       d. Wall-Distance berechnen (KDTree auf building + ground Faces)
       e. Adaptives Subsampling (gebaeude-nah fein, Fernfeld grob)
       f. Node-Type aus Patch-Zugehoerigkeit bestimmen (One-Hot)
       g. Face-Connectivity -> bidirektionale Kanten (Fallback: k-NN)
       h. Input- und Output-Features als Tensoren zusammenstellen
       i. PyTorch Geometric Data-Objekt erzeugen und speichern
    3. Train/Val/Test Split (70/15/15)
    4. Zusammenfassung ausgeben

Input-Features pro Knoten (14):
    - x, y, z                   (3)  Knotenposition
    - wall_distance              (1)  Abstand zur naechsten Wand
    - cell_volume                (1)  Zellvolumen
    - node_type                  (7)  One-Hot: interior/inlet/outlet/
                                      ground/building/top/sides
    - U_ref                      (1)  Referenzwindgeschwindigkeit [global]
    - angle                      (1)  Windwinkel [global]

Output-Features pro Knoten (6):
    - Ux, Uy, Uz                (3)  Geschwindigkeitskomponenten
    - p                          (1)  Druck
    - k                          (1)  Turbulente kinetische Energie
    - epsilon                    (1)  Dissipationsrate

Referenzen:
    - Pfaff et al. (2020): MeshGraphNets, Encode-Process-Decode
    - Gladstone et al. (2024): GNN Surrogates for time-independent PDEs
    - Fortunato et al. (2022): Multiscale MeshGraphNets
    - pyvista Dokumentation: https://docs.pyvista.org/
    - PyTorch Geometric: https://pytorch-geometric.readthedocs.io/

Siehe QUELLEN.md fuer vollstaendige Quellenangaben.
"""

import os
import sys
import csv
import time
import shutil
import tarfile
import argparse
import subprocess
from datetime import datetime

import numpy as np
import yaml
import torch
from torch_geometric.data import Data
from scipy.spatial import KDTree


# ======================================================================
# Optionale Imports mit Fehlermeldung
# ======================================================================

try:
    import pyvista as pv
except ImportError:
    print("FEHLER: 'pyvista' nicht installiert.")
    print("        Installation: pip install pyvista")
    sys.exit(1)


# ======================================================================
# Konfiguration
# ======================================================================

# Patch-Namen -> Node-Type Index (fuer One-Hot Encoding)
PATCH_TYPE_MAP = {
    "interior": 0,
    "inlet":    1,
    "outlet":   2,
    "ground":   3,
    "building": 4,
    "top":      5,
    "sides":    6,
}
NUM_NODE_TYPES = len(PATCH_TYPE_MAP)

# Subsampling-Parameter (wandabstandsbasiert) - 3 Level
#
# COARSE: ~10x weniger Knoten als MEDIUM, fuer schnelle Experimente
# MEDIUM: aktueller Stand (Ergebnisse vorhanden)
# FINE:   grosse Knotenanzahl, zielt auf ~70-90% der 96 GB VRAM der
#         NVIDIA RTX Pro 6000 Q Max. Die genauen Ratios muessen ggf.
#         nach einem ersten Testlauf kalibriert werden.
#
# Format: (max_wall_distance_m, keep_ratio)
SUBSAMPLE_ZONES = {
    "coarse": [
        (2.0,    0.01),    # Gebaeudenah:  1%
        (10.0,   0.002),   # Nahfeld:      0.2%
        (50.0,   0.0005),  # Mittelfeld:   0.05%
        (np.inf, 0.0002),  # Fernfeld:     0.02%
    ],
    "medium": [
        (2.0,    0.10),    # Gebaeudenah:  10% (Abloesung, Rezirkulation)
        (10.0,   0.02),    # Nahfeld:      2%  (Nachlauf, Wirbelstrukturen)
        (50.0,   0.005),   # Mittelfeld:   0.5% (Uebergangszone)
        (np.inf, 0.002),   # Fernfeld:     0.2% (nahezu Freistrom)
    ],
    "fine": [
        (2.0,    1.0),     # Gebaeudenah:  100% (alle Zellen behalten)
        (10.0,   0.30),    # Nahfeld:      30%
        (50.0,   0.05),    # Mittelfeld:   5%
        (np.inf, 0.02),    # Fernfeld:     2%
    ],
    "building_focus": [
        (2.0,    0.50),    # Gebaeudenah:  50% (Fokus auf Abloesung, Rezirkulation)
        (10.0,   0.10),    # Nahfeld:      10%
        (50.0,   0.02),    # Mittelfeld:   2%
        (np.inf, 0.005),   # Fernfeld:     0.5%
    ],
    "building_focus_25": [
        (2.0,    0.25),    # Gebaeudenah:  25%
        (10.0,   0.05),    # Nahfeld:      5%
        (50.0,   0.01),    # Mittelfeld:   1%
        (np.inf, 0.003),   # Fernfeld:     0.3%
    ],
}


# ======================================================================
# Konfigurationsdatei laden
# ======================================================================

def load_config(config_path):
    """Laedt die YAML-Konfigurationsdatei."""
    if not os.path.exists(config_path):
        print(f"FEHLER: '{config_path}' nicht gefunden.")
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ======================================================================
# S3-Hilfsfunktionen
# ======================================================================

def download_from_s3(s3_path, local_path, s3_cfg, max_retries=3):
    """
    Laedt eine Datei von S3 herunter mit Retry-Logik.

    Parameter
    ---------
    s3_path : str
        S3-URI (z.B. s3://bucket/prefix/file.csv).
    local_path : str
        Lokaler Zielpfad.
    s3_cfg : dict
        S3-Konfiguration aus config.yaml.
    max_retries : int
        Maximale Anzahl Download-Versuche.

    Rueckgabe
    ---------
    bool : True bei Erfolg, False bei Fehler.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            subprocess.run(
                ["aws", "s3", "cp", s3_path, local_path,
                 "--cli-read-timeout", "120",
                 "--cli-connect-timeout", "30"],
                check=True,
                capture_output=True,
            )
            if os.path.exists(local_path):
                return True
        except Exception as e:
            if attempt < max_retries:
                wait = 10 * (2 ** (attempt - 1))
                print(f"   WARNUNG: S3-Download fehlgeschlagen "
                      f"(Versuch {attempt}/{max_retries}), "
                      f"naechster Versuch in {wait}s...")
                time.sleep(wait)
            else:
                err_msg = str(e)
                if hasattr(e, "stderr") and e.stderr:
                    err_msg = e.stderr.decode("utf-8", errors="replace").strip()
                print(f"   FEHLER: S3-Download endgueltig fehlgeschlagen "
                      f"nach {max_retries} Versuchen: {err_msg}")
    return False


def upload_to_s3(local_path, s3_path, s3_cfg, max_retries=3):
    """
    Laedt eine lokale Datei nach S3 hoch mit Retry-Logik und Verifikation.

    Parameter
    ---------
    local_path : str
        Lokaler Quellpfad.
    s3_path : str
        S3-URI (z.B. s3://bucket/prefix/file.pt).
    s3_cfg : dict
        S3-Konfiguration aus config.yaml.
    max_retries : int
        Maximale Anzahl Upload-Versuche.

    Rueckgabe
    ---------
    bool : True bei Erfolg, False bei Fehler.
    """
    local_size = os.path.getsize(local_path)

    for attempt in range(1, max_retries + 1):
        try:
            subprocess.run(
                ["aws", "s3", "cp", local_path, s3_path,
                 "--cli-read-timeout", "120",
                 "--cli-connect-timeout", "30"],
                check=True,
                capture_output=True,
            )

            # Verifikation: Dateigroesse in S3 pruefen
            result = subprocess.run(
                ["aws", "s3", "ls", s3_path],
                capture_output=True, check=True,
            )
            parts = result.stdout.decode().strip().split()
            if len(parts) >= 3 and int(parts[2]) == local_size:
                return True
            else:
                raise RuntimeError("Verifikation fehlgeschlagen: "
                                   "Dateigroesse in S3 stimmt nicht")

        except Exception as e:
            if attempt < max_retries:
                wait = 10 * (2 ** (attempt - 1))
                print(f"   WARNUNG: S3-Upload fehlgeschlagen "
                      f"(Versuch {attempt}/{max_retries}), "
                      f"naechster Versuch in {wait}s...")
                time.sleep(wait)
            else:
                err_msg = str(e)
                if hasattr(e, "stderr") and e.stderr:
                    err_msg = e.stderr.decode("utf-8", errors="replace").strip()
                print(f"   FEHLER: S3-Upload endgueltig fehlgeschlagen "
                      f"nach {max_retries} Versuchen: {err_msg}")
    return False


def download_and_extract_vtk(sim_id, s3_cfg, vtk_base_dir):
    """
    Laedt sim_XXX.tar.gz von S3 herunter und entpackt es.

    Parameter
    ---------
    sim_id : str
        Simulationsname (z.B. sim_000).
    s3_cfg : dict
        S3-Konfiguration aus config.yaml.
    vtk_base_dir : str
        Lokales Basisverzeichnis fuer VTK-Dateien.

    Rueckgabe
    ---------
    str : Pfad zum entpackten VTK-Verzeichnis (sim_XXX_VTK/).
    """
    tar_name = f"{sim_id}.tar.gz"
    s3_path = f"s3://{s3_cfg['bucket']}/{s3_cfg['prefix']}/{tar_name}"
    local_tar = os.path.join(vtk_base_dir, tar_name)

    os.makedirs(vtk_base_dir, exist_ok=True)

    # Herunterladen
    ok = download_from_s3(s3_path, local_tar, s3_cfg)
    if not ok:
        raise RuntimeError(f"Download von {s3_path} fehlgeschlagen")

    # Entpacken
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=vtk_base_dir)

    # tar.gz sofort loeschen
    os.remove(local_tar)

    # Das tar.gz enthaelt sim_XXX/ — wir brauchen sim_XXX_VTK/
    # In main.py wird mit arcname=sim_id gepackt, also entpackt als sim_XXX/
    extracted_dir = os.path.join(vtk_base_dir, sim_id)
    vtk_dir = os.path.join(vtk_base_dir, f"{sim_id}_VTK")

    if os.path.isdir(extracted_dir) and not os.path.isdir(vtk_dir):
        os.rename(extracted_dir, vtk_dir)

    if not os.path.isdir(vtk_dir):
        raise RuntimeError(f"Entpacktes VTK-Verzeichnis nicht gefunden: {vtk_dir}")

    return vtk_dir


# ======================================================================
# CSV einlesen
# ======================================================================

def load_simulation_metadata(csv_path):
    """
    Liest dataset_overview.csv und filtert auf gueltige Simulationen.

    Rueckgabe
    ---------
    list[dict] : Liste mit Dicts {id, U_ref, angle, status}
    """
    if not os.path.exists(csv_path):
        print(f"FEHLER: '{csv_path}' nicht gefunden.")
        sys.exit(1)

    valid_status = {"Converged", "NotConverged"}
    simulations = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Status"] in valid_status:
                simulations.append({
                    "id":    row["ID"],
                    "U_ref": float(row["U_ref"]),
                    "angle": float(row["Angle"]),
                    "status": row["Status"],
                })

    return simulations


# ======================================================================
# VTK laden
# ======================================================================

def _find_timestep_dir(vtk_dir):
    """
    Findet das Zeitschritt-Unterverzeichnis innerhalb eines VTK-Ordners.

    foamToVTK erzeugt z.B. sim_000_VTK/sim_000_1104/internal.vtu,
    wobei der Unterordner-Name den Zeitschritt enthaelt und variabel ist.

    Sucht nach dem ersten Unterverzeichnis, das internal.vtu enthaelt.
    Gibt vtk_dir direkt zurueck, falls internal.vtu dort liegt (Fallback).

    Parameter
    ---------
    vtk_dir : str
        Pfad zum VTK-Verzeichnis (z.B. results/vtks/sim_000_VTK/).

    Rueckgabe
    ---------
    str : Pfad zum Verzeichnis, das internal.vtu enthaelt.
    """
    # Fallback: internal.vtu liegt direkt in vtk_dir
    if os.path.exists(os.path.join(vtk_dir, "internal.vtu")):
        return vtk_dir

    # Suche in Unterverzeichnissen
    for entry in sorted(os.listdir(vtk_dir)):
        subdir = os.path.join(vtk_dir, entry)
        if os.path.isdir(subdir) and os.path.exists(
                os.path.join(subdir, "internal.vtu")):
            return subdir

    # Nichts gefunden — gib vtk_dir zurueck (Fehler wird spaeter geworfen)
    return vtk_dir


def load_volume_mesh(vtk_dir):
    """
    Laedt internal.vtu (Volumen-Mesh) aus dem VTK-Verzeichnis.

    Sucht automatisch in Zeitschritt-Unterverzeichnissen, da foamToVTK
    die Dateien unter einem variablen Zeitschritt-Ordner ablegt.

    Parameter
    ---------
    vtk_dir : str
        Pfad zum VTK-Verzeichnis (z.B. results/vtks/sim_000_VTK/).

    Rueckgabe
    ---------
    pyvista.UnstructuredGrid
        Das geladene Volumen-Mesh mit CellData.
    """
    data_dir = _find_timestep_dir(vtk_dir)
    vtu_path = os.path.join(data_dir, "internal.vtu")
    if not os.path.exists(vtu_path):
        raise FileNotFoundError(f"internal.vtu nicht gefunden in {vtk_dir}")
    return pv.read(vtu_path)


def load_boundary_patches(vtk_dir):
    """
    Laedt alle Boundary-Patches aus dem boundary/-Unterverzeichnis.

    Sucht automatisch in Zeitschritt-Unterverzeichnissen.

    Rueckgabe
    ---------
    dict : {patch_name: pyvista.PolyData}
    """
    data_dir = _find_timestep_dir(vtk_dir)
    boundary_dir = os.path.join(data_dir, "boundary")
    patches = {}

    if not os.path.isdir(boundary_dir):
        print(f"  [WARNUNG] Kein boundary/-Verzeichnis in {vtk_dir}")
        return patches

    for fname in os.listdir(boundary_dir):
        if fname.endswith(".vtp"):
            patch_name = fname.replace(".vtp", "")
            patches[patch_name] = pv.read(os.path.join(boundary_dir, fname))

    return patches


# ======================================================================
# Feature-Extraktion
# ======================================================================

def extract_cell_centers(mesh):
    """
    Berechnet die Zellmittelpunkte eines UnstructuredGrid.

    Rueckgabe
    ---------
    np.ndarray : (N, 3) Zellzentren [x, y, z]
    """
    centers = mesh.cell_centers()
    return np.array(centers.points, dtype=np.float32)


def extract_cell_volumes(mesh):
    """
    Berechnet die Zellvolumina.

    Rueckgabe
    ---------
    np.ndarray : (N,) Zellvolumina [m^3]
    """
    sized = mesh.compute_cell_sizes(length=False, area=False, volume=True)
    return np.array(sized.cell_data["Volume"], dtype=np.float32)


def extract_flow_fields(mesh):
    """
    Extrahiert die Stroemungsfelder aus CellData.

    Rueckgabe
    ---------
    dict : {field_name: np.ndarray}
        U: (N, 3), p: (N,), k: (N,), epsilon: (N,)
    """
    fields = {}

    # Geschwindigkeit (Vektorfeld)
    if "U" in mesh.cell_data:
        fields["U"] = np.array(mesh.cell_data["U"], dtype=np.float32)
    else:
        raise KeyError("Feld 'U' nicht in CellData gefunden.")

    # Skalarfelder
    for name in ["p", "k", "epsilon"]:
        if name in mesh.cell_data:
            fields[name] = np.array(mesh.cell_data[name], dtype=np.float32)
        else:
            raise KeyError(f"Feld '{name}' nicht in CellData gefunden.")

    return fields


# ======================================================================
# Wall-Distance Berechnung
# ======================================================================

def compute_wall_distance(cell_centers, patches):
    """
    Berechnet den Wandabstand jedes Zellzentrums zur naechsten
    Wandflaeche (building + ground Patches).

    Verwendet scipy.spatial.KDTree auf den Face-Mittelpunkten der
    Wand-Patches fuer effiziente Nearest-Neighbor-Suche.

    Parameter
    ---------
    cell_centers : np.ndarray
        (N, 3) Zellzentren des Volumen-Meshes.
    patches : dict
        {patch_name: pyvista.PolyData} Boundary-Patches.

    Rueckgabe
    ---------
    np.ndarray : (N,) Wandabstand [m]
    """
    wall_patches = ["building", "ground"]
    wall_points = []

    for name in wall_patches:
        if name in patches:
            patch = patches[name]
            # Face-Mittelpunkte des Boundary-Patches
            face_centers = patch.cell_centers()
            wall_points.append(np.array(face_centers.points, dtype=np.float32))

    if not wall_points:
        print("  [WARNUNG] Keine Wand-Patches gefunden. "
              "wall_distance wird auf 0 gesetzt.")
        return np.zeros(len(cell_centers), dtype=np.float32)

    wall_points = np.vstack(wall_points)

    # KDTree fuer effiziente Nearest-Neighbor-Suche
    tree = KDTree(wall_points)
    distances, _ = tree.query(cell_centers)

    return distances.astype(np.float32)


# ======================================================================
# Node-Type Bestimmung
# ======================================================================

def compute_node_types(cell_centers, patches, mesh):
    """
    Bestimmt den Node-Type jeder Zelle als One-Hot-Vektor.

    Strategie:
        1. Alle internen Zellen erhalten Type 'interior' (Default)
        2. Fuer jeden Boundary-Patch: Finde die Volumenzellen, die
           an den Patch grenzen (ueber KDTree-Naechster-Nachbar)

    Parameter
    ---------
    cell_centers : np.ndarray
        (N, 3) Zellzentren.
    patches : dict
        {patch_name: pyvista.PolyData} Boundary-Patches.
    mesh : pyvista.UnstructuredGrid
        Volumen-Mesh (fuer Zellgroessen-Referenz).

    Rueckgabe
    ---------
    np.ndarray : (N, 7) One-Hot-Encoding der Node-Types
    """
    n_cells = len(cell_centers)
    node_types = np.zeros((n_cells, NUM_NODE_TYPES), dtype=np.float32)

    # Default: alle interior
    node_types[:, PATCH_TYPE_MAP["interior"]] = 1.0

    # Typische Basiszellgroesse (fuer Schwellwert)
    # Naechster-Nachbar Abstand < 2x Zellgroesse = Randzelle
    cell_volumes = extract_cell_volumes(mesh)
    median_cell_size = np.median(np.cbrt(cell_volumes))
    distance_threshold = 2.0 * median_cell_size

    # KDTree auf Zellzentren fuer schnelle Zuordnung
    cell_tree = KDTree(cell_centers)

    for patch_name, patch_data in patches.items():
        if patch_name not in PATCH_TYPE_MAP:
            continue
        if patch_name == "interior":
            continue

        type_idx = PATCH_TYPE_MAP[patch_name]

        # Face-Mittelpunkte des Boundary-Patches
        face_centers = np.array(
            patch_data.cell_centers().points, dtype=np.float32)

        if len(face_centers) == 0:
            continue

        # Finde die naechsten Volumenzellen zu jeder Boundary-Face
        dists, indices = cell_tree.query(face_centers)

        # Nur Zellen innerhalb des Schwellwerts zuordnen
        valid = dists < distance_threshold
        valid_indices = np.unique(indices[valid])

        # Ueberschreibe den Node-Type (Boundary > Interior)
        node_types[valid_indices, :] = 0.0
        node_types[valid_indices, type_idx] = 1.0

    return node_types


# ======================================================================
# Adaptives Subsampling
# ======================================================================

def adaptive_subsample(cell_centers, wall_distances, rng, zones=None):
    """
    Fuehrt wandabstandsbasiertes Subsampling durch.

    Parameter
    ---------
    cell_centers : np.ndarray
        (N, 3) Zellzentren.
    wall_distances : np.ndarray
        (N,) Wandabstaende.
    rng : np.random.Generator
        Zufallsgenerator (fuer reproduzierbares Sampling).
    zones : list[tuple] oder None
        Liste von (max_wall_distance, keep_ratio) Tupeln.
        Falls None, wird SUBSAMPLE_ZONES["medium"] verwendet.

    Rueckgabe
    ---------
    np.ndarray : Indizes der behaltenen Zellen (sortiert)
    """
    if zones is None:
        zones = SUBSAMPLE_ZONES["medium"]

    n_total = len(cell_centers)
    keep_mask = np.zeros(n_total, dtype=bool)

    prev_limit = 0.0
    for max_dist, keep_ratio in zones:
        zone_mask = (wall_distances >= prev_limit) & (wall_distances < max_dist)
        zone_indices = np.where(zone_mask)[0]
        n_zone = len(zone_indices)

        if n_zone == 0:
            prev_limit = max_dist
            continue

        if keep_ratio >= 1.0:
            # Alle behalten
            keep_mask[zone_indices] = True
        else:
            # Zufaellig samplen
            n_keep = max(1, int(n_zone * keep_ratio))
            selected = rng.choice(zone_indices, size=n_keep, replace=False)
            keep_mask[selected] = True

        prev_limit = max_dist

    return np.sort(np.where(keep_mask)[0])


# ======================================================================
# Graph-Kanten (k-Nearest Neighbors)
# ======================================================================

def build_knn_edges(positions, k=20):
    """
    Erzeugt bidirektionale Kanten via k-Nearest-Neighbors.

    Verwendet KDTree fuer effiziente Suche. Kanten werden
    bidirektional angelegt (i->j und j->i).

    Parameter
    ---------
    positions : np.ndarray
        (N, 3) Knotenpositionen.
    k : int
        Anzahl naechster Nachbarn pro Knoten.

    Rueckgabe
    ---------
    np.ndarray : (2, E) Edge-Index [source, target]

    Referenz:
        k-NN Graphen sind Standard in MeshGraphNets (Pfaff et al., 2020)
        und GNN-Surrogaten (Gladstone et al., 2024).
    """
    n_nodes = len(positions)
    actual_k = min(k + 1, n_nodes)  # +1 weil KDTree den Punkt selbst findet

    tree = KDTree(positions)
    _, indices = tree.query(positions, k=actual_k)

    # Kanten aufbauen (Self-Loops ausschliessen)
    sources = []
    targets = []

    for i in range(n_nodes):
        for j_idx in range(actual_k):
            j = indices[i, j_idx]
            if i != j:
                sources.append(i)
                targets.append(j)

    edge_index = np.array([sources, targets], dtype=np.int64)
    return edge_index


# ======================================================================
# Graph-Erstellung fuer eine Simulation
# ======================================================================

def process_simulation(sim_meta, vtk_base_dir, rng, k_neighbors=20, zones=None):
    """
    Verarbeitet eine einzelne Simulation zu einem PyTorch Geometric
    Data-Objekt.

    Parameter
    ---------
    sim_meta : dict
        Metadaten: {id, U_ref, angle, status}.
    vtk_base_dir : str
        Basispfad zu den VTK-Verzeichnissen.
    rng : np.random.Generator
        Zufallsgenerator.
    k_neighbors : int
        Anzahl k-NN Nachbarn fuer Kanten.
    zones : list[tuple] oder None
        Subsampling-Zonen. Falls None: medium-Level.

    Rueckgabe
    ---------
    torch_geometric.data.Data : Graph-Objekt
    dict : Statistiken {n_original, n_subsampled, n_edges}
    """
    sim_id = sim_meta["id"]
    U_ref = sim_meta["U_ref"]
    angle = sim_meta["angle"]

    vtk_dir = os.path.join(vtk_base_dir, f"{sim_id}_VTK")
    if not os.path.isdir(vtk_dir):
        raise FileNotFoundError(f"VTK-Verzeichnis nicht gefunden: {vtk_dir}")

    # ------------------------------------------------------------------
    # 1. VTK laden
    # ------------------------------------------------------------------
    mesh = load_volume_mesh(vtk_dir)
    patches = load_boundary_patches(vtk_dir)

    # ------------------------------------------------------------------
    # 2. Features aus dem vollen Mesh extrahieren
    # ------------------------------------------------------------------
    cell_centers = extract_cell_centers(mesh)
    cell_volumes = extract_cell_volumes(mesh)
    fields = extract_flow_fields(mesh)
    n_original = len(cell_centers)

    # ------------------------------------------------------------------
    # 3. Wall-Distance berechnen (auf vollem Mesh)
    # ------------------------------------------------------------------
    wall_distances = compute_wall_distance(cell_centers, patches)

    # ------------------------------------------------------------------
    # 4. Node-Types bestimmen (auf vollem Mesh)
    # ------------------------------------------------------------------
    node_types = compute_node_types(cell_centers, patches, mesh)

    # ------------------------------------------------------------------
    # 5. Adaptives Subsampling
    # ------------------------------------------------------------------
    keep_indices = adaptive_subsample(cell_centers, wall_distances, rng,
                                      zones=zones)
    n_subsampled = len(keep_indices)

    # Features auf Subsample reduzieren
    sub_centers     = cell_centers[keep_indices]
    sub_volumes     = cell_volumes[keep_indices]
    sub_wall_dist   = wall_distances[keep_indices]
    sub_node_types  = node_types[keep_indices]
    sub_U           = fields["U"][keep_indices]
    sub_p           = fields["p"][keep_indices]
    sub_k           = fields["k"][keep_indices]
    sub_eps         = fields["epsilon"][keep_indices]

    # ------------------------------------------------------------------
    # 6. Graph-Kanten (k-NN auf subgesampleten Positionen)
    # ------------------------------------------------------------------
    edge_index = build_knn_edges(sub_centers, k=k_neighbors)
    n_edges = edge_index.shape[1]

    # ------------------------------------------------------------------
    # 7. Feature-Tensoren zusammenstellen
    # ------------------------------------------------------------------

    # Input-Features: [x, y, z, wall_dist, cell_vol, node_type(7), U_ref, angle]
    n = n_subsampled
    global_features = np.column_stack([
        np.full(n, U_ref, dtype=np.float32),
        np.full(n, angle, dtype=np.float32),
    ])

    x = np.column_stack([
        sub_centers,                            # (N, 3)  x, y, z
        sub_wall_dist.reshape(-1, 1),           # (N, 1)  wall_distance
        sub_volumes.reshape(-1, 1),             # (N, 1)  cell_volume
        sub_node_types,                         # (N, 7)  node_type one-hot
        global_features,                        # (N, 2)  U_ref, angle
    ])  # Gesamt: (N, 14)

    # Output-Features (Targets): [Ux, Uy, Uz, p, k, epsilon]
    y = np.column_stack([
        sub_U,                                  # (N, 3)  Ux, Uy, Uz
        sub_p.reshape(-1, 1),                   # (N, 1)  p
        sub_k.reshape(-1, 1),                   # (N, 1)  k
        sub_eps.reshape(-1, 1),                 # (N, 1)  epsilon
    ])  # Gesamt: (N, 6)

    # ------------------------------------------------------------------
    # 8. PyTorch Geometric Data-Objekt
    # ------------------------------------------------------------------
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        pos=torch.tensor(sub_centers, dtype=torch.float32),
    )

    # Metadaten als Attribute
    data.sim_id = sim_id
    data.U_ref = U_ref
    data.angle = angle
    data.status = sim_meta["status"]
    data.n_original = n_original
    data.n_subsampled = n_subsampled

    stats = {
        "n_original": n_original,
        "n_subsampled": n_subsampled,
        "n_edges": n_edges,
    }

    return data, stats


# ======================================================================
# Train / Val / Test Split
# ======================================================================

def split_dataset(data_list, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Teilt den Datensatz in Train/Val/Test auf.

    Parameter
    ---------
    data_list : list[Data]
        Liste aller Graph-Objekte.
    train_ratio : float
        Anteil Trainingsdaten.
    val_ratio : float
        Anteil Validierungsdaten.
    seed : int
        Random Seed fuer Reproduzierbarkeit.

    Rueckgabe
    ---------
    tuple : (train_list, val_list, test_list)
    """
    rng = np.random.default_rng(seed)
    n = len(data_list)
    indices = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_list = [data_list[i] for i in train_idx]
    val_list = [data_list[i] for i in val_idx]
    test_list = [data_list[i] for i in test_idx]

    return train_list, val_list, test_list


# ======================================================================
# Speichern
# ======================================================================

def save_dataset(train, val, test, output_dir, subsample_level, zones):
    """
    Speichert die Datensatz-Splits als .pt Dateien.

    Verzeichnisstruktur:
        output_dir/
            train.pt        -> Liste von Data-Objekten
            val.pt          -> Liste von Data-Objekten
            test.pt         -> Liste von Data-Objekten
            metadata.yaml   -> Datensatz-Metadaten
    """
    os.makedirs(output_dir, exist_ok=True)

    torch.save(train, os.path.join(output_dir, "train.pt"))
    torch.save(val, os.path.join(output_dir, "val.pt"))
    torch.save(test, os.path.join(output_dir, "test.pt"))

    # Metadaten-Datei
    metadata = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "subsample_level": subsample_level,
        "n_total": len(train) + len(val) + len(test),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "input_features": 14,
        "output_features": 6,
        "feature_names_input": [
            "x", "y", "z", "wall_distance", "cell_volume",
            "type_interior", "type_inlet", "type_outlet",
            "type_ground", "type_building", "type_top", "type_sides",
            "U_ref", "angle",
        ],
        "feature_names_output": [
            "Ux", "Uy", "Uz", "p", "k", "epsilon",
        ],
        "subsampling_zones": [
            {"max_distance_m": z[0] if z[0] != np.inf else "inf",
             "keep_ratio": z[1]}
            for z in zones
        ],
    }

    with open(os.path.join(output_dir, "metadata.yaml"), "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)


def write_subsample_report(output_dir, subsample_level, zones,
                           sim_stats, total_time, k_neighbors):
    """
    Schreibt einen detaillierten Subsampling-Report nach output_dir/subsample_report.yaml.

    Enthaelt pro Simulation: Knotenanzahl (original + subsampled), Kantenanzahl,
    Laufzeit und VRAM-Schaetzung (Rohdaten). Zusaetzlich Gesamtstatistiken.

    VRAM-Schaetzung (Rohdaten, float32):
        x:          N * 14 * 4 Bytes
        y:          N *  6 * 4 Bytes
        pos:        N *  3 * 4 Bytes
        edge_index: E *  2 * 8 Bytes  (E = N * k_neighbors * 2, bidirektional)
    Hinweis: Tatsaechlicher VRAM-Bedarf beim GNN-Training ist durch
    Intermediate-Activations, Gradienten und Modellparameter deutlich hoeher.
    """
    os.makedirs(output_dir, exist_ok=True)

    nodes_list  = [s["n_subsampled"] for s in sim_stats]
    orig_list   = [s["n_original"]   for s in sim_stats]
    edges_list  = [s["n_edges"]      for s in sim_stats]
    time_list   = [s["time_s"]       for s in sim_stats]

    def _vram_mb(n_nodes, n_edges):
        b = (n_nodes * 14 * 4 +   # x
             n_nodes *  6 * 4 +   # y
             n_nodes *  3 * 4 +   # pos
             n_edges *  2 * 8)    # edge_index
        return round(b / (1024 * 1024), 2)

    per_sim = []
    for s in sim_stats:
        per_sim.append({
            "sim_id":       s["sim_id"],
            "n_original":   int(s["n_original"]),
            "n_subsampled": int(s["n_subsampled"]),
            "reduction_%":  round(s["n_subsampled"] / max(s["n_original"], 1) * 100, 3),
            "n_edges":      int(s["n_edges"]),
            "time_s":       round(s["time_s"], 2),
            "vram_raw_MB":  _vram_mb(s["n_subsampled"], s["n_edges"]),
        })

    n = len(nodes_list)
    avg_n    = int(sum(nodes_list) / max(n, 1))
    avg_e    = int(sum(edges_list) / max(n, 1))
    avg_vram = _vram_mb(avg_n, avg_e)

    report = {
        "created":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "subsample_level":  subsample_level,
        "k_neighbors":      k_neighbors,
        "zones": [
            {"max_distance_m": (z[0] if z[0] != np.inf else "inf"),
             "keep_ratio": z[1]}
            for z in zones
        ],
        "summary": {
            "n_simulations":        n,
            "total_time_min":       round(total_time / 60, 2),
            "nodes_avg":            avg_n,
            "nodes_min":            int(min(nodes_list)) if nodes_list else 0,
            "nodes_max":            int(max(nodes_list)) if nodes_list else 0,
            "edges_avg":            avg_e,
            "edges_min":            int(min(edges_list)) if edges_list else 0,
            "edges_max":            int(max(edges_list)) if edges_list else 0,
            "original_nodes_avg":   int(sum(orig_list) / max(n, 1)),
            "reduction_%_avg":      round(avg_n / max(int(sum(orig_list) / max(n, 1)), 1) * 100, 3),
            "time_per_sim_avg_s":   round(sum(time_list) / max(n, 1), 2),
            "time_per_sim_min_s":   round(min(time_list), 2) if time_list else 0,
            "time_per_sim_max_s":   round(max(time_list), 2) if time_list else 0,
            "vram_raw_avg_MB":      avg_vram,
            "vram_raw_min_MB":      _vram_mb(min(nodes_list) if nodes_list else 0,
                                             min(edges_list) if edges_list else 0),
            "vram_raw_max_MB":      _vram_mb(max(nodes_list) if nodes_list else 0,
                                             max(edges_list) if edges_list else 0),
            "note_vram": (
                "Rohdaten-Schaetzung (x+y+pos+edge_index). "
                "Tatsaechlicher GPU-Bedarf beim Training ist durch Activations, "
                "Gradienten und Modellparameter deutlich hoeher (~5-10x)."
            ),
        },
        "per_simulation": per_sim,
    }

    report_path = os.path.join(output_dir, "subsample_report.yaml")
    with open(report_path, "w") as f:
        yaml.dump(report, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False)
    return report_path


# ======================================================================
# Hauptprogramm
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VTK -> PyTorch Geometric Graph-Datensatz"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Pfad zur Konfigurationsdatei (Standard: config.yaml)"
    )
    parser.add_argument(
        "--k-neighbors", type=int, default=20,
        help="Anzahl k-NN Nachbarn fuer Graph-Kanten (Standard: 20)"
    )
    parser.add_argument(
        "--subsample-level",
        choices=["coarse", "medium", "fine", "building_focus", "building_focus_25"],
        default="medium",
        help=(
            "Subsampling-Level:\n"
            "  coarse - ~10x weniger Knoten als medium, fuer schnelle Tests\n"
            "  medium - aktueller Stand (Ergebnisse vorhanden)\n"
            "  fine   - maximale Aufloesung, zielt auf ~70-90%% der 96 GB VRAM "
            "(NVIDIA RTX Pro 6000 Q Max)"
        ),
    )
    args = parser.parse_args()
    zones = SUBSAMPLE_ZONES[args.subsample_level]

    # ------------------------------------------------------------------
    # Konfiguration laden
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    gen = cfg["general"]
    seed = gen["random_seed"]
    rng = np.random.default_rng(seed)
    s3_cfg = cfg.get("s3", {})
    s3_enabled = s3_cfg.get("enabled", False)

    base_path = os.getcwd()
    results_dir = os.path.join(base_path, gen["results_dir"])
    vtk_base_dir = os.path.join(results_dir, "vtks")
    output_dir = os.path.join(results_dir,
                               f"graph_dataset_{args.subsample_level}")
    csv_path = os.path.join(results_dir, "dataset_overview.csv")

    # ------------------------------------------------------------------
    # dataset_overview.csv von S3 herunterladen (falls nicht lokal)
    # ------------------------------------------------------------------
    if s3_enabled and not os.path.exists(csv_path):
        s3_csv = (f"s3://{s3_cfg['bucket']}/{s3_cfg['prefix']}/"
                  f"dataset_overview.csv")
        print(f"   Lade {s3_csv} herunter...")
        os.makedirs(results_dir, exist_ok=True)
        ok = download_from_s3(s3_csv, csv_path, s3_cfg)
        if not ok:
            print("FEHLER: dataset_overview.csv konnte nicht von S3 "
                  "heruntergeladen werden.")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Simulationsmetadaten laden
    # ------------------------------------------------------------------
    simulations = load_simulation_metadata(csv_path)
    n_sims = len(simulations)

    if n_sims == 0:
        print("FEHLER: Keine gueltigen Simulationen in der CSV gefunden.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Uebersicht
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"   VTK -> GRAPH DATENSATZ")
    print(f"{'=' * 60}")
    print(f"")
    print(f"   Simulationen:      {n_sims}")
    print(f"   Subsample-Level:   {args.subsample_level.upper()}")
    print(f"   VTK-Quelle:        ", end="")
    if s3_enabled:
        print(f"s3://{s3_cfg['bucket']}/{s3_cfg['prefix']}/")
    else:
        print(f"{vtk_base_dir}")
    print(f"   Ausgabe:           {output_dir}")
    if s3_enabled:
        graph_prefix_s3 = (s3_cfg.get("graph_prefix", "graph-dataset")
                           + f"_{args.subsample_level}")
        print(f"   S3-Upload:         s3://{s3_cfg['bucket']}/"
              f"{graph_prefix_s3}/")
    print(f"   k-NN Nachbarn:     {args.k_neighbors}")
    print(f"   Random Seed:       {seed}")
    print(f"   Subsampling-Zonen:")
    prev = 0.0
    for max_d, ratio in zones:
        label = f"{prev:.0f}-{max_d:.0f}m" if max_d != np.inf else f">{prev:.0f}m"
        print(f"     {label:>12s}: {ratio*100:5.2f}% behalten")
        prev = max_d
    print(f"")
    print(f"{'=' * 60}")
    print(f"   STARTE KONVERTIERUNG")
    print(f"{'=' * 60}")
    print(f"")

    # ------------------------------------------------------------------
    # Simulationen verarbeiten
    # ------------------------------------------------------------------
    start_time = time.time()
    data_list = []
    sim_stats_list = []   # fuer subsample_report
    n_success = 0
    n_failed = 0
    total_nodes_original = 0
    total_nodes_subsampled = 0
    total_edges = 0

    for i, sim in enumerate(simulations):
        sim_start = time.time()
        sim_id = sim["id"]
        vtk_downloaded = False

        # Fortschritt
        msg = f"\r   [{i+1}/{n_sims}] {sim_id}: Verarbeite..."
        sys.stdout.write(f"{msg:<70}")
        sys.stdout.flush()

        try:
            # S3: VTK herunterladen und entpacken
            vtk_dir = os.path.join(vtk_base_dir, f"{sim_id}_VTK")
            if s3_enabled and not os.path.isdir(vtk_dir):
                sys.stdout.write(
                    f"\r   [{i+1}/{n_sims}] {sim_id}: "
                    f"Lade von S3...{' ' * 30}")
                sys.stdout.flush()
                download_and_extract_vtk(sim_id, s3_cfg, vtk_base_dir)
                vtk_downloaded = True

            data, stats = process_simulation(
                sim, vtk_base_dir, rng,
                k_neighbors=args.k_neighbors,
                zones=zones)

            sim_time = time.time() - sim_start
            data_list.append(data)
            n_success += 1
            total_nodes_original += stats["n_original"]
            total_nodes_subsampled += stats["n_subsampled"]
            total_edges += stats["n_edges"]

            sim_stats_list.append({
                "sim_id":       sim_id,
                "n_original":   stats["n_original"],
                "n_subsampled": stats["n_subsampled"],
                "n_edges":      stats["n_edges"],
                "time_s":       sim_time,
            })

            ratio = stats["n_subsampled"] / stats["n_original"] * 100
            print(f"\r   [OK] [{i+1}/{n_sims}] {sim_id}: "
                  f"{stats['n_original']:,} -> {stats['n_subsampled']:,} "
                  f"Knoten ({ratio:.1f}%), "
                  f"{stats['n_edges']:,} Kanten, "
                  f"{sim_time:.1f}s"
                  f"{' ' * 10}")

        except Exception as e:
            n_failed += 1
            sim_time = time.time() - sim_start
            print(f"\r   [XX] [{i+1}/{n_sims}] {sim_id}: "
                  f"FEHLER - {e} ({sim_time:.1f}s)"
                  f"{' ' * 10}")

        finally:
            # Lokale VTK-Dateien sofort loeschen (Speicher sparen)
            if s3_enabled and vtk_downloaded:
                vtk_dir = os.path.join(vtk_base_dir, f"{sim_id}_VTK")
                if os.path.isdir(vtk_dir):
                    shutil.rmtree(vtk_dir)

    # ------------------------------------------------------------------
    # Train / Val / Test Split
    # ------------------------------------------------------------------
    if len(data_list) == 0:
        print("\nFEHLER: Kein einziger Graph erfolgreich erstellt.")
        sys.exit(1)

    print(f"\n   Erstelle Train/Val/Test Split (70/15/15)...")
    train, val, test = split_dataset(data_list, seed=seed)

    # ------------------------------------------------------------------
    # Speichern
    # ------------------------------------------------------------------
    print(f"   Speichere nach {output_dir}...")
    save_dataset(train, val, test, output_dir,
                 subsample_level=args.subsample_level, zones=zones)

    # ------------------------------------------------------------------
    # Subsampling-Report schreiben
    # ------------------------------------------------------------------
    total_time_so_far = time.time() - start_time
    report_path = write_subsample_report(
        output_dir=output_dir,
        subsample_level=args.subsample_level,
        zones=zones,
        sim_stats=sim_stats_list,
        total_time=total_time_so_far,
        k_neighbors=args.k_neighbors,
    )
    print(f"   Subsampling-Report: {report_path}")

    # ------------------------------------------------------------------
    # S3-Upload der Graph-Dateien
    # ------------------------------------------------------------------
    if s3_enabled:
        graph_prefix = (s3_cfg.get("graph_prefix", "graph-dataset")
                        + f"_{args.subsample_level}")
        s3_base = f"s3://{s3_cfg['bucket']}/{graph_prefix}"
        print(f"\n   Lade Graph-Datensatz nach {s3_base}/ hoch...")

        upload_files = ["train.pt", "val.pt", "test.pt",
                        "metadata.yaml", "subsample_report.yaml"]
        all_ok = True
        for fname in upload_files:
            local = os.path.join(output_dir, fname)
            if not os.path.exists(local):
                print(f"   WARNUNG: {fname} nicht gefunden, ueberspringe.")
                continue
            s3_dest = f"{s3_base}/{fname}"
            ok = upload_to_s3(local, s3_dest, s3_cfg)
            if ok:
                size_mb = os.path.getsize(local) / (1024 * 1024)
                print(f"   [S3 OK] {fname} ({size_mb:.1f} MB)")
            else:
                print(f"   [S3 FEHLER] {fname}")
                all_ok = False

        if all_ok:
            # Lokale Graph-Dateien aufraeumen
            shutil.rmtree(output_dir)
            print(f"   Lokale Dateien aufgeraeumt.")
        else:
            print(f"   WARNUNG: Nicht alle Dateien hochgeladen. "
                  f"Lokale Dateien behalten in {output_dir}")

    # ------------------------------------------------------------------
    # Zusammenfassung
    # ------------------------------------------------------------------
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    # Dategroessen
    total_size_bytes = 0
    for fname in ["train.pt", "val.pt", "test.pt"]:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            total_size_bytes += os.path.getsize(fpath)
    total_size_mb = total_size_bytes / (1024 * 1024)

    avg_nodes = total_nodes_subsampled // max(n_success, 1)
    avg_edges = total_edges // max(n_success, 1)

    print(f"\n{'=' * 60}")
    print(f"   ZUSAMMENFASSUNG")
    print(f"{'=' * 60}")
    print(f"")
    print(f"   Erfolgreich:       {n_success}/{n_sims}")
    print(f"   Fehlgeschlagen:    {n_failed}/{n_sims}")
    print(f"")
    print(f"   Train:             {len(train)} Graphen")
    print(f"   Validation:        {len(val)} Graphen")
    print(f"   Test:              {len(test)} Graphen")
    print(f"")
    print(f"   Input-Features:    14 pro Knoten")
    print(f"   Output-Features:   6 pro Knoten")
    print(f"   k-NN Nachbarn:     {args.k_neighbors}")
    print(f"")
    print(f"   Orig. Knoten:      {total_nodes_original:,} gesamt")
    print(f"   Subsampled:        {total_nodes_subsampled:,} gesamt")
    print(f"   Reduktion:         {total_nodes_subsampled/max(total_nodes_original,1)*100:.1f}%")
    print(f"   Avg. Knoten/Graph: {avg_nodes:,}")
    print(f"   Avg. Kanten/Graph: {avg_edges:,}")
    print(f"")
    if s3_enabled:
        graph_prefix = (s3_cfg.get("graph_prefix", "graph-dataset")
                        + f"_{args.subsample_level}")
        print(f"   Datensatzgroesse:  {total_size_mb:.1f} MB")
        print(f"   Laufzeit:          {minutes}min {seconds}s")
        print(f"   S3-Ausgabe:        s3://{s3_cfg['bucket']}/{graph_prefix}/")
    else:
        print(f"   Datensatzgroesse:  {total_size_mb:.1f} MB")
        print(f"   Laufzeit:          {minutes}min {seconds}s")
        print(f"   Ausgabe:           {output_dir}")
        print(f"   Report:            {output_dir}/subsample_report.yaml")
    print(f"")
    print(f"{'=' * 60}")
    print(f"   FERTIG")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()