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
import argparse
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

# Subsampling-Parameter (wandabstandsbasiert)
SUBSAMPLE_ZONES = [
    # (max_wall_distance, keep_ratio)
    (2.0,    0.10),    # Gebaeudenah: 10% (Abloesung, Rezirkulation)
    (10.0,   0.02),    # Nahfeld: 2% (Nachlauf, Wirbelstrukturen)
    (50.0,   0.005),   # Mittelfeld: 0.5% (Uebergangszone)
    (np.inf, 0.002),   # Fernfeld: 0.2% (nahezu Freistrom)
]


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

def adaptive_subsample(cell_centers, wall_distances, rng):
    """
    Fuehrt wandabstandsbasiertes Subsampling durch.

    Zonen (aus SUBSAMPLE_ZONES):
        - wall_distance < 2 m:     10% (Gebaeudenah)
        - wall_distance 2-10 m:    2% (Nahfeld)
        - wall_distance 10-50 m:   0.5% (Mittelfeld)
        - wall_distance > 50 m:    0.2% (Fernfeld)

    Parameter
    ---------
    cell_centers : np.ndarray
        (N, 3) Zellzentren.
    wall_distances : np.ndarray
        (N,) Wandabstaende.
    rng : np.random.Generator
        Zufallsgenerator (fuer reproduzierbares Sampling).

    Rueckgabe
    ---------
    np.ndarray : Indizes der behaltenen Zellen (sortiert)
    """
    n_total = len(cell_centers)
    keep_mask = np.zeros(n_total, dtype=bool)

    prev_limit = 0.0
    for max_dist, keep_ratio in SUBSAMPLE_ZONES:
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

def process_simulation(sim_meta, vtk_base_dir, rng, k_neighbors=20):
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
    keep_indices = adaptive_subsample(cell_centers, wall_distances, rng)
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

def save_dataset(train, val, test, output_dir):
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
            for z in SUBSAMPLE_ZONES
        ],
    }

    with open(os.path.join(output_dir, "metadata.yaml"), "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)


# ======================================================================
# Hauptprogramm
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VTK -> PyTorch Geometric Graph-Datensatz"
    )
    parser.add_argument(
        "--config", default="test_config.yaml",
        help="Pfad zur Konfigurationsdatei (Standard: config.yaml)"
    )
    parser.add_argument(
        "--k-neighbors", type=int, default=20,
        help="Anzahl k-NN Nachbarn fuer Graph-Kanten (Standard: 20)"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Konfiguration laden
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    gen = cfg["general"]
    seed = gen["random_seed"]
    rng = np.random.default_rng(seed)

    base_path = os.getcwd()
    results_dir = os.path.join(base_path, gen["results_dir"])
    vtk_base_dir = os.path.join(results_dir, "vtks")
    output_dir = os.path.join(results_dir, "graph_dataset")
    csv_path = os.path.join(results_dir, "dataset_overview.csv")

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
    print(f"   VTK-Verzeichnis:   {vtk_base_dir}")
    print(f"   Ausgabe:           {output_dir}")
    print(f"   k-NN Nachbarn:     {args.k_neighbors}")
    print(f"   Random Seed:       {seed}")
    print(f"   Subsampling-Zonen:")
    prev = 0.0
    for max_d, ratio in SUBSAMPLE_ZONES:
        label = f"{prev:.0f}-{max_d:.0f}m" if max_d != np.inf else f">{prev:.0f}m"
        print(f"     {label:>12s}: {ratio*100:5.1f}% behalten")
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
    n_success = 0
    n_failed = 0
    total_nodes_original = 0
    total_nodes_subsampled = 0
    total_edges = 0

    for i, sim in enumerate(simulations):
        sim_start = time.time()
        sim_id = sim["id"]

        # Fortschritt
        msg = f"\r   [{i+1}/{n_sims}] {sim_id}: Verarbeite..."
        sys.stdout.write(f"{msg:<70}")
        sys.stdout.flush()

        try:
            data, stats = process_simulation(
                sim, vtk_base_dir, rng, k_neighbors=args.k_neighbors)

            data_list.append(data)
            n_success += 1
            total_nodes_original += stats["n_original"]
            total_nodes_subsampled += stats["n_subsampled"]
            total_edges += stats["n_edges"]

            sim_time = time.time() - sim_start
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
    save_dataset(train, val, test, output_dir)

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
    print(f"   Datensatzgroesse:  {total_size_mb:.1f} MB")
    print(f"   Laufzeit:          {minutes}min {seconds}s")
    print(f"   Ausgabe:           {output_dir}")
    print(f"")
    print(f"{'=' * 60}")
    print(f"   FERTIG")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()