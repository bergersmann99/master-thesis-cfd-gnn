"""
Räumliche Interpolation von GNN-Vorhersagen auf das vollständige CFD-Netz.

Das GNN trifft Vorhersagen nur an den ~507k subsampled Knoten.
Dieses Skript interpoliert diese sparse Vorhersagen auf das vollständige
CFD-Simulationsgitter (~8M Punkte) mittels Inverse Distance Weighting (IDW)
oder Nearest-Neighbor-Interpolation.

Eingabe:
  --pos   : NumPy .npy  — GNN-Knotenpositionen        (N × 3)
  --pred  : NumPy .npy  — GNN-Vorhersagen             (N × 6)
  --mesh  : vollständiges CFD-Netz als .vtu/.vtk/.pt  (~8M Punkte)

Ausgabe:
  VTU-Datei mit interpolierten Feldern auf dem vollen Netz

Verwendung (nach predict.py):
  python interpolate_to_full_mesh.py \
    --pos   predictions/sim_001_GCN_pos.npy \
    --pred  predictions/sim_001_GCN_pred.npy \
    --mesh  /path/to/full_mesh.vtu \
    --output predictions/sim_001_GCN_full_mesh.vtu

Verwendung mit Ground-Truth-Vergleich (Fehler auf Vollnetz):
  python interpolate_to_full_mesh.py \
    --pos   predictions/sim_001_GCN_pos.npy \
    --pred  predictions/sim_001_GCN_pred.npy \
    --true  predictions/sim_001_GCN_true.npy \
    --mesh  /path/to/full_mesh.vtu \
    --output predictions/sim_001_GCN_full_mesh.vtu

Methoden:
  idw     : Inverse Distance Weighting (Standard, genauer, etwas langsamer)
  nearest : Nächster Nachbar           (schnellste Option)

Hinweis Laufzeit:
  KD-Tree Aufbau (507k Punkte):     ~0.5s
  IDW Abfrage    (8M Punkte, k=8):  ~60–120s
  Nearest Neighbor (8M Punkte):     ~10–20s
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import pyvista as pv
from scipy.spatial import KDTree

FIELD_NAMES = ["Ux", "Uy", "Uz", "p", "k", "epsilon"]


# ── Interpolationsmethoden ────────────────────────────────────────────────────

def idw_interpolate(tree: KDTree,
                    source_values: np.ndarray,
                    query_points: np.ndarray,
                    k: int = 8,
                    power: float = 2.0) -> np.ndarray:
    """
    Inverse Distance Weighting (IDW).

    Für jeden Zielpunkt: gewichteter Mittelwert der k nächsten GNN-Vorhersagen.
    Gewicht w_i = 1 / dist_i^power — nahe Punkte haben stärkeren Einfluss.

    Falls ein Zielpunkt exakt auf einem GNN-Knoten liegt (dist = 0),
    wird der GNN-Wert direkt übernommen (kein Div-by-Zero).

    Args:
        tree          : KDTree der GNN-Knotenpositionen (Stützpunkte)
        source_values : GNN-Vorhersagen          (N_src × 6)
        query_points  : Vollnetz-Positionen       (N_dst × 3)
        k             : Anzahl nächster Nachbarn
        power         : Gewichtungsexponent (typisch: 1–3)

    Returns:
        interpolierte Werte (N_dst × 6)
    """
    distances, indices = tree.query(query_points, k=k, workers=-1)

    # Division-by-Zero vermeiden
    eps = 1e-12
    distances = np.maximum(distances, eps)

    weights = 1.0 / distances ** power           # (M, k)
    weights /= weights.sum(axis=1, keepdims=True)  # normalisieren → Summe = 1

    # source_values[indices]: (M, k, 6)
    # weights[:, :, None]:    (M, k, 1)
    interpolated = (weights[:, :, None] * source_values[indices]).sum(axis=1)  # (M, 6)
    return interpolated


def nearest_neighbor_interpolate(tree: KDTree,
                                  source_values: np.ndarray,
                                  query_points: np.ndarray) -> np.ndarray:
    """
    Nächster-Nachbar-Interpolation.
    Jeder Zielpunkt bekommt den Wert des nächstgelegenen GNN-Knotens.
    Schnellste Methode, leichte Blockartefakte möglich.
    """
    _, indices = tree.query(query_points, k=1, workers=-1)
    return source_values[indices.ravel()]


# ── Mesh-Loader ───────────────────────────────────────────────────────────────

def load_full_mesh(mesh_path: Path) -> tuple[np.ndarray, pv.DataSet | None]:
    """
    Liest das vollständige CFD-Netz.

    Unterstützte Formate:
      .vtu / .vtk  — direkter PyVista-Import (Topologie bleibt erhalten)
      .pt          — PyG Data-Objekt mit .pos (nur Punktpositionen)

    Returns:
        (positions, pyvista_mesh)
        pyvista_mesh ist None bei .pt-Eingabe.
    """
    suffix = mesh_path.suffix.lower()

    if suffix in (".vtu", ".vtk", ".pvtu"):
        mesh = pv.read(str(mesh_path))
        pos  = np.asarray(mesh.points, dtype=np.float32)
        print(f"  VTK-Netz gelesen: {pos.shape[0]:,} Punkte")
        return pos, mesh

    elif suffix == ".pt":
        data = torch.load(str(mesh_path), weights_only=False)
        if isinstance(data, list):
            print(f"  Liste mit {len(data)} Graphen — nehme Index 0")
            data = data[0]
        pos = data.pos.numpy().astype(np.float32)
        print(f"  PyG-Graph gelesen: {pos.shape[0]:,} Knoten")
        return pos, None

    else:
        raise ValueError(
            f"Unbekanntes Mesh-Format: '{suffix}'. "
            f"Unterstützt: .vtu, .vtk, .pvtu, .pt"
        )


# ── Hauptprogramm ─────────────────────────────────────────────────────────────

def main():
    """Parst CLI-Argumente, interpoliert die Vorhersage und schreibt die VTU-Ausgabe."""
    parser = argparse.ArgumentParser(
        description="Interpoliert GNN-Vorhersagen auf das vollständige CFD-Netz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pos",    required=True,  type=str,
                        help="NumPy .npy — GNN-Knotenpositionen (N×3)")
    parser.add_argument("--pred",   required=True,  type=str,
                        help="NumPy .npy — GNN-Vorhersagen (N×6)")
    parser.add_argument("--mesh",   required=True,  type=str,
                        help="Vollständiges CFD-Netz (.vtu/.vtk oder .pt)")
    parser.add_argument("--output", required=True,  type=str,
                        help="Ausgabedatei (.vtu)")
    parser.add_argument("--true",   type=str,       default=None,
                        help="Optional: .npy mit Ground Truth (N×6) für Fehlerberechnung")
    parser.add_argument("--method", choices=["idw", "nearest"], default="idw",
                        help="Interpolationsmethode: idw (Standard) oder nearest")
    parser.add_argument("--k-neighbors", type=int,   default=8,
                        help="Anzahl Nachbarn für IDW (Standard: 8)")
    parser.add_argument("--idw-power",   type=float, default=2.0,
                        help="IDW Gewichtungsexponent (Standard: 2.0)")
    args = parser.parse_args()

    # ── GNN-Daten laden ───────────────────────────────────────────────────────
    print("\nLade GNN-Vorhersagen...")
    src_pos  = np.load(args.pos).astype(np.float32)   # (N, 3)
    src_pred = np.load(args.pred).astype(np.float32)  # (N, 6)
    print(f"  GNN-Stützpunkte: {src_pos.shape[0]:,}")

    src_true = None
    if args.true:
        src_true = np.load(args.true).astype(np.float32)
        print(f"  Ground Truth geladen: {src_true.shape}")

    # ── Vollnetz laden ────────────────────────────────────────────────────────
    print("\nLade vollständiges Netz...")
    full_pos, full_mesh = load_full_mesh(Path(args.mesh))
    print(f"  Zielpunkte: {full_pos.shape[0]:,}")

    ratio = full_pos.shape[0] / src_pos.shape[0]
    print(f"  Auflösungsverhältnis: {ratio:.1f}×  "
          f"({src_pos.shape[0]:,} → {full_pos.shape[0]:,} Punkte)")

    # ── KD-Tree aufbauen ──────────────────────────────────────────────────────
    print("\nBaue KD-Tree auf...")
    tree = KDTree(src_pos)
    print("  KD-Tree bereit.")

    # ── Interpolation der Vorhersage ──────────────────────────────────────────
    print(f"\nInterpoliere Vorhersage ({args.method.upper()}"
          + (f", k={args.k_neighbors}" if args.method == "idw" else "") + ")...")

    if args.method == "idw":
        interp_pred = idw_interpolate(
            tree, src_pred, full_pos,
            k=args.k_neighbors, power=args.idw_power
        )
    else:
        interp_pred = nearest_neighbor_interpolate(tree, src_pred, full_pos)

    print(f"  Fertig: {interp_pred.shape}")

    # ── Ausgabe-Netz aufbauen ─────────────────────────────────────────────────
    print("\nErstelle VTU-Ausgabe...")
    if full_mesh is not None:
        # Originale Zelltopologie erhalten (Verbindungen, Netzstruktur)
        out_mesh = full_mesh.copy()
    else:
        # Nur Punktwolke (aus .pt)
        out_mesh = pv.PolyData(full_pos)

    # Interpolierte Felder einfügen
    for i, name in enumerate(FIELD_NAMES):
        out_mesh.point_data[name] = interp_pred[:, i]

    # ── Fehlerfelder (Ground Truth → interpolieren → Differenz) ──────────────
    if src_true is not None:
        print("Interpoliere Ground Truth für Fehlerberechnung...")
        if args.method == "idw":
            interp_true = idw_interpolate(
                tree, src_true, full_pos,
                k=args.k_neighbors, power=args.idw_power
            )
        else:
            interp_true = nearest_neighbor_interpolate(tree, src_true, full_pos)

        for i, name in enumerate(FIELD_NAMES):
            abs_err = np.abs(interp_pred[:, i] - interp_true[:, i])
            rel_err = abs_err / (np.abs(interp_true[:, i]) + 1e-8)
            out_mesh.point_data[f"{name}_abs_error"] = abs_err
            out_mesh.point_data[f"{name}_rel_error"] = rel_err

        # Ground Truth auch im Output (für direkten Vergleich in ParaView)
        for i, name in enumerate(FIELD_NAMES):
            out_mesh.point_data[f"{name}_true"] = interp_true[:, i]

    # ── Speichern ─────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_mesh.save(str(out_path))

    print(f"\n✓ Gespeichert: {out_path}")
    print(f"  Punkte im Ausgabe-Netz: {full_pos.shape[0]:,}")
    print(f"  Felder: {', '.join(FIELD_NAMES)}")
    if src_true is not None:
        print(f"  + Fehlerfelder: *_abs_error, *_rel_error, *_true")
    print(f"\n  In ParaView öffnen: File → Open → {out_path.name}")
    print("  Tipp: 'Surface' Darstellung mit 'Ux' oder 'p' färben")


if __name__ == "__main__":
    main()
