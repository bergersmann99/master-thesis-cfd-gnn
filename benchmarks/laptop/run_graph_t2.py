"""
T2-Teilmessung — Graph-Konstruktion fuer EINEN Extrapolationsfall
auf Stufe MEDIUM, mit Wandclock-Zeit je Stufe.

Repliziert process_simulation aus createGraphDataset.py mit Timing,
verwendet identische Helper-Funktionen, keine Modifikationen.
"""
import os
import sys
import time
import argparse

REAL_DIR = "/home/tim-bergermann/laptop_timing/scripts"
sys.path.insert(0, REAL_DIR)

import yaml
import numpy as np
import torch
from torch_geometric.data import Data

import createGraphDataset as CGD

VTK_BASE        = "/home/tim-bergermann/laptop_timing/vtks"
OUTPUT_DIR      = "/home/tim-bergermann/laptop_timing/graphs"
K_NEIGHBORS     = 20
SUBSAMPLE_LEVEL = "medium"
SEED            = 42


def main():
    """Fuehrt die Graph-Konstruktion mit Timing je Stufe aus und schreibt optional ein YAML."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-id",      default="sturm_25ms_45deg",
                    help="Simulations-ID (z.B. sturm_25ms_45deg)")
    ap.add_argument("--u-ref",       type=float, default=25.0,
                    help="Referenzwindgeschwindigkeit [m/s]")
    ap.add_argument("--angle",       type=float, default=45.0,
                    help="Windwinkel [Grad]")
    ap.add_argument("--output-yaml", default=None,
                    help="Pfad zum YAML-Timing-Output (optional)")
    args = ap.parse_args()

    SIM_ID = args.sim_id
    U_REF  = args.u_ref
    ANGLE  = args.angle

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    zones = CGD.SUBSAMPLE_ZONES[SUBSAMPLE_LEVEL]
    vtk_dir = os.path.join(VTK_BASE, f"{SIM_ID}_VTK")
    print(f"Sim: {SIM_ID}, Level: {SUBSAMPLE_LEVEL}, k={K_NEIGHBORS}")
    print(f"VTK: {vtk_dir}")
    print(f"Out: {OUTPUT_DIR}")
    print("-" * 60)

    timings = {}

    # 1. VTK laden -----------------------------------------------------
    t0 = time.time()
    mesh = CGD.load_volume_mesh(vtk_dir)
    patches = CGD.load_boundary_patches(vtk_dir)
    timings["1_vtk_load"] = time.time() - t0
    print(f"[1] VTK laden:          {timings['1_vtk_load']:.2f}s  "
          f"(mesh: {mesh.n_cells} cells, patches: {list(patches.keys())})")

    # 2. Features aus vollem Mesh -------------------------------------
    t0 = time.time()
    cell_centers = CGD.extract_cell_centers(mesh)
    cell_volumes = CGD.extract_cell_volumes(mesh)
    fields = CGD.extract_flow_fields(mesh)
    timings["2_features"] = time.time() - t0
    n_original = len(cell_centers)
    print(f"[2] Feature-Extract:    {timings['2_features']:.2f}s  "
          f"(n_original={n_original:,})")

    # 3. Wall-Distance -------------------------------------------------
    t0 = time.time()
    wall_distances = CGD.compute_wall_distance(cell_centers, patches)
    timings["3_wall_distance"] = time.time() - t0
    print(f"[3] Wall-Distance:      {timings['3_wall_distance']:.2f}s  "
          f"(min={wall_distances.min():.3f}, max={wall_distances.max():.2f})")

    # 4. Node-Types ----------------------------------------------------
    t0 = time.time()
    node_types = CGD.compute_node_types(cell_centers, patches, mesh)
    timings["4_node_types"] = time.time() - t0
    print(f"[4] Node-Types:         {timings['4_node_types']:.2f}s  "
          f"(shape={node_types.shape})")

    # 5. Adaptives Subsampling ----------------------------------------
    t0 = time.time()
    keep_indices = CGD.adaptive_subsample(cell_centers, wall_distances, rng,
                                          zones=zones)
    timings["5_subsample"] = time.time() - t0
    n_subsampled = len(keep_indices)
    print(f"[5] Subsample (medium): {timings['5_subsample']:.2f}s  "
          f"(n_sub={n_subsampled:,}, "
          f"reduktion={100*n_subsampled/n_original:.2f}%)")

    sub_centers    = cell_centers[keep_indices]
    sub_volumes    = cell_volumes[keep_indices]
    sub_wall_dist  = wall_distances[keep_indices]
    sub_node_types = node_types[keep_indices]
    sub_U          = fields["U"][keep_indices]
    sub_p          = fields["p"][keep_indices]
    sub_k          = fields["k"][keep_indices]
    sub_eps        = fields["epsilon"][keep_indices]

    # 6. kNN-Graph -----------------------------------------------------
    t0 = time.time()
    edge_index = CGD.build_knn_edges(sub_centers, k=K_NEIGHBORS)
    timings["6_knn"] = time.time() - t0
    n_edges = edge_index.shape[1]
    print(f"[6] kNN-Kanten (k=20):  {timings['6_knn']:.2f}s  "
          f"(n_edges={n_edges:,})")

    # 7. Tensoren + Speichern -----------------------------------------
    t0 = time.time()
    n = n_subsampled
    global_features = np.column_stack([
        np.full(n, U_REF, dtype=np.float32),
        np.full(n, ANGLE, dtype=np.float32),
    ])
    x = np.column_stack([
        sub_centers, sub_wall_dist.reshape(-1, 1),
        sub_volumes.reshape(-1, 1), sub_node_types, global_features,
    ])
    y = np.column_stack([
        sub_U, sub_p.reshape(-1, 1),
        sub_k.reshape(-1, 1), sub_eps.reshape(-1, 1),
    ])
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        pos=torch.tensor(sub_centers, dtype=torch.float32),
    )
    data.sim_id = SIM_ID
    data.U_ref  = U_REF
    data.angle  = ANGLE
    out_path = os.path.join(OUTPUT_DIR, f"{SIM_ID}_medium.pt")
    torch.save([data], out_path)  # Liste — predict.py erwartet iterierbares Objekt
    timings["7_save"] = time.time() - t0
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[7] Tensoren+Speichern: {timings['7_save']:.2f}s  "
          f"({out_path}, {size_mb:.1f} MB)")

    total = sum(timings.values())
    print("-" * 60)
    print(f"GESAMT:                 {total:.2f}s")
    print()
    print("YAML-Zusammenfassung:")
    print("timings_seconds:")
    for k, v in timings.items():
        print(f"  {k}: {v:.3f}")
    print(f"  total: {total:.3f}")
    print(f"n_original: {n_original}")
    print(f"n_subsampled: {n_subsampled}")
    print(f"n_edges: {n_edges}")
    print(f"file_size_mb: {size_mb:.2f}")

    if args.output_yaml:
        timing_data = {
            "sim_id": SIM_ID, "u_ref": U_REF, "angle": ANGLE,
            "subsample_level": SUBSAMPLE_LEVEL, "k_neighbors": K_NEIGHBORS,
            "timings_seconds": timings,
            "total_seconds": total,
            "n_original": n_original,
            "n_subsampled": n_subsampled,
            "n_edges": n_edges,
            "file_size_mb": float(size_mb),
        }
        with open(args.output_yaml, "w") as f:
            yaml.safe_dump(timing_data, f, sort_keys=False)
        print(f"YAML gespeichert: {args.output_yaml}")


if __name__ == "__main__":
    main()
