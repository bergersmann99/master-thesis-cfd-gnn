"""
Interpoliere die GCN-RERUN-Vorhersagen für sim_013 auf das Vollnetz (~8M Punkte)
und ersetze die veralteten gcn_*.vtu auf S3.

Quelle (sparse): prediction_sim_013_gcn_rerun/gcn_<res>/{vorhersage,ground_truth}.vtu
Vollnetz       : /tmp/sim_013_extracted/sim_013/sim_013_1151/internal.vtu
Ziel lokal     : prediction_sim_013_gcn_rerun/full_mesh/gcn_<res>.vtu
Ziel S3        : predictions/vorhersagen_test_sims/sim_013/gcn_<res>.vtu
"""
import os, subprocess
import numpy as np
import pyvista as pv
import boto3

BASE = '/home/tbergermann/Python/predictions/prediction_sim_013_gcn_rerun'
MESH = '/tmp/sim_013_extracted/sim_013/sim_013_1151/internal.vtu'
INTERP = '/home/tbergermann/Python/interpolate_to_full_mesh.py'
PYTHON = '/home/tbergermann/Python/venv/bin/python'
OUT_LOCAL = os.path.join(BASE, 'full_mesh')
BUCKET = 'amzn-master-sim-bucket'
S3_PREFIX = 'predictions/vorhersagen_test_sims/sim_013'
RESOLUTIONS = ['coarse', 'medium', 'bf25']

s3 = boto3.client('s3')
cfg = boto3.s3.transfer.TransferConfig(multipart_threshold=64*1024*1024,
                                       multipart_chunksize=64*1024*1024)


def stack(m):
    """Stapelt U, p, k und epsilon eines Meshes zu einem (N, 6)-Array."""
    U = np.asarray(m['U'], dtype=np.float32)
    return np.column_stack([U[:, 0], U[:, 1], U[:, 2],
                            np.asarray(m['p'], dtype=np.float32),
                            np.asarray(m['k'], dtype=np.float32),
                            np.asarray(m['epsilon'], dtype=np.float32)]).astype(np.float32)


def extract_npy(res, tmp_dir):
    """Extrahiert Positionen, Vorhersage und Ground Truth der Auflösung res nach .npy."""
    src = os.path.join(BASE, f'gcn_{res}')
    pred_mesh = pv.read(os.path.join(src, 'vorhersage.vtu'))
    gt_mesh = pv.read(os.path.join(src, 'ground_truth.vtu'))
    os.makedirs(tmp_dir, exist_ok=True)
    np.save(os.path.join(tmp_dir, 'positions.npy'),
            np.asarray(pred_mesh.points, dtype=np.float32))
    np.save(os.path.join(tmp_dir, 'prediction.npy'), stack(pred_mesh))
    np.save(os.path.join(tmp_dir, 'true.npy'), stack(gt_mesh))


def main():
    """Interpoliert alle GCN-Rerun-Auflösungen auf das Vollnetz und lädt sie nach S3."""
    os.makedirs(OUT_LOCAL, exist_ok=True)
    for res in RESOLUTIONS:
        print(f"\n=== gcn_{res} / sim_013 ===", flush=True)
        tmp_dir = f'/tmp/gcn_{res}_sim_013_npy'
        print("  npy extrahieren ...", flush=True)
        extract_npy(res, tmp_dir)
        out_path = os.path.join(OUT_LOCAL, f'gcn_{res}.vtu')
        print("  IDW-Interpolation auf Vollnetz ...", flush=True)
        cmd = [PYTHON, INTERP,
               '--pos', os.path.join(tmp_dir, 'positions.npy'),
               '--pred', os.path.join(tmp_dir, 'prediction.npy'),
               '--true', os.path.join(tmp_dir, 'true.npy'),
               '--mesh', MESH,
               '--output', out_path,
               '--method', 'idw', '--k-neighbors', '8']
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FEHLER:\n{r.stdout}\n{r.stderr}", flush=True)
            continue
        size_gb = os.path.getsize(out_path) / 1e9
        print(f"  -> {out_path} ({size_gb:.2f} GB)", flush=True)
        key = f'{S3_PREFIX}/gcn_{res}.vtu'
        print(f"  Upload nach s3://{BUCKET}/{key} ...", flush=True)
        s3.upload_file(out_path, BUCKET, key, Config=cfg)
        print("  Upload fertig.", flush=True)
    print("\nAlle Auflösungen fertig.", flush=True)


if __name__ == '__main__':
    main()
