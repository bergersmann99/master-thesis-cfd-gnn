"""
Batch-Interpolation: alle Netzwerke auf alle Test-Sims interpolieren.

Erwartet existierende sparse VTU-Dateien in:
  evaluation_{network}/sim_XXX/{vorhersage,ground_truth}.vtu

Erstellt pro (Netzwerk, Sim) eine vollst.ndige interpolierte VTU.
"""
import os
import sys
import subprocess
import numpy as np
import pyvista as pv

NETWORKS = ['gatv2_coarse', 'gatv2_medium', 'gatv2_bf25',
            'gcn_coarse', 'gcn_medium', 'gcn_bf25']
SIMS = ['sim_001', 'sim_013', 'sim_014']
ITERATIONS = {'sim_001': '1227', 'sim_013': '1151', 'sim_014': '1243'}

BASE = '/home/tbergermann/Python/predictions'
TMP = '/tmp'

VTK_SUBDIRS = {
    'sim_001': '/tmp/sim_001_extracted/sim_001/sim_001_1227',
    'sim_013': '/tmp/sim_013_extracted/sim_013/sim_013_1151',
    'sim_014': '/tmp/sim_014_extracted/sim_014_VTK/sim_014_1243',
}

INTERP_SCRIPT = '/home/tbergermann/Python/interpolate_to_full_mesh.py'
PYTHON = '/home/tbergermann/Python/venv/bin/python'


def extract_npy(eval_dir, sim, out_dir):
    pred_path = os.path.join(eval_dir, sim, 'vorhersage.vtu')
    gt_path = os.path.join(eval_dir, sim, 'ground_truth.vtu')
    pred_mesh = pv.read(pred_path)
    gt_mesh = pv.read(gt_path)

    def stack(m):
        U = np.asarray(m['U'], dtype=np.float32)
        return np.column_stack([U[:,0], U[:,1], U[:,2],
                                np.asarray(m['p'], dtype=np.float32),
                                np.asarray(m['k'], dtype=np.float32),
                                np.asarray(m['epsilon'], dtype=np.float32)]).astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'positions.npy'),
            np.asarray(pred_mesh.points, dtype=np.float32))
    np.save(os.path.join(out_dir, 'prediction.npy'), stack(pred_mesh))
    np.save(os.path.join(out_dir, 'true.npy'), stack(gt_mesh))


def main():
    only_network = sys.argv[1] if len(sys.argv) > 1 else None
    only_sim = sys.argv[2] if len(sys.argv) > 2 else None

    for net in NETWORKS:
        if only_network and net != only_network:
            continue
        eval_dir = os.path.join(BASE, f'evaluation_{net}')
        if not os.path.isdir(eval_dir):
            print(f"[SKIP] {net}: kein eval-Verzeichnis")
            continue

        for sim in SIMS:
            if only_sim and sim != only_sim:
                continue
            sparse_sim = os.path.join(eval_dir, sim)
            if not os.path.isdir(sparse_sim):
                print(f"[SKIP] {net}/{sim}: keine sparse VTU")
                continue

            print(f"\n=== {net} / {sim} ===")
            npy_dir = os.path.join(TMP, f'{net}_{sim}_npy')
            print(f"  npy extrahieren ...")
            extract_npy(eval_dir, sim, npy_dir)

            mesh_path = os.path.join(VTK_SUBDIRS[sim], 'internal.vtu')
            output_dir = os.path.join(BASE, 'vorhersagen_test_sims', sim)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{net}.vtu')

            print(f"  IDW-Interpolation ...")
            cmd = [PYTHON, INTERP_SCRIPT,
                   '--pos', os.path.join(npy_dir, 'positions.npy'),
                   '--pred', os.path.join(npy_dir, 'prediction.npy'),
                   '--true', os.path.join(npy_dir, 'true.npy'),
                   '--mesh', mesh_path,
                   '--output', output_path,
                   '--method', 'idw', '--k-neighbors', '8']
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  FEHLER: {r.stderr}")
                continue
            size_gb = os.path.getsize(output_path) / 1e9
            print(f"  -> {output_path} ({size_gb:.2f} GB)")


if __name__ == '__main__':
    main()
