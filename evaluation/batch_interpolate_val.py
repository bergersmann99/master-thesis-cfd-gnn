"""Val-Sim sim_012: alle 6 Netzwerke auf volles CFD-Mesh interpolieren."""
import os, subprocess, numpy as np, pyvista as pv

NETWORKS = ['gatv2_coarse', 'gatv2_medium', 'gatv2_bf25',
            'gcn_coarse', 'gcn_medium', 'gcn_bf25']
SIM = 'sim_012'
MESH = '/tmp/sim_012_extracted/sim_012_VTK/sim_012_1305/internal.vtu'
BASE = '/home/tbergermann/Python/predictions'
OUT = os.path.join(BASE, 'vorhersagen_val_sim_012')
TMP = '/tmp'
PYTHON = '/home/tbergermann/Python/venv/bin/python'
INTERP = '/home/tbergermann/Python/interpolate_to_full_mesh.py'

def extract(m):
    """Stapelt U, p, k und epsilon eines Meshes zu einem (N, 6)-Array."""
    U = np.asarray(m['U'], dtype=np.float32)
    return np.column_stack([U[:,0], U[:,1], U[:,2],
                            np.asarray(m['p'], dtype=np.float32),
                            np.asarray(m['k'], dtype=np.float32),
                            np.asarray(m['epsilon'], dtype=np.float32)]).astype(np.float32)

def main():
    """Interpoliert die Vorhersagen aller Netzwerke für sim_012 auf das volle CFD-Mesh."""
    os.makedirs(OUT, exist_ok=True)

    for net in NETWORKS:
        sparse_dir = os.path.join(BASE, f'evaluation_{net}_val', SIM)
        if not os.path.isdir(sparse_dir):
            print(f'[SKIP] {net}: kein eval-Verzeichnis')
            continue
        print(f"\n=== {net} ===")
        pred_mesh = pv.read(os.path.join(sparse_dir, 'vorhersage.vtu'))
        gt_mesh = pv.read(os.path.join(sparse_dir, 'ground_truth.vtu'))
        npy_dir = os.path.join(TMP, f'{net}_val_npy')
        os.makedirs(npy_dir, exist_ok=True)
        np.save(os.path.join(npy_dir,'positions.npy'), np.asarray(pred_mesh.points, dtype=np.float32))
        np.save(os.path.join(npy_dir,'prediction.npy'), extract(pred_mesh))
        np.save(os.path.join(npy_dir,'true.npy'), extract(gt_mesh))

        out_vtu = os.path.join(OUT, f'{net}.vtu')
        cmd = [PYTHON, INTERP,
               '--pos', os.path.join(npy_dir,'positions.npy'),
               '--pred', os.path.join(npy_dir,'prediction.npy'),
               '--true', os.path.join(npy_dir,'true.npy'),
               '--mesh', MESH, '--output', out_vtu,
               '--method', 'idw', '--k-neighbors', '8']
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FEHLER: {r.stderr[-500:]}")
            continue
        print(f"  -> {out_vtu} ({os.path.getsize(out_vtu)/1e9:.2f} GB)")


if __name__ == '__main__':
    main()
