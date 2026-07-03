"""
Extrapolations-Vorhersagen: alle 6 Netzwerke × 2 Configs (sturm + schwachwind).

Workflow:
  1. Pro (Resolution, Config): einmalig eval mode → sparse VTU + R²
  2. Pro (Netzwerk, Config): Interpolation auf volles CFD-Mesh
  3. Upload nach S3 unter predictions/vorhersagen_extrapolation/<config>/<netz>.vtu
"""
import os, json, subprocess
import numpy as np
import pyvista as pv
import boto3

NETWORKS = {
    'gatv2_coarse': ('coarse', '/home/tbergermann/Python/GAT/output_gatv2_coarse_h128/best_model.pt'),
    'gatv2_medium': ('medium', '/home/tbergermann/Python/GAT/output_gatv2_medium_h128/best_model.pt'),
    'gatv2_bf25':   ('bf25',   '/home/tbergermann/Python/GAT/output_gatv2_bf25_h128/best_model.pt'),
    'gcn_coarse':   ('coarse', '/home/tbergermann/Python/GNN/output_gcn_coarse/best_model.pt'),
    'gcn_medium':   ('medium', '/home/tbergermann/Python/GNN/output_gcn_medium/best_model.pt'),
    'gcn_bf25':     ('bf25',   '/home/tbergermann/Python/GNN/output_gcn_building_focus_25/best_model.pt'),
}
CONFIGS = ['sturm_25ms_45deg', 'schwachwind_1_5ms_45deg']

PT_DIR  = '/home/tbergermann/Python/predictions/extrapolation_pt'
EVAL_BASE = '/home/tbergermann/Python/predictions'  # eval_<net>_<config> wird hier erzeugt
MESH_DIR = '/tmp/extrap_meshes'  # internal.vtu pro Config
OUT_BASE = '/home/tbergermann/Python/predictions/vorhersagen_extrapolation'
PYTHON = '/home/tbergermann/Python/venv/bin/python'
PREDICT = '/home/tbergermann/Python/predictions/predict.py'
INTERP  = '/home/tbergermann/Python/interpolate_to_full_mesh.py'

s3 = boto3.client('s3')
BUCKET = 'amzn-master-sim-bucket'
S3_CFG = boto3.s3.transfer.TransferConfig(multipart_threshold=64*1024*1024, multipart_chunksize=64*1024*1024)


def write_eval_config(net, cfg, ckpt, res, out_dir):
    """Erzeuge YAML-Eval-Config für diesen Netz/Config-Lauf."""
    yaml_path = f'/tmp/extrap_eval_{net}_{cfg}.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"""mode: eval
checkpoint: {ckpt}
graph_source: {PT_DIR}/{res}_{cfg}.pt
output_dir: {out_dir}
export_vtk: true
export_numpy: false
""")
    return yaml_path


def extract_npy(eval_dir, sim_name):
    pred = pv.read(os.path.join(eval_dir, sim_name, 'vorhersage.vtu'))
    gt   = pv.read(os.path.join(eval_dir, sim_name, 'ground_truth.vtu'))
    pos = np.asarray(pred.points, dtype=np.float32)

    def stack(m):
        U = np.asarray(m['U'], dtype=np.float32)
        return np.column_stack([U[:,0], U[:,1], U[:,2],
                                np.asarray(m['p'],       dtype=np.float32),
                                np.asarray(m['k'],       dtype=np.float32),
                                np.asarray(m['epsilon'], dtype=np.float32)]).astype(np.float32)
    return pos, stack(pred), stack(gt)


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    summary = []

    for net, (res, ckpt) in NETWORKS.items():
        for cfg in CONFIGS:
            print(f"\n{'='*72}")
            print(f"  {net} | {cfg}")
            print('='*72)

            # ── 1) Eval ──────────────────────────────────────
            eval_dir = os.path.join(EVAL_BASE, f'evaluation_{net}_extrap_{cfg}')
            yaml_path = write_eval_config(net, cfg, ckpt, res, eval_dir)
            r = subprocess.run([PYTHON, PREDICT, '--config', yaml_path],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  EVAL FEHLER:\n{r.stderr[-800:]}")
                continue
            # R² aus eval_metrics.json
            with open(os.path.join(eval_dir, 'eval_metrics.json')) as f:
                m = json.load(f)
            r2_all = m['metrics']['gesamt']['R2']
            r2_4 = float(np.mean([m['metrics'][k]['R2'] for k in ['Ux','Uy','Uz','p']]))
            print(f"  R² (alle 6): {r2_all:.4f} | R² ohne k,eps: {r2_4:.4f}")
            summary.append((net, cfg, r2_all, r2_4))

            # ── 2) Interpolation auf volles Mesh ────────────
            sim_name = cfg  # in eval_dir steckt sim als sim_id
            sims = os.listdir(eval_dir)
            sim_dir = next((s for s in sims if os.path.isdir(os.path.join(eval_dir, s))), None)
            if sim_dir is None:
                print("  KEIN sim-Verzeichnis gefunden")
                continue

            pos, pred, true = extract_npy(eval_dir, sim_dir)
            npy_dir = f'/tmp/extrap_npy_{net}_{cfg}'
            os.makedirs(npy_dir, exist_ok=True)
            np.save(f'{npy_dir}/positions.npy', pos)
            np.save(f'{npy_dir}/prediction.npy', pred)
            np.save(f'{npy_dir}/true.npy', true)

            mesh_path = f'{MESH_DIR}/{cfg}_internal.vtu'
            out_dir = os.path.join(OUT_BASE, cfg)
            os.makedirs(out_dir, exist_ok=True)
            out_vtu = os.path.join(out_dir, f'{net}.vtu')

            r2 = subprocess.run([PYTHON, INTERP,
                                 '--pos',  f'{npy_dir}/positions.npy',
                                 '--pred', f'{npy_dir}/prediction.npy',
                                 '--true', f'{npy_dir}/true.npy',
                                 '--mesh', mesh_path,
                                 '--output', out_vtu,
                                 '--method', 'idw', '--k-neighbors', '8'],
                                capture_output=True, text=True)
            if r2.returncode != 0:
                print(f"  INTERP FEHLER:\n{r2.stderr[-500:]}")
                continue
            size = os.path.getsize(out_vtu) / 1e9
            print(f"  → {out_vtu} ({size:.2f} GB)")

            # ── 3) Upload nach S3 ───────────────────────────
            key = f'predictions/vorhersagen_extrapolation/{cfg}/{net}.vtu'
            s3.upload_file(out_vtu, BUCKET, key, Config=S3_CFG)
            print(f"  ✓ S3 upload: s3://{BUCKET}/{key}")

    # ── Zusammenfassung ─────────────────────────────────────
    print('\n' + '='*72)
    print('  ZUSAMMENFASSUNG R² (Extrapolation)')
    print('='*72)
    print(f"{'Netzwerk':<15} {'Config':<25} {'R² (alle 6)':>12} {'R² ohne k,eps':>15}")
    print('-' * 72)
    for net, cfg, r2a, r24 in summary:
        print(f"{net:<15} {cfg:<25} {r2a:>12.4f} {r24:>15.4f}")

    # Save summary JSON
    summary_path = os.path.join(OUT_BASE, 'r2_summary.json')
    with open(summary_path, 'w') as f:
        json.dump([{'network': n, 'config': c, 'r2_all_6': a, 'r2_no_k_eps': b}
                   for n,c,a,b in summary], f, indent=2)
    s3.upload_file(summary_path, BUCKET,
                   'predictions/vorhersagen_extrapolation/r2_summary.json')
    print(f"\nSummary: {summary_path} (auch auf S3)")


if __name__ == '__main__':
    main()
