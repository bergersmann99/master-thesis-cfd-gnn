"""
Schlanker Komplett-Batch:

1) Extrapolation gcn_bf25 schwachwind  (einziger fehlender Original-Lauf)
   -> mit IDW-Interpolation auf volles CFD-Mesh

2) no_cellvol: Test (sim_001/013/014) + Val (sim_012) + Extrapolation
   -> NUR sparse VTU (kein IDW, kein CFD-Mesh) -- mesh-freie Variante
"""
import os, json, subprocess, shutil
import numpy as np
import torch
import pyvista as pv
import boto3

PYTHON  = '/home/tbergermann/Python/venv/bin/python'
PREDICT = '/home/tbergermann/Python/predictions/predict.py'
INTERP  = '/home/tbergermann/Python/interpolate_to_full_mesh.py'
BUCKET  = 'amzn-master-sim-bucket'
s3 = boto3.client('s3')
TCFG = boto3.s3.transfer.TransferConfig(
    multipart_threshold=64*1024*1024, multipart_chunksize=64*1024*1024)

NOCELLVOL_CKPT = ('/home/tbergermann/Python/GAT/'
                  'output_gatv2_medium_no_cellvol_h128/best_model.pt')
NOCELLVOL_NAME = 'gatv2_medium_no_cellvol'
GCN_BF25_CKPT  = ('/home/tbergermann/Python/GNN/'
                  'output_gcn_building_focus_25/best_model.pt')

EXTRAP_PT_DIR  = '/home/tbergermann/Python/predictions/extrapolation_pt'
NOCELLVOL_DATA = '/home/tbergermann/Python/datasets/medium_no_cellvol'
EXTRAP_MESHES  = '/tmp/extrap_meshes'


def write_eval_yaml(checkpoint, graph_source, output_dir, path):
    with open(path, 'w') as f:
        f.write(f"""mode: eval
checkpoint: {checkpoint}
graph_source: {graph_source}
output_dir: {output_dir}
export_vtk: true
export_numpy: false
""")


def run_eval(yaml_path, label):
    print(f"\n  >> EVAL {label}")
    r = subprocess.run([PYTHON, PREDICT, '--config', yaml_path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    FEHLER:\n{r.stderr[-800:]}")
        return False
    return True


def upload_sparse(eval_dir, sim_subdir, s3_key, label):
    """Lädt nur die sparse vorhersage.vtu hoch (mesh-frei)."""
    src = os.path.join(eval_dir, sim_subdir, 'vorhersage.vtu')
    if not os.path.isfile(src):
        print(f"    {sim_subdir}: vorhersage.vtu fehlt")
        return False
    print(f"  >> UPLOAD sparse {s3_key}")
    s3.upload_file(src, BUCKET, s3_key, Config=TCFG)
    print(f"    ✓ s3://{BUCKET}/{s3_key} ({os.path.getsize(src)/1e6:.1f} MB)")
    return True


def interp_and_upload(eval_dir, sim_subdir, mesh_path, s3_key, out_vtu, label):
    pred = pv.read(os.path.join(eval_dir, sim_subdir, 'vorhersage.vtu'))
    gt   = pv.read(os.path.join(eval_dir, sim_subdir, 'ground_truth.vtu'))
    pos  = np.asarray(pred.points, dtype=np.float32)

    def stack(m):
        U = np.asarray(m['U'], dtype=np.float32)
        return np.column_stack([U[:,0], U[:,1], U[:,2],
                                np.asarray(m['p'],       dtype=np.float32),
                                np.asarray(m['k'],       dtype=np.float32),
                                np.asarray(m['epsilon'], dtype=np.float32)]).astype(np.float32)
    npy_dir = f'/tmp/npy_{label.replace("/","_").replace(" ","_")}'
    os.makedirs(npy_dir, exist_ok=True)
    np.save(f'{npy_dir}/positions.npy',  pos)
    np.save(f'{npy_dir}/prediction.npy', stack(pred))
    np.save(f'{npy_dir}/true.npy',       stack(gt))

    print(f"  >> INTERP {label}")
    r = subprocess.run([PYTHON, INTERP,
                        '--pos',  f'{npy_dir}/positions.npy',
                        '--pred', f'{npy_dir}/prediction.npy',
                        '--true', f'{npy_dir}/true.npy',
                        '--mesh', mesh_path,
                        '--output', out_vtu,
                        '--method', 'idw', '--k-neighbors', '8'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    FEHLER:\n{r.stderr[-500:]}")
        return False
    print(f"    -> {out_vtu} ({os.path.getsize(out_vtu)/1e9:.2f} GB)")
    print(f"  >> UPLOAD {s3_key}")
    s3.upload_file(out_vtu, BUCKET, s3_key, Config=TCFG)
    print(f"    ✓ s3://{BUCKET}/{s3_key}")
    os.remove(out_vtu)
    shutil.rmtree(npy_dir, ignore_errors=True)
    return True


# ------------------------------------------------------------------
# 1) Extrapolation gcn_bf25 schwachwind
# ------------------------------------------------------------------
def task_extrap_gcn_bf25_schwachwind():
    print("\n" + "="*70)
    print("  1) EXTRAPOLATION gcn_bf25 / schwachwind_1_5ms_45deg")
    print("="*70)
    cfg = 'schwachwind_1_5ms_45deg'
    eval_dir = f'/home/tbergermann/Python/predictions/eval_gcn_bf25_extrap_{cfg}'
    yaml_path = f'/tmp/yaml_gcn_bf25_extrap_{cfg}.yaml'
    write_eval_yaml(GCN_BF25_CKPT,
                    f'{EXTRAP_PT_DIR}/bf25_{cfg}.pt',
                    eval_dir, yaml_path)
    if not run_eval(yaml_path, f'gcn_bf25 / {cfg}'):
        return
    sims = [s for s in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, s))]
    sim_sub = sims[0]
    out_dir = f'/home/tbergermann/Python/predictions/vorhersagen_extrapolation/{cfg}'
    os.makedirs(out_dir, exist_ok=True)
    out_vtu = f'{out_dir}/gcn_bf25.vtu'
    mesh = f'{EXTRAP_MESHES}/{cfg}_internal.vtu'
    interp_and_upload(eval_dir, sim_sub, mesh,
                      f'predictions/vorhersagen_extrapolation/{cfg}/gcn_bf25.vtu',
                      out_vtu, f'extrap gcn_bf25/{cfg}')
    shutil.rmtree(eval_dir, ignore_errors=True)


# ------------------------------------------------------------------
# 2) no_cellvol Test/Val (mesh-frei, nur sparse VTU)
# ------------------------------------------------------------------
def task_no_cellvol_sparse(label, pt_path, sim_filter, s3_dir_template):
    """
    label: kurze Beschreibung
    pt_path: graph_source (.pt)
    sim_filter: Liste von sim_ids die hochgeladen werden sollen
    s3_dir_template: f-string mit {sim} placeholder oder fester Pfad
    """
    print("\n" + "="*70)
    print(f"  {label}")
    print("="*70)
    eval_dir = f'/home/tbergermann/Python/predictions/eval_{label.lower().replace(" ","_")}'
    yaml_path = f'/tmp/yaml_{label.lower().replace(" ","_")}.yaml'
    write_eval_yaml(NOCELLVOL_CKPT, pt_path, eval_dir, yaml_path)
    if not run_eval(yaml_path, label):
        return

    for sim in sim_filter:
        sim_sub = sim if os.path.isdir(os.path.join(eval_dir, sim)) else None
        if sim_sub is None:
            print(f"    {sim} fehlt im eval_dir")
            continue
        s3_key = s3_dir_template.format(sim=sim) + f'/{NOCELLVOL_NAME}.vtu'
        upload_sparse(eval_dir, sim_sub, s3_key, label)

    shutil.rmtree(eval_dir, ignore_errors=True)


# ------------------------------------------------------------------
# 3) no_cellvol Extrapolation (mesh-frei)
# ------------------------------------------------------------------
def task_no_cellvol_extrap_sparse():
    print("\n" + "="*70)
    print("  no_cellvol Extrapolation (sparse, mesh-frei)")
    print("="*70)
    for cfg in ['sturm_25ms_45deg', 'schwachwind_1_5ms_45deg']:
        # 14-Feature .pt laden, cell_volume (Index 4) entfernen
        src_pt = f'{EXTRAP_PT_DIR}/medium_{cfg}.pt'
        new_pt = f'{EXTRAP_PT_DIR}/medium_no_cellvol_{cfg}.pt'
        print(f"  >> Strippe cell_volume aus medium_{cfg}.pt")
        data = torch.load(src_pt, weights_only=False, map_location='cpu')
        if not isinstance(data, list):
            data = [data]
        for g in data:
            if g.x.shape[1] == 14:
                g.x = torch.cat([g.x[:, :4], g.x[:, 5:]], dim=1)
        torch.save(data, new_pt)
        print(f"    -> {new_pt} (x.shape={data[0].x.shape})")

        eval_dir = f'/home/tbergermann/Python/predictions/eval_no_cellvol_extrap_{cfg}'
        yaml_path = f'/tmp/yaml_no_cellvol_extrap_{cfg}.yaml'
        write_eval_yaml(NOCELLVOL_CKPT, new_pt, eval_dir, yaml_path)
        if not run_eval(yaml_path, f'no_cellvol extrap {cfg}'):
            continue

        sims = [s for s in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, s))]
        sim_sub = sims[0]
        s3_key = f'predictions/vorhersagen_extrapolation/{cfg}/{NOCELLVOL_NAME}.vtu'
        upload_sparse(eval_dir, sim_sub, s3_key, f'no_cellvol extrap {cfg}')

        shutil.rmtree(eval_dir, ignore_errors=True)
        os.remove(new_pt)


def main():
    task_extrap_gcn_bf25_schwachwind()

    task_no_cellvol_sparse(
        label='no_cellvol Test',
        pt_path=f'{NOCELLVOL_DATA}/test.pt',
        sim_filter=['sim_001', 'sim_013', 'sim_014'],
        s3_dir_template='predictions/vorhersagen_test_sims/{sim}',
    )

    task_no_cellvol_sparse(
        label='no_cellvol Val',
        pt_path=f'{NOCELLVOL_DATA}/val.pt',
        sim_filter=['sim_012'],
        s3_dir_template='predictions/vorhersagen_val_sim_012',
    )

    task_no_cellvol_extrap_sparse()

    print("\n" + "="*70)
    print("  ALLE TASKS ABGESCHLOSSEN")
    print("="*70)


if __name__ == '__main__':
    main()
