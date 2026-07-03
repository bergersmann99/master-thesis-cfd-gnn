"""
Vollstaendige R²-Auswertung:

1) sim_012 (Val) pro Netzwerk: U_ref, angle, R² gesamt, R² ohne k,eps, R² fuer |U|
2) Extrapolation (sturm + schwachwind) pro Netzwerk
3) |U|-Betrag und einzelne Feld-R² fuer GCN Medium und GATv2 Medium auf Test
"""
import os, subprocess
import numpy as np
import pyvista as pv

NETWORKS = ['gatv2_coarse', 'gatv2_medium', 'gatv2_bf25',
            'gcn_coarse',   'gcn_medium',   'gcn_bf25']

CHECKPOINTS = {
    'gatv2_coarse': '/home/tbergermann/Python/GAT/output_gatv2_coarse_h128/best_model.pt',
    'gatv2_medium': '/home/tbergermann/Python/GAT/output_gatv2_medium_h128/best_model.pt',
    'gatv2_bf25':   '/home/tbergermann/Python/GAT/output_gatv2_bf25_h128/best_model.pt',
    'gcn_coarse':   '/home/tbergermann/Python/GNN/output_gcn_coarse/best_model.pt',
    'gcn_medium':   '/home/tbergermann/Python/GNN/output_gcn_medium/best_model.pt',
    'gcn_bf25':     '/home/tbergermann/Python/GNN/output_gcn_building_focus_25/best_model.pt',
}

RES = {'gatv2_coarse':'coarse','gatv2_medium':'medium','gatv2_bf25':'bf25',
       'gcn_coarse':'coarse','gcn_medium':'medium','gcn_bf25':'bf25'}

EXTRAP_PT = '/home/tbergermann/Python/predictions/extrapolation_pt'
PYTHON = '/home/tbergermann/Python/venv/bin/python'
PREDICT = '/home/tbergermann/Python/predictions/predict.py'


def stack_fields(mesh):
    """Liest die sechs Zielfelder eines Meshes in ein dict von Arrays."""
    U = np.asarray(mesh['U'], dtype=np.float64)
    return {
        'Ux': U[:,0], 'Uy': U[:,1], 'Uz': U[:,2],
        'p':       np.asarray(mesh['p'],       dtype=np.float64),
        'k':       np.asarray(mesh['k'],       dtype=np.float64),
        'epsilon': np.asarray(mesh['epsilon'], dtype=np.float64),
    }


def r2(y, yhat):
    """R² zwischen Ground Truth y und Vorhersage yhat."""
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def field_r2_from_vtu(eval_dir, sim_sub):
    """Berechnet feldweise R² (inkl. |U|, gesamt, ohne k/eps) aus den sparse VTUs."""
    pred_path = os.path.join(eval_dir, sim_sub, 'vorhersage.vtu')
    gt_path   = os.path.join(eval_dir, sim_sub, 'ground_truth.vtu')
    if not os.path.isfile(pred_path):
        return None
    pm = pv.read(pred_path)
    gm = pv.read(gt_path)
    pf = stack_fields(pm)
    tf = stack_fields(gm)
    out = {f: r2(tf[f], pf[f]) for f in ['Ux','Uy','Uz','p','k','epsilon']}
    # |U|-Betrag
    pred_mag = np.sqrt(pf['Ux']**2 + pf['Uy']**2 + pf['Uz']**2)
    true_mag = np.sqrt(tf['Ux']**2 + tf['Uy']**2 + tf['Uz']**2)
    out['|U|'] = r2(true_mag, pred_mag)
    out['ohne_k_eps'] = float(np.mean([out[f] for f in ['Ux','Uy','Uz','p']]))
    out['gesamt'] = float(np.mean([out[f] for f in ['Ux','Uy','Uz','p','k','epsilon']]))
    return out


def write_eval_yaml(checkpoint, graph_source, output_dir, path):
    """Schreibt eine YAML-Eval-Config für predict.py."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"""mode: eval
checkpoint: {checkpoint}
graph_source: {graph_source}
output_dir: {output_dir}
export_vtk: true
export_numpy: false
""")


def run_eval(yaml_path):
    """Führt predict.py im Eval-Modus aus; liefert True bei Erfolg."""
    return subprocess.run([PYTHON, PREDICT, '--config', yaml_path],
                          capture_output=True, text=True).returncode == 0


def main():
    """Führt die drei R²-Auswertungen aus: Val (sim_012), Extrapolation, Test-Set."""
    # ────────────────────────────────────────────────────────────────────
    # 1) sim_012 (Val) pro Netzwerk
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "="*88)
    print("  1) sim_012 (Val) — U_ref=19.27 m/s, angle=30.29°")
    print("="*88)
    print(f"{'Netzwerk':<15} {'Ux':>7} {'Uy':>7} {'Uz':>7} {'|U|':>7} {'p':>7} "
          f"{'k':>7} {'ε':>7} {'gesamt':>8} {'oh.k,ε':>8}")
    print('-'*88)
    for net in NETWORKS:
        eval_dir = f'/home/tbergermann/Python/predictions/evaluation_{net}_val'
        r = field_r2_from_vtu(eval_dir, 'sim_012')
        if r:
            print(f"{net:<15} {r['Ux']:>7.4f} {r['Uy']:>7.4f} {r['Uz']:>7.4f} {r['|U|']:>7.4f} "
                  f"{r['p']:>7.4f} {r['k']:>7.4f} {r['epsilon']:>7.4f} "
                  f"{r['gesamt']:>8.4f} {r['ohne_k_eps']:>8.4f}")


    # ────────────────────────────────────────────────────────────────────
    # 2) Extrapolation — Eval frisch laufen lassen
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "="*88)
    print("  2) EXTRAPOLATION — Sturm (25 m/s) + Schwachwind (1.5 m/s), beide @ 45°")
    print("="*88)
    extrap_results = {}
    for cfg in ['sturm_25ms_45deg', 'schwachwind_1_5ms_45deg']:
        extrap_results[cfg] = {}
        for net in NETWORKS:
            eval_dir = f'/home/tbergermann/Python/predictions/eval_extrap_{net}_{cfg}'
            if not os.path.isfile(os.path.join(eval_dir, 'eval_metrics.json')):
                yaml_path = f'/tmp/y_extrap_{net}_{cfg}.yaml'
                graph = f'{EXTRAP_PT}/{RES[net]}_{cfg}.pt'
                write_eval_yaml(CHECKPOINTS[net], graph, eval_dir, yaml_path)
                print(f"  -> Eval {net} / {cfg} ...")
                run_eval(yaml_path)
            sims = [s for s in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, s))]
            if not sims:
                continue
            r = field_r2_from_vtu(eval_dir, sims[0])
            if r:
                extrap_results[cfg][net] = r

    for cfg, label in [('sturm_25ms_45deg', 'STURM (25 m/s)'),
                       ('schwachwind_1_5ms_45deg', 'SCHWACHWIND (1.5 m/s)')]:
        print(f"\n  --- {label} ---")
        print(f"{'Netzwerk':<15} {'Ux':>7} {'Uy':>7} {'Uz':>7} {'|U|':>7} {'p':>7} "
              f"{'k':>7} {'ε':>7} {'gesamt':>8} {'oh.k,ε':>8}")
        print('-'*88)
        for net in NETWORKS:
            r = extrap_results.get(cfg, {}).get(net)
            if r:
                print(f"{net:<15} {r['Ux']:>7.4f} {r['Uy']:>7.4f} {r['Uz']:>7.4f} {r['|U|']:>7.4f} "
                      f"{r['p']:>7.4f} {r['k']:>7.4f} {r['epsilon']:>7.4f} "
                      f"{r['gesamt']:>8.4f} {r['ohne_k_eps']:>8.4f}")


    # ────────────────────────────────────────────────────────────────────
    # 3) Test-Set per-Sim per-Field fuer GCN Medium und GATv2 Medium
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "="*88)
    print("  3) TEST-SET — pro Sim und Feld (GCN Medium und GATv2 Medium)")
    print("="*88)
    TEST_SIMS = ['sim_001','sim_002','sim_008','sim_013','sim_014','sim_035','sim_038','sim_048']

    for net in ['gcn_medium', 'gatv2_medium']:
        print(f"\n  --- {net.upper()} ---")
        print(f"{'Sim':<10} {'Ux':>7} {'Uy':>7} {'Uz':>7} {'|U|':>7} {'p':>7} "
              f"{'k':>7} {'ε':>7} {'gesamt':>8} {'oh.k,ε':>8}")
        print('-'*88)
        eval_dir = f'/home/tbergermann/Python/predictions/evaluation_{net}'
        for sim in TEST_SIMS:
            r = field_r2_from_vtu(eval_dir, sim)
            if r:
                print(f"{sim:<10} {r['Ux']:>7.4f} {r['Uy']:>7.4f} {r['Uz']:>7.4f} {r['|U|']:>7.4f} "
                      f"{r['p']:>7.4f} {r['k']:>7.4f} {r['epsilon']:>7.4f} "
                      f"{r['gesamt']:>8.4f} {r['ohne_k_eps']:>8.4f}")


if __name__ == '__main__':
    main()
