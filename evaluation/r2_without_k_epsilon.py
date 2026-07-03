"""
Berechnet R² fuer alle Vorhersagen ohne k und epsilon
(nur Ux, Uy, Uz, p — die fuer Ingenieurspraxis relevanten Felder).

Zwei Tabellen:
  1. Aggregiert pro Netzwerk (aus eval_metrics.json)
  2. Pro Sim aus sparse VTU neu berechnet
"""
import os, json
import numpy as np
import pyvista as pv

BASE = '/home/tbergermann/Python/predictions'
NETWORKS = ['gatv2_coarse', 'gatv2_medium', 'gatv2_bf25',
            'gcn_coarse',   'gcn_medium',   'gcn_bf25']
FIELDS_4 = ['Ux', 'Uy', 'Uz', 'p']  # ohne k, epsilon
FIELDS_ALL = ['Ux', 'Uy', 'Uz', 'p', 'k', 'epsilon']

FOCUS_TEST = ['sim_001', 'sim_013', 'sim_014']
FOCUS_VAL = ['sim_012']


def aggregate_r2_no_ke(eval_dir):
    """Liest eval_metrics.json und liefert (R² alle Felder, R² ohne k/eps)."""
    p = os.path.join(eval_dir, 'eval_metrics.json')
    if not os.path.isfile(p):
        return None, None
    with open(p) as f:
        d = json.load(f)
    m = d['metrics']
    r2_4 = np.mean([m[k]['R2'] for k in FIELDS_4])
    r2_all = m['gesamt']['R2']
    return r2_all, r2_4


def per_sim_r2(eval_dir, sim):
    """R² pro Sim aus sparse VTU (ohne k, epsilon vs. mit allen)."""
    pred_p = os.path.join(eval_dir, sim, 'vorhersage.vtu')
    gt_p   = os.path.join(eval_dir, sim, 'ground_truth.vtu')
    if not os.path.isfile(pred_p):
        return None, None
    pm, gm = pv.read(pred_p), pv.read(gt_p)

    def stack(m, fields4):
        """Stapelt die Zielfelder (bei fields4 nur Ux, Uy, Uz, p) zu einem Array."""
        U = np.asarray(m['U'], dtype=np.float64)
        cols = [U[:,0], U[:,1], U[:,2], np.asarray(m['p'], dtype=np.float64)]
        if not fields4:
            cols += [np.asarray(m['k'], dtype=np.float64),
                     np.asarray(m['epsilon'], dtype=np.float64)]
        return np.column_stack(cols)

    def per_field_r2(pred, true):
        """R² je Feld, gemittelt."""
        # R² je Feld, dann Mittel
        scores = []
        for i in range(pred.shape[1]):
            y, yhat = true[:,i], pred[:,i]
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            scores.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        return float(np.mean(scores))

    r2_4   = per_field_r2(stack(pm, True),  stack(gm, True))
    r2_all = per_field_r2(stack(pm, False), stack(gm, False))
    return r2_all, r2_4


def main():
    """Gibt beide R²-Tabellen (aggregiert und pro Sim) auf der Konsole aus."""
    print('\n' + '='*78)
    print('  AGGREGIERT pro Netzwerk (Mittelwert aller Sims im Eval)')
    print('='*78)
    print(f"{'Netzwerk':<16} {'Split':<6} "
          f"{'R² alle (Ux,Uy,Uz,p,k,eps)':>26}  {'R² ohne k,eps':>16}  {'Δ':>7}")
    print('-' * 78)

    results = []
    for split, suffix in [('test', ''), ('val', '_val')]:
        for net in NETWORKS:
            ed = os.path.join(BASE, f'evaluation_{net}{suffix}')
            r2_all, r2_4 = aggregate_r2_no_ke(ed)
            if r2_all is None:
                continue
            print(f"{net:<16} {split:<6} "
                  f"{r2_all:>26.4f}  {r2_4:>16.4f}  "
                  f"{r2_4 - r2_all:>+7.4f}")
            results.append((split, net, r2_all, r2_4))

    print('\n' + '='*78)
    print('  PRO SIM (Test: sim_001/013/014, Val: sim_012)')
    print('='*78)
    print(f"{'Sim':<10} {'Netzwerk':<16} {'Split':<6} "
          f"{'R² (alle)':>10}  {'R² ohne k,eps':>14}")
    print('-' * 78)

    for split, suffix, sims in [('test', '', FOCUS_TEST),
                                 ('val',  '_val', FOCUS_VAL)]:
        for sim in sims:
            for net in NETWORKS:
                ed = os.path.join(BASE, f'evaluation_{net}{suffix}')
                r2_all, r2_4 = per_sim_r2(ed, sim)
                if r2_all is None:
                    continue
                print(f"{sim:<10} {net:<16} {split:<6} "
                      f"{r2_all:>10.4f}  {r2_4:>14.4f}")
            print()


if __name__ == '__main__':
    main()
