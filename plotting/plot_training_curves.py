"""
plot_training_curves.py
=======================
Erzeugt pro Subsampling-Stufe (Coarse, Medium, bf_25) eine Abbildung mit
Trainings- und Validierungsfehler sowie Lernrate je Epoche fuer das GCN-
und das GATv2-Modell der jeweiligen Stufe (efficient-Rerun-Laeufe).

Datenquellen
------------
- GCN  (alle drei Stufen): training_history.json aus
  output_gcn_<stufe>_rerun/ — enthaelt train_loss, val_loss,
  learning_rate, best_epoch, best_val_loss vollstaendig.
- GATv2 (alle drei Stufen): training_history.json aus
  output_gatv2_<stufe>_h128/. Diese Logs enthalten nur train_loss
  und val_loss. Der Lernratenverlauf wird deterministisch aus dem
  val_loss-Verlauf rekonstruiert (selbe ReduceLROnPlateau-Konfiguration
  wie im Training: mode=min, factor=0.5, patience=20, init_lr=1e-4,
  min_lr=1e-6). Das Ergebnis ist eindeutig.

Plot
----
Pro Stufe ein PNG (figures/03_Ergebnisse/training_curves_<stufe>.png) mit
zwei Subplots:
  - oben:  Trainings- und Validierungsfehler je Epoche, log-skaliert,
           GCN und GATv2 in einer Abbildung, beste Epoche je Modell
           als Marker.
  - unten: Lernrate je Epoche, log-skaliert.

Verwendung
----------
    python plot_training_curves.py

Schreibt drei PNGs nach figures/03_Ergebnisse/ und gibt eine kompakte
Konsolentabelle aus.

Siehe QUELLEN_plot_training_curves.md fuer Quellen.
"""

import os
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Pfad-Konfiguration (verifiziert; nicht raten)
# ----------------------------------------------------------------------

RUNS = {
    "coarse": {
        "gcn": "/home/tbergermann/Python/GNN/output_gcn_coarse_rerun",
        "gatv2": "/home/tbergermann/Python/GAT/output_gatv2_coarse_h128",
    },
    "medium": {
        "gcn": "/home/tbergermann/Python/GNN/output_gcn_medium_rerun",
        "gatv2": "/home/tbergermann/Python/GAT/output_gatv2_medium_h128",
    },
    "bf25": {
        "gcn": "/home/tbergermann/Python/GNN/output_gcn_bf25_rerun",
        "gatv2": "/home/tbergermann/Python/GAT/output_gatv2_bf25_h128",
    },
}

FIG_DIR = "/home/tbergermann/figures/03_Ergebnisse"


# ----------------------------------------------------------------------
# Daten laden
# ----------------------------------------------------------------------

def load_history(json_path):
    """Laedt training_history.json und gibt ein Dict mit train_loss,
    val_loss, learning_rate, best_epoch, best_val_loss zurueck.

    learning_rate wird rekonstruiert, falls nicht enthalten."""
    h = json.load(open(json_path))
    train = np.asarray(h["train_loss"], dtype=float)
    val = np.asarray(h["val_loss"], dtype=float)
    n = len(train)
    if "learning_rate" in h and len(h["learning_rate"]) == n:
        lr = np.asarray(h["learning_rate"], dtype=float)
    else:
        lr = reconstruct_lr(val)
    if "best_epoch" in h and h["best_epoch"]:
        best_ep = int(h["best_epoch"])
        best_val = float(h["best_val_loss"])
    else:
        # Aus val_loss-Argmin
        idx = int(np.argmin(val))
        best_ep = idx + 1
        best_val = float(val[idx])
    return {
        "train": train,
        "val": val,
        "lr": lr,
        "best_epoch": best_ep,
        "best_val": best_val,
        "n_epochs": n,
    }


def reconstruct_lr(val_loss, init_lr=1e-4, factor=0.5,
                   patience=20, min_lr=1e-6, threshold=1e-4):
    """Rekonstruiert den Lernratenverlauf aus dem Validierungsfehler
    durch deterministische Simulation von torch.optim.lr_scheduler.
    ReduceLROnPlateau mit mode='min'.

    Dieselbe Konfiguration wie im Training (siehe trainGCN_efficient.py
    und trainGATv2_efficient.py):
        factor=0.5, patience=20, min_lr=1e-6, threshold=1e-4 (Default).
    """
    n = len(val_loss)
    lr = np.empty(n, dtype=float)
    current = float(init_lr)
    best = float("inf")
    bad = 0
    for i in range(n):
        # scheduler.step wird nach val_loss aufgerufen; lr ist
        # der Wert WAEHREND der Epoche (vor moeglicher Reduktion)
        lr[i] = current
        v = float(val_loss[i])
        # PyTorch ReduceLROnPlateau (mode='min', threshold_mode='rel'):
        # Verbesserung, wenn v < best*(1-threshold). Mit best=inf ist die erste
        # Epoche immer eine Verbesserung. (Die fruehere Form best-threshold*best
        # ergab fuer best=inf nan, sodass der Vergleich stets False war und die
        # LR rein zeitgesteuert zerfiel — falsch.)
        if v < best * (1.0 - threshold):
            best = v
            bad = 0
        else:
            bad += 1
            if bad > patience:
                new = max(current * factor, min_lr)
                if current - new > 1e-8:
                    current = new
                # num_bad_epochs wird nach dem Ausloesen immer zurueckgesetzt
                bad = 0
    return lr


# ----------------------------------------------------------------------
# Plot je Stufe
# ----------------------------------------------------------------------

def plot_stage(stage, gcn_h, gat_h, out_path):
    fig, (ax_loss, ax_lr) = plt.subplots(
        2, 1, figsize=(9, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08},
    )

    color_gcn = "tab:blue"
    color_gat = "tab:orange"

    ep_gcn = np.arange(1, gcn_h["n_epochs"] + 1)
    ep_gat = np.arange(1, gat_h["n_epochs"] + 1)

    # Loss-Achse (log y)
    ax_loss.plot(ep_gcn, gcn_h["train"], color=color_gcn,
                 linestyle="-", linewidth=1.2,
                 label="GCN — Trainingsfehler")
    ax_loss.plot(ep_gcn, gcn_h["val"], color=color_gcn,
                 linestyle="--", linewidth=1.2,
                 label="GCN — Validierungsfehler")
    ax_loss.plot(ep_gat, gat_h["train"], color=color_gat,
                 linestyle="-", linewidth=1.2,
                 label="GATv2 — Trainingsfehler")
    ax_loss.plot(ep_gat, gat_h["val"], color=color_gat,
                 linestyle="--", linewidth=1.2,
                 label="GATv2 — Validierungsfehler")

    # Marker auf bester Epoche je Modell
    ax_loss.plot(gcn_h["best_epoch"], gcn_h["best_val"], "o",
                 color=color_gcn, markersize=8, markeredgecolor="black",
                 markeredgewidth=0.8, zorder=5,
                 label=f"GCN beste Epoche {gcn_h['best_epoch']} "
                       f"(Val {gcn_h['best_val']:.4f})")
    ax_loss.plot(gat_h["best_epoch"], gat_h["best_val"], "s",
                 color=color_gat, markersize=8, markeredgecolor="black",
                 markeredgewidth=0.8, zorder=5,
                 label=f"GATv2 beste Epoche {gat_h['best_epoch']} "
                       f"(Val {gat_h['best_val']:.4f})")

    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("MSE-Verlust (normalisiert)")
    ax_loss.grid(True, which="both", alpha=0.25)
    title = f"Trainingsverlauf — Subsampling-Stufe {stage}"
    ax_loss.set_title(title, fontsize=12)
    ax_loss.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # Lernraten-Achse (log y)
    ax_lr.plot(ep_gcn, gcn_h["lr"], color=color_gcn, linewidth=1.2,
               label="GCN")
    ax_lr.plot(ep_gat, gat_h["lr"], color=color_gat, linewidth=1.2,
               label="GATv2 (rekonstruiert)")
    ax_lr.set_yscale("log")
    ax_lr.set_xlabel("Epoche")
    ax_lr.set_ylabel("Lernrate")
    ax_lr.grid(True, which="both", alpha=0.25)
    ax_lr.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Hauptlauf
# ----------------------------------------------------------------------

def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"{'Stufe':<8} {'Modell':<7} {'#Ep':>6} {'best_ep':>9} "
          f"{'best_val':>10} {'final_train':>13} {'final_val':>11}")
    print("-" * 70)

    for stage, dirs in RUNS.items():
        # GCN
        gcn_p = os.path.join(dirs["gcn"], "training_history.json")
        gat_p = os.path.join(dirs["gatv2"], "training_history.json")
        if not os.path.exists(gcn_p):
            raise FileNotFoundError(gcn_p)
        if not os.path.exists(gat_p):
            raise FileNotFoundError(gat_p)
        gcn_h = load_history(gcn_p)
        gat_h = load_history(gat_p)

        for name, h in [("GCN", gcn_h), ("GATv2", gat_h)]:
            print(f"{stage:<8} {name:<7} {h['n_epochs']:>6} "
                  f"{h['best_epoch']:>9} {h['best_val']:>10.4f} "
                  f"{float(h['train'][-1]):>13.4f} "
                  f"{float(h['val'][-1]):>11.4f}")

        out_path = os.path.join(
            FIG_DIR, f"training_curves_{stage}.png")
        plot_stage(stage, gcn_h, gat_h, out_path)
        print(f"  -> {out_path}")
        print()

    print("Fertig.")


if __name__ == "__main__":
    main()
