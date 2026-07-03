"""
plot_training_curves_combined.py
================================
Zeichnet alle Trainingsverlaeufe in EIN Diagramm: Trainings- und
Validierungsfehler je Epoche fuer die sechs Laeufe (GCN und GATv2 auf
Coarse/Medium/bf_25), log-skaliert.

Datenquelle: results/training_curves_all.csv (sechs Laeufe; no_cellvol ist
dort bewusst nicht enthalten, da keine vollstaendige Trainingskurve vorliegt).

Konvention: Farbe = Lauf (GCN Blautoene, GATv2 Orangetoene), Linienstil =
durchgezogen Training, gestrichelt Validierung. Kein Titel (steht in der
LaTeX-\\caption).

Verwendung:
    python plot_training_curves_combined.py
Schreibt figures/03_Ergebnisse/training_curves_all.png/pdf.
"""

import csv
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV_PATH = "/home/tbergermann/results/training_curves_all.csv"
OUT_BASE = "/home/tbergermann/figures/03_Ergebnisse/training_curves_all"

FIGSIZE = (11.0, 6.0)
DPI = 300

# Lauf -> (Anzeigename, Farbe, Marker). Reihenfolge = Legendenreihenfolge.
# Hinweis: Die Marker-Eintraege werden aktuell nicht genutzt (Kurven werden nur
# ueber Farbe und Linienstil unterschieden); Struktur bewusst so belassen.
RUN_STYLE = OrderedDict([
    ("gcn_coarse",   ("GCN Coarse",   "#93C5FD", "o")),
    ("gcn_medium",   ("GCN Medium",   "#2563EB", "o")),
    ("gcn_bf_25",    ("GCN bf_25",    "#1E3A8A", "o")),
    ("gatv2_coarse", ("GATv2 Coarse", "#FDBA74", "s")),
    ("gatv2_medium", ("GATv2 Medium", "#F97316", "s")),
    ("gatv2_bf_25",  ("GATv2 bf_25",  "#9A3412", "s")),
])


def setup_style():
    """Setzt den Matplotlib-Stil fuer den kombinierten Plot."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.dpi": DPI,
        "figure.dpi": DPI,
    })


def load_curves(path):
    """Liest die CSV und gruppiert nach run -> dict mit Arrays."""
    data = {run: {"epoch": [], "train": [], "val": [], "best": None}
            for run in RUN_STYLE}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            run = row["run"]
            if run not in data:
                continue
            d = data[run]
            d["epoch"].append(int(row["epoch"]))
            d["train"].append(float(row["train_loss"]))
            d["val"].append(float(row["val_loss"]))
            if row["is_best"] == "True":
                d["best"] = (int(row["epoch"]), float(row["val_loss"]))
    for d in data.values():
        d["epoch"] = np.asarray(d["epoch"])
        d["train"] = np.asarray(d["train"])
        d["val"] = np.asarray(d["val"])
    return data


def main():
    """Zeichnet alle sechs Trainingsverlaeufe in ein gemeinsames Diagramm."""
    setup_style()
    data = load_curves(CSV_PATH)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for run, (label, color, _marker) in RUN_STYLE.items():
        d = data[run]
        if d["epoch"].size == 0:
            continue
        # Training durchgezogen, Validierung gestrichelt, gleiche Farbe
        ax.plot(d["epoch"], d["train"], color=color, linestyle="-",
                linewidth=1.2, alpha=0.9)
        ax.plot(d["epoch"], d["val"], color=color, linestyle="--",
                linewidth=1.2, alpha=0.9)

    ax.set_yscale("log")
    ax.set_xlabel("Epoche")
    ax.set_ylabel("MSE-Verlust (normalisiert)")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)

    # Zwei Legenden: Laeufe (Farbe) und Linienstil (Training/Validierung).
    run_handles = [Line2D([0], [0], color=color, linewidth=2.0, label=label)
                   for label, color, _ in RUN_STYLE.values()]
    style_handles = [
        Line2D([0], [0], color="0.3", linestyle="-", linewidth=1.5,
               label="Training"),
        Line2D([0], [0], color="0.3", linestyle="--", linewidth=1.5,
               label="Validierung"),
    ]
    leg1 = ax.legend(handles=run_handles, loc="upper left",
                     bbox_to_anchor=(1.01, 1.0), title="Lauf",
                     framealpha=0.9)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=style_handles, loc="lower left",
                     bbox_to_anchor=(1.01, 0.0), title="Kurve", framealpha=0.9)

    # bbox_extra_artists stellt sicher, dass die ausserhalb der Achse liegenden
    # Legenden vollstaendig im Bild bleiben (sonst rechts beschnitten).
    extra = (leg1, leg2)
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight", bbox_extra_artists=extra)
    fig.savefig(OUT_BASE + ".png", bbox_inches="tight", bbox_extra_artists=extra)
    plt.close(fig)
    print(f"  geschrieben: {OUT_BASE}.png / .pdf")


if __name__ == "__main__":
    main()
