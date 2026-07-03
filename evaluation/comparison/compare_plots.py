"""
compare_plots.py
================
Erzeugt publikationsfertige Vergleichsplots fuer die GNN-Modelle.

Aktiv erzeugt werden (fuer die Masterarbeit, ohne Titel — Titel steht in der
LaTeX-\\caption):
    1. comparison_r2_fields.png/pdf  — R² pro Stroemungsfeld (gruppiertes Balkendiagramm)
    5. comparison_epsilon.png/pdf    — epsilon-R² (Dissipationsrate) separat

Datenquelle:
    R²-Werte je Feld stammen aus <model_dir>/test_metrics.json. Die Modell-
    Verzeichnisse stehen in models.yaml und zeigen auf die fairen GCN-Rerun-
    Laeufe (output_gcn_<stufe>_rerun) und die GATv2-h128-Laeufe — NICHT auf die
    alten, untertrainierten GCN-Modelle. Zusaetzlich wird die Ablation
    "GATv2 Medium no_cellvol" (13 Merkmale, output_gatv2_medium_no_cellvol_h128)
    als eigene Serie aufgenommen.

Die Skalierungs-/Kosten-Nutzen-Plots (plot_scaling_r2, plot_scaling_rl2,
plot_cost_benefit) werden nicht mehr aktiv erzeugt, die Funktionen bleiben aber
fuer manuelle Nutzung erhalten und lauffaehig.

Verwendung:
    python compare_plots.py --config models.yaml
    python compare_plots.py --config models.yaml --output /pfad/comparison

Ausgabe (Standard): /home/tbergermann/figures/03_Ergebnisse/comparison_*.png/pdf

Siehe QUELLEN_compare_plots.md fuer uebernommene Techniken (Dezimalkomma-
Formatter, Legende ausserhalb der Datenflaeche).
"""

import os
import sys
import json
import argparse

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ======================================================================
# Daten laden (identisch zu compare_tables.py)
# ======================================================================

FIELD_NAMES = ["Ux", "Uy", "Uz", "p", "k", "epsilon"]

# Ablation: GATv2 Medium ohne cell_volume (13 Merkmale). Wird als zusaetzliche
# Serie nur in den beiden Abbildungen gefuehrt (nicht in models.yaml, damit
# compare_tables.py unveraendert mit den sechs Kernmodellen arbeitet).
NO_CELLVOL = {
    "dir": "/home/tbergermann/Python/GAT/output_gatv2_medium_no_cellvol_h128",
    "name": "GATv2 Medium no_cellvol",
    "nodes": 507000,
    "time_min": 0.0,   # fuer die beiden Abbildungen ohne Bedeutung
    "params": 403974,
}

FIG_DIR = "/home/tbergermann/figures/03_Ergebnisse"


def load_model_data(model_dir, name, n_nodes, train_time_min,
                    n_params):
    """Laedt test_metrics.json und kombiniert mit Metadaten."""
    metrics_path = os.path.join(model_dir, "test_metrics.json")
    if not os.path.exists(metrics_path):
        print(f"WARNUNG: {metrics_path} nicht gefunden, "
              f"ueberspringe {name}")
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)

    return {
        "name": name,
        "dir": model_dir,
        "n_nodes": n_nodes,
        "train_time_min": train_time_min,
        "n_params": n_params,
        "metrics": metrics,
    }


def load_from_config(config_path):
    """Laedt Modell-Definitionen aus YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    models = []
    for m in cfg["models"]:
        data = load_model_data(
            m["dir"], m["name"], m["nodes"],
            m["time_min"], m["params"])
        if data is not None:
            models.append(data)

    return models


# ======================================================================
# Stil-Konfiguration (zentral; gemeinsame figsize/dpi/Schriftgroessen,
# damit beide Abbildungen in der Arbeit gleich gross wirken)
# ======================================================================

FIGSIZE = (10.0, 5.5)
DPI = 300


def setup_style():
    """Setzt den Matplotlib-Stil fuer alle Plots."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.facecolor": "white",
        "savefig.dpi": DPI,
        "figure.dpi": DPI,
    })


# Farben fuer GCN (blau) und GATv2 (rot) — fuer die (nicht mehr aktiven)
# Skalierungs-/Kosten-Plots.
COLORS_GCN = "#2563EB"
COLORS_GATV2 = "#DC2626"

# Eigene Farbskala fuer die Balkenabbildungen: drei Blautoene (GCN),
# drei Rottoene (GATv2) und Gruen fuer die no_cellvol-Ablation, damit die
# Stufen innerhalb einer Familie unterscheidbar sind.
LEVEL_COLORS = {
    "GCN Coarse": "#93C5FD",
    "GCN Medium": "#3B82F6",
    "GCN bf_25": "#1E40AF",
    "GATv2 Coarse": "#FCA5A5",
    "GATv2 Medium": "#EF4444",
    "GATv2 bf_25": "#991B1B",
    "GATv2 Medium no_cellvol": "#059669",
}


def model_color(name):
    """Farbe je Modellname fuer die Balkenabbildungen."""
    return LEVEL_COLORS.get(name, "#6B7280")


def get_color(name):
    """Gibt die Farbe basierend auf dem Modellnamen zurueck."""
    if "GATv2" in name:
        return COLORS_GATV2
    return COLORS_GCN


def get_marker(name):
    """Gibt den Marker basierend auf dem Modellnamen zurueck."""
    if "GATv2" in name:
        return "s"
    return "o"


def komma_formatter(decimals=2):
    """Achsen-Formatter mit deutschem Dezimalkomma (z. B. 0,95)."""
    return ticker.FuncFormatter(
        lambda v, _pos: f"{v:.{decimals}f}".replace(".", ","))


def komma(value, decimals=3):
    """Zahl als String mit deutschem Dezimalkomma."""
    return f"{value:.{decimals}f}".replace(".", ",")


# ======================================================================
# Plot 1: R² pro Feld (Balkendiagramm)
# ======================================================================

def plot_r2_fields(models, output_path):
    """
    Gruppiertes Balkendiagramm: R² pro Feld fuer alle Modelle.
    Ohne Titel; epsilon als Mathtext; Dezimalkomma auf der y-Achse;
    Legende ausserhalb der Datenflaeche (unterhalb).
    """
    n_models = len(models)
    fields = FIELD_NAMES + ["gesamt"]
    labels = ["Ux", "Uy", "Uz", "p", "k", r"$\varepsilon$", "Gesamt"]

    x = np.arange(len(fields))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for i, m in enumerate(models):
        values = [m["metrics"][f]["R2"] for f in fields]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, values, width * 0.9, label=m["name"],
               color=model_color(m["name"]), edgecolor="white",
               linewidth=0.4)

    ax.set_ylabel("R²")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.70, 1.00)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(komma_formatter(2))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    # Legende ausserhalb der Datenflaeche (unterhalb), zwei Reihen.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=4, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path + "_r2_fields.pdf", bbox_inches="tight")
    fig.savefig(output_path + "_r2_fields.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {output_path}_r2_fields.png")


# ======================================================================
# Plot 2: R² Skalierungskurve  (nicht mehr aktiv erzeugt)
# ======================================================================

def plot_scaling_r2(models, output_path):
    """
    R² ueber Knotenanzahl, getrennt nach GCN und GATv2.
    """
    gcn = sorted([m for m in models if "GCN" in m["name"]
                  and "GATv2" not in m["name"]],
                 key=lambda m: m["n_nodes"])
    gatv2 = sorted([m for m in models if "GATv2" in m["name"]],
                   key=lambda m: m["n_nodes"])

    fig, ax = plt.subplots(figsize=(8, 5))

    # GCN
    if gcn:
        nodes = [m["n_nodes"] for m in gcn]
        r2 = [m["metrics"]["gesamt"]["R2"] for m in gcn]
        ax.plot(nodes, r2, "o-", color=COLORS_GCN, linewidth=2,
                markersize=8, label="GCN (h128, L10)",
                zorder=3)
        for m in gcn:
            ax.annotate(
                f'{m["metrics"]["gesamt"]["R2"]:.3f}',
                (m["n_nodes"], m["metrics"]["gesamt"]["R2"]),
                textcoords="offset points", xytext=(0, 12),
                fontsize=8, ha="center", color=COLORS_GCN)

    # GATv2
    if gatv2:
        nodes = [m["n_nodes"] for m in gatv2]
        r2 = [m["metrics"]["gesamt"]["R2"] for m in gatv2]
        ax.plot(nodes, r2, "s-", color=COLORS_GATV2, linewidth=2,
                markersize=8, label="GATv2 (h128, L10)",
                zorder=3)
        for m in gatv2:
            ax.annotate(
                f'{m["metrics"]["gesamt"]["R2"]:.3f}',
                (m["n_nodes"], m["metrics"]["gesamt"]["R2"]),
                textcoords="offset points", xytext=(0, -16),
                fontsize=8, ha="center", color=COLORS_GATV2)

    ax.set_xlabel("Knoten pro Graph", fontsize=11)
    ax.set_ylabel("R² (Test, gesamt)", fontsize=11)
    ax.set_title("Skalierungsverhalten: R² vs. Graphaufloesung",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, _: f"{x / 1e6:.1f}M" if x >= 1e6
            else f"{x / 1e3:.0f}k"))
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path + "_scaling_r2.pdf",
                bbox_inches="tight", dpi=300)
    fig.savefig(output_path + "_scaling_r2.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Plot: {output_path}_scaling_r2.pdf")


# ======================================================================
# Plot 3: rL2 Skalierungskurve  (nicht mehr aktiv erzeugt)
# ======================================================================

def plot_scaling_rl2(models, output_path):
    """
    rL2 ueber Knotenanzahl, getrennt nach GCN und GATv2.
    """
    gcn = sorted([m for m in models if "GCN" in m["name"]
                  and "GATv2" not in m["name"]],
                 key=lambda m: m["n_nodes"])
    gatv2 = sorted([m for m in models if "GATv2" in m["name"]],
                   key=lambda m: m["n_nodes"])

    fig, ax = plt.subplots(figsize=(8, 5))

    if gcn:
        nodes = [m["n_nodes"] for m in gcn]
        rl2 = [m["metrics"]["gesamt"]["rL2"] for m in gcn]
        ax.plot(nodes, rl2, "o-", color=COLORS_GCN, linewidth=2,
                markersize=8, label="GCN (h128, L10)",
                zorder=3)
        for m in gcn:
            ax.annotate(
                f'{m["metrics"]["gesamt"]["rL2"]:.3f}',
                (m["n_nodes"], m["metrics"]["gesamt"]["rL2"]),
                textcoords="offset points", xytext=(0, -16),
                fontsize=8, ha="center", color=COLORS_GCN)

    if gatv2:
        nodes = [m["n_nodes"] for m in gatv2]
        rl2 = [m["metrics"]["gesamt"]["rL2"] for m in gatv2]
        ax.plot(nodes, rl2, "s-", color=COLORS_GATV2, linewidth=2,
                markersize=8, label="GATv2 (h128, L10)",
                zorder=3)
        for m in gatv2:
            ax.annotate(
                f'{m["metrics"]["gesamt"]["rL2"]:.3f}',
                (m["n_nodes"], m["metrics"]["gesamt"]["rL2"]),
                textcoords="offset points", xytext=(0, 12),
                fontsize=8, ha="center", color=COLORS_GATV2)

    ax.set_xlabel("Knoten pro Graph", fontsize=11)
    ax.set_ylabel("rL2 (Test, gesamt)", fontsize=11)
    ax.set_title("Skalierungsverhalten: rL2 vs. Graphaufloesung",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, _: f"{x / 1e6:.1f}M" if x >= 1e6
            else f"{x / 1e3:.0f}k"))
    ax.invert_yaxis()
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path + "_scaling_rl2.pdf",
                bbox_inches="tight", dpi=300)
    fig.savefig(output_path + "_scaling_rl2.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Plot: {output_path}_scaling_rl2.pdf")


# ======================================================================
# Plot 4: Kosten-Nutzen (Trainingszeit vs. R²)  (nicht mehr aktiv erzeugt)
# ======================================================================

def plot_cost_benefit(models, output_path):
    """
    Scatter-Plot: Trainingszeit vs. R², Groesse = Knotenanzahl.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for m in models:
        r2 = m["metrics"]["gesamt"]["R2"]
        t_h = m["train_time_min"] / 60
        size = np.sqrt(m["n_nodes"]) / 10  # Groesse skaliert
        color = get_color(m["name"])
        marker = get_marker(m["name"])

        ax.scatter(t_h, r2, s=size, c=color, marker=marker,
                   alpha=0.8, edgecolors="white", linewidth=0.5,
                   zorder=3)
        ax.annotate(
            m["name"], (t_h, r2),
            textcoords="offset points", xytext=(8, 0),
            fontsize=7.5, color="#374151")

    ax.set_xlabel("Trainingszeit [h]", fontsize=11)
    ax.set_ylabel("R² (Test, gesamt)", fontsize=11)
    ax.set_title("Kosten-Nutzen: Trainingszeit vs. Genauigkeit",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Legende manuell
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLORS_GCN, markersize=10,
               label="GCN"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=COLORS_GATV2, markersize=10,
               label="GATv2"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path + "_cost_benefit.pdf",
                bbox_inches="tight", dpi=300)
    fig.savefig(output_path + "_cost_benefit.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Plot: {output_path}_cost_benefit.pdf")


# ======================================================================
# Plot 5: Epsilon R² (schwierigstes Feld)
# ======================================================================

def plot_epsilon(models, output_path):
    """
    Balkendiagramm: epsilon-R² fuer alle Modelle. Ohne Titel; epsilon als
    Mathtext im Achsentitel; Dezimalkomma; Wertelabels rechts ausserhalb der
    Balken (kollisionsfrei).
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    names = [m["name"] for m in models]
    eps_r2 = [m["metrics"]["epsilon"]["R2"] for m in models]
    colors = [model_color(n) for n in names]
    y = np.arange(len(models))

    ax.barh(y, eps_r2, color=colors, edgecolor="white", linewidth=0.4)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel(r"R² ($\varepsilon$)")
    ax.set_xlim(0.70, 0.95)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(komma_formatter(2))
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.invert_yaxis()

    # Werte rechts ausserhalb der Balken (einheitlich drei Nachkommastellen).
    for yi, val in zip(y, eps_r2):
        ax.text(val + 0.004, yi, komma(val, 3),
                ha="left", va="center", fontsize=9, color="#111827")

    fig.tight_layout()
    fig.savefig(output_path + "_epsilon.pdf", bbox_inches="tight")
    fig.savefig(output_path + "_epsilon.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {output_path}_epsilon.png")


# ======================================================================
# Hauptprogramm
# ======================================================================

def main():
    """Laedt die Modelle und erzeugt die beiden aktiven Vergleichsabbildungen."""
    parser = argparse.ArgumentParser(
        description="Vergleichsplots fuer GNN-Modelle")

    parser.add_argument(
        "--model", nargs=5, action="append", metavar=(
            "DIR", "NAME", "NODES", "TIME_MIN", "PARAMS"),
        help="Modell: <verzeichnis> <name> <knoten> "
             "<zeit_min> <parameter>")
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML-Konfigurationsdatei")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Ausgabe-Basispfad (Standard: "
             f"{FIG_DIR}/comparison)")

    args = parser.parse_args()

    setup_style()

    # Modelle laden (sechs Kernmodelle)
    models = []
    if args.config:
        models = load_from_config(args.config)
    elif args.model:
        for m in args.model:
            data = load_model_data(
                m[0], m[1], int(m[2]), float(m[3]), int(m[4]))
            if data is not None:
                models.append(data)
    else:
        print("FEHLER: --model oder --config angeben.")
        sys.exit(1)

    if len(models) == 0:
        print("FEHLER: Keine Modelle geladen.")
        sys.exit(1)

    # Ablation no_cellvol als zusaetzliche Serie fuer die beiden Abbildungen.
    nc = load_model_data(NO_CELLVOL["dir"], NO_CELLVOL["name"],
                         NO_CELLVOL["nodes"], NO_CELLVOL["time_min"],
                         NO_CELLVOL["params"])
    models_fig = models + [nc] if nc is not None else models

    out_base = args.output or os.path.join(FIG_DIR, "comparison")
    os.makedirs(os.path.dirname(out_base), exist_ok=True)

    print(f"\n  {len(models_fig)} Modelle geladen (inkl. no_cellvol).")
    print(f"  Erzeuge Abbildungen nach {os.path.dirname(out_base)} ...\n")

    plot_r2_fields(models_fig, out_base)
    plot_epsilon(models_fig, out_base)

    # Hinweis: plot_scaling_r2 / plot_scaling_rl2 / plot_cost_benefit werden
    # bewusst nicht mehr erzeugt (nicht mehr benoetigt), bleiben aber als
    # Funktionen erhalten und koennen bei Bedarf manuell aufgerufen werden.

    print(f"\n  Fertig. Abbildungen: {out_base}_r2_fields.* / "
          f"{out_base}_epsilon.*\n")


if __name__ == "__main__":
    main()
