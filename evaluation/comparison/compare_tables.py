"""
compare_tables.py
=================
Erzeugt Vergleichstabellen fuer alle trainierten GNN-Modelle.

Liest test_metrics.json Dateien aus den Output-Verzeichnissen und
erzeugt:
    1. Gesamtuebersicht: R², rL2, Trainingszeit, Parameter
    2. Detail-Tabelle: R² pro Feld fuer alle Modelle
    3. Ranking: Bestes Modell pro Feld und gesamt

Ausgabe als Markdown (.md) und LaTeX (.tex).

Verwendung:
    python compare_tables.py \\
        --model output_gcn_coarse "GCN Coarse" 50000 168 236422 \\
        --model output_gatv2_coarse_h128 "GATv2 Coarse" 50000 137 236000 \\
        --model output_gcn_medium "GCN Medium" 507000 168 236422 \\
        ...

    Jedes --model erwartet: <verzeichnis> <name> <knoten> <zeit_min> <parameter>

Kurzform mit Konfig-Datei:
    python compare_tables.py --config models.yaml
"""

import os
import sys
import json
import argparse

import yaml


# ======================================================================
# Daten laden
# ======================================================================

def load_model_data(model_dir, name, n_nodes, train_time_min,
                    n_params):
    """
    Laedt test_metrics.json und kombiniert mit Metadaten.

    Parameter
    ---------
    model_dir : str
        Verzeichnis mit test_metrics.json.
    name : str
        Anzeigename des Modells.
    n_nodes : int
        Durchschnittliche Knotenanzahl pro Graph.
    train_time_min : float
        Trainingszeit in Minuten.
    n_params : int
        Anzahl trainierbarer Parameter.

    Rueckgabe
    ---------
    dict : Modell-Daten mit Metriken und Metadaten.
    """
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
    """
    Laedt Modell-Definitionen aus einer YAML-Konfigurationsdatei.

    Format:
        models:
          - dir: output_gcn_coarse
            name: "GCN Coarse"
            nodes: 50000
            time_min: 55
            params: 236422
          - dir: output_gatv2_coarse_h128
            name: "GATv2 Coarse"
            ...

    Parameter
    ---------
    config_path : str
        Pfad zur YAML-Datei.

    Rueckgabe
    ---------
    list[dict] : Liste von Modell-Daten.
    """
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
# Tabellen erzeugen
# ======================================================================

FIELD_NAMES = ["Ux", "Uy", "Uz", "p", "k", "epsilon"]


def generate_overview_table(models):
    """
    Erzeugt die Gesamtuebersicht als Markdown und LaTeX.

    Parameter
    ---------
    models : list[dict]
        Liste von Modell-Daten.

    Rueckgabe
    ---------
    tuple : (markdown_str, latex_str)
    """
    # Bestes Modell finden
    best_r2_idx = max(range(len(models)),
                      key=lambda i: models[i]["metrics"]["gesamt"]["R2"])
    best_rl2_idx = min(range(len(models)),
                       key=lambda i: models[i]["metrics"]["gesamt"]["rL2"])

    # Markdown
    md = []
    md.append("## Gesamtuebersicht\n")
    md.append("| Modell | Knoten/Graph | Parameter | "
              "Trainingszeit | R² | rL2 |")
    md.append("|--------|-------------|-----------|"
              "--------------|-----|-----|")

    for i, m in enumerate(models):
        r2 = m["metrics"]["gesamt"]["R2"]
        rl2 = m["metrics"]["gesamt"]["rL2"]

        # Zeitformatierung
        t = m["train_time_min"]
        if t >= 60:
            time_str = f"{t / 60:.1f} h"
        else:
            time_str = f"{t:.0f} min"

        # Beste markieren
        r2_str = f"**{r2:.4f}**" if i == best_r2_idx else f"{r2:.4f}"
        rl2_str = f"**{rl2:.4f}**" if i == best_rl2_idx else f"{rl2:.4f}"

        md.append(
            f"| {m['name']} "
            f"| {m['n_nodes']:,} "
            f"| {m['n_params']:,} "
            f"| {time_str} "
            f"| {r2_str} "
            f"| {rl2_str} |")

    md.append("")
    md.append(f"Bestes R²: **{models[best_r2_idx]['name']}** "
              f"({models[best_r2_idx]['metrics']['gesamt']['R2']:.4f})")
    md.append(f"Bestes rL2: **{models[best_rl2_idx]['name']}** "
              f"({models[best_rl2_idx]['metrics']['gesamt']['rL2']:.4f})")

    # LaTeX
    tex = []
    tex.append("\\begin{table}[htbp]")
    tex.append("\\centering")
    tex.append("\\caption{Gesamtuebersicht aller Modelle}")
    tex.append("\\label{tab:model_overview}")
    tex.append("\\begin{tabular}{l r r r r r}")
    tex.append("\\toprule")
    tex.append("Modell & Knoten & Parameter & "
               "Trainingszeit & $R^2$ & $rL_2$ \\\\")
    tex.append("\\midrule")

    for i, m in enumerate(models):
        r2 = m["metrics"]["gesamt"]["R2"]
        rl2 = m["metrics"]["gesamt"]["rL2"]
        t = m["train_time_min"]
        time_str = (f"{t / 60:.1f}\\,h" if t >= 60
                    else f"{t:.0f}\\,min")

        r2_str = (f"\\textbf{{{r2:.4f}}}" if i == best_r2_idx
                  else f"{r2:.4f}")
        rl2_str = (f"\\textbf{{{rl2:.4f}}}" if i == best_rl2_idx
                   else f"{rl2:.4f}")

        tex.append(
            f"{m['name']} & {m['n_nodes']:,} & {m['n_params']:,} & "
            f"{time_str} & {r2_str} & {rl2_str} \\\\")

    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}")

    return "\n".join(md), "\n".join(tex)


def generate_field_table(models):
    """
    Erzeugt die Detail-Tabelle (R² pro Feld) als Markdown und LaTeX.

    Parameter
    ---------
    models : list[dict]
        Liste von Modell-Daten.

    Rueckgabe
    ---------
    tuple : (markdown_str, latex_str)
    """
    # Bestes Modell pro Feld finden
    best_per_field = {}
    for field in FIELD_NAMES + ["gesamt"]:
        best_idx = max(range(len(models)),
                       key=lambda i: models[i]["metrics"][field]["R2"])
        best_per_field[field] = best_idx

    # Markdown
    md = []
    md.append("## R² pro Feld\n")

    header = "| Modell |"
    sep = "|--------|"
    for f in FIELD_NAMES + ["Gesamt"]:
        header += f" {f} |"
        sep += "------|"
    md.append(header)
    md.append(sep)

    for i, m in enumerate(models):
        row = f"| {m['name']} |"
        for field in FIELD_NAMES:
            r2 = m["metrics"][field]["R2"]
            if i == best_per_field[field]:
                row += f" **{r2:.4f}** |"
            else:
                row += f" {r2:.4f} |"
        # Gesamt
        r2_ges = m["metrics"]["gesamt"]["R2"]
        if i == best_per_field["gesamt"]:
            row += f" **{r2_ges:.4f}** |"
        else:
            row += f" {r2_ges:.4f} |"
        md.append(row)

    # Ranking
    md.append("")
    md.append("### Ranking pro Feld\n")
    for field in FIELD_NAMES + ["gesamt"]:
        idx = best_per_field[field]
        r2 = models[idx]["metrics"][field]["R2"]
        label = field if field != "gesamt" else "Gesamt"
        md.append(f"- **{label}:** {models[idx]['name']} "
                  f"(R²={r2:.4f})")

    # LaTeX
    tex = []
    tex.append("\\begin{table}[htbp]")
    tex.append("\\centering")
    tex.append("\\caption{$R^2$ pro Stroemungsfeld}")
    tex.append("\\label{tab:r2_per_field}")
    n_cols = len(FIELD_NAMES) + 2  # Modell + Felder + Gesamt
    col_fmt = "l " + "r " * (n_cols - 1)
    tex.append(f"\\begin{{tabular}}{{{col_fmt.strip()}}}")
    tex.append("\\toprule")

    header_tex = "Modell"
    for f in FIELD_NAMES:
        header_tex += f" & ${f}$"
    header_tex += " & Gesamt \\\\"
    # Fix epsilon display
    header_tex = header_tex.replace("$epsilon$", "$\\varepsilon$")
    tex.append(header_tex)
    tex.append("\\midrule")

    for i, m in enumerate(models):
        row_tex = f"{m['name']}"
        for field in FIELD_NAMES:
            r2 = m["metrics"][field]["R2"]
            if i == best_per_field[field]:
                row_tex += f" & \\textbf{{{r2:.4f}}}"
            else:
                row_tex += f" & {r2:.4f}"
        r2_ges = m["metrics"]["gesamt"]["R2"]
        if i == best_per_field["gesamt"]:
            row_tex += f" & \\textbf{{{r2_ges:.4f}}}"
        else:
            row_tex += f" & {r2_ges:.4f}"
        row_tex += " \\\\"
        tex.append(row_tex)

    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}")

    return "\n".join(md), "\n".join(tex)


def generate_rl2_table(models):
    """
    Erzeugt die Detail-Tabelle (rL2 pro Feld) als Markdown und LaTeX.

    Parameter
    ---------
    models : list[dict]
        Liste von Modell-Daten.

    Rueckgabe
    ---------
    tuple : (markdown_str, latex_str)
    """
    best_per_field = {}
    for field in FIELD_NAMES + ["gesamt"]:
        best_idx = min(range(len(models)),
                       key=lambda i: models[i]["metrics"][field]["rL2"])
        best_per_field[field] = best_idx

    # Markdown
    md = []
    md.append("## rL2 pro Feld\n")

    header = "| Modell |"
    sep = "|--------|"
    for f in FIELD_NAMES + ["Gesamt"]:
        header += f" {f} |"
        sep += "------|"
    md.append(header)
    md.append(sep)

    for i, m in enumerate(models):
        row = f"| {m['name']} |"
        for field in FIELD_NAMES:
            rl2 = m["metrics"][field]["rL2"]
            if i == best_per_field[field]:
                row += f" **{rl2:.4f}** |"
            else:
                row += f" {rl2:.4f} |"
        rl2_ges = m["metrics"]["gesamt"]["rL2"]
        if i == best_per_field["gesamt"]:
            row += f" **{rl2_ges:.4f}** |"
        else:
            row += f" {rl2_ges:.4f} |"
        md.append(row)

    # LaTeX
    tex = []
    tex.append("\\begin{table}[htbp]")
    tex.append("\\centering")
    tex.append("\\caption{Relative $L_2$-Norm pro Stroemungsfeld}")
    tex.append("\\label{tab:rl2_per_field}")
    n_cols = len(FIELD_NAMES) + 2
    col_fmt = "l " + "r " * (n_cols - 1)
    tex.append(f"\\begin{{tabular}}{{{col_fmt.strip()}}}")
    tex.append("\\toprule")

    header_tex = "Modell"
    for f in FIELD_NAMES:
        header_tex += f" & ${f}$"
    header_tex += " & Gesamt \\\\"
    header_tex = header_tex.replace("$epsilon$", "$\\varepsilon$")
    tex.append(header_tex)
    tex.append("\\midrule")

    for i, m in enumerate(models):
        row_tex = f"{m['name']}"
        for field in FIELD_NAMES:
            rl2 = m["metrics"][field]["rL2"]
            if i == best_per_field[field]:
                row_tex += f" & \\textbf{{{rl2:.4f}}}"
            else:
                row_tex += f" & {rl2:.4f}"
        rl2_ges = m["metrics"]["gesamt"]["rL2"]
        if i == best_per_field["gesamt"]:
            row_tex += f" & \\textbf{{{rl2_ges:.4f}}}"
        else:
            row_tex += f" & {rl2_ges:.4f}"
        row_tex += " \\\\"
        tex.append(row_tex)

    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}")

    return "\n".join(md), "\n".join(tex)


def generate_scaling_table(models):
    """
    Erzeugt eine Skalierungstabelle: R² Verbesserung pro Stufe.

    Parameter
    ---------
    models : list[dict]
        Liste von Modell-Daten (sollte nach Architektur und
        Knotenanzahl sortiert sein).

    Rueckgabe
    ---------
    str : Markdown-Tabelle
    """
    # Nach Architektur gruppieren
    gcn = [m for m in models if "GCN" in m["name"]
           and "GATv2" not in m["name"]]
    gatv2 = [m for m in models if "GATv2" in m["name"]]

    gcn.sort(key=lambda m: m["n_nodes"])
    gatv2.sort(key=lambda m: m["n_nodes"])

    md = []
    md.append("## Skalierungsvergleich\n")
    md.append("| Level | Knoten | GCN R² | GCN Δ | "
              "GATv2 R² | GATv2 Δ | GATv2 Vorteil |")
    md.append("|-------|--------|--------|-------|"
              "----------|---------|---------------|")

    max_levels = max(len(gcn), len(gatv2))
    for i in range(max_levels):
        # GCN-Daten
        if i < len(gcn):
            gcn_name = gcn[i]["name"].replace("GCN ", "")
            gcn_nodes = gcn[i]["n_nodes"]
            gcn_r2 = gcn[i]["metrics"]["gesamt"]["R2"]
            gcn_delta = (f"+{gcn_r2 - gcn[i-1]['metrics']['gesamt']['R2']:.4f}"
                         if i > 0 else "—")
        else:
            gcn_name = "—"
            gcn_nodes = 0
            gcn_r2 = None
            gcn_delta = "—"

        # GATv2-Daten
        if i < len(gatv2):
            gatv2_r2 = gatv2[i]["metrics"]["gesamt"]["R2"]
            gatv2_delta = (f"+{gatv2_r2 - gatv2[i-1]['metrics']['gesamt']['R2']:.4f}"
                           if i > 0 else "—")
        else:
            gatv2_r2 = None
            gatv2_delta = "—"

        # Vorteil
        if gcn_r2 is not None and gatv2_r2 is not None:
            diff = gatv2_r2 - gcn_r2
            vorteil = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        else:
            vorteil = "—"

        # Knoten (nehme das Maximum)
        nodes = max(gcn_nodes,
                    gatv2[i]["n_nodes"] if i < len(gatv2) else 0)

        gcn_r2_str = f"{gcn_r2:.4f}" if gcn_r2 is not None else "—"
        gatv2_r2_str = f"{gatv2_r2:.4f}" if gatv2_r2 is not None else "—"

        level = gcn_name if gcn_name != "—" else (
            gatv2[i]["name"].replace("GATv2 ", "") if i < len(gatv2)
            else "—")

        md.append(
            f"| {level} | {nodes:,} | {gcn_r2_str} | {gcn_delta} | "
            f"{gatv2_r2_str} | {gatv2_delta} | {vorteil} |")

    return "\n".join(md)


def generate_cost_table(models):
    """
    Erzeugt eine Kosten-Nutzen-Tabelle.

    Parameter
    ---------
    models : list[dict]
        Liste von Modell-Daten.

    Rueckgabe
    ---------
    str : Markdown-Tabelle
    """
    md = []
    md.append("## Kosten-Nutzen Vergleich\n")
    md.append("| Modell | R² | Trainingszeit | "
              "R² pro Stunde | Knoten |")
    md.append("|--------|-----|---------------|"
              "--------------|--------|")

    for m in models:
        r2 = m["metrics"]["gesamt"]["R2"]
        t_min = m["train_time_min"]
        t_h = t_min / 60
        r2_per_h = r2 / t_h if t_h > 0 else 0

        time_str = (f"{t_h:.1f} h" if t_h >= 1
                    else f"{t_min:.0f} min")

        md.append(
            f"| {m['name']} | {r2:.4f} | {time_str} | "
            f"{r2_per_h:.4f} | {m['n_nodes']:,} |")

    return "\n".join(md)


# ======================================================================
# Hauptprogramm
# ======================================================================

def main():
    """Laedt die Modelle und schreibt alle Vergleichstabellen (Markdown, LaTeX)."""
    parser = argparse.ArgumentParser(
        description="Vergleichstabellen fuer GNN-Modelle")

    parser.add_argument(
        "--model", nargs=5, action="append", metavar=(
            "DIR", "NAME", "NODES", "TIME_MIN", "PARAMS"),
        help="Modell hinzufuegen: <verzeichnis> <name> "
             "<knoten> <zeit_min> <parameter>")
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML-Konfigurationsdatei mit Modell-Definitionen")
    parser.add_argument(
        "--output", type=str, default="comparison",
        help="Ausgabe-Basispfad (Standard: comparison)")

    args = parser.parse_args()

    # Modelle laden
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

    print(f"\n  {len(models)} Modelle geladen.")

    # Tabellen erzeugen
    overview_md, overview_tex = generate_overview_table(models)
    field_md, field_tex = generate_field_table(models)
    rl2_md, rl2_tex = generate_rl2_table(models)
    scaling_md = generate_scaling_table(models)
    cost_md = generate_cost_table(models)

    # Markdown-Datei schreiben
    md_path = args.output + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# GNN-Modellvergleich\n\n")
        f.write(overview_md + "\n\n")
        f.write(field_md + "\n\n")
        f.write(rl2_md + "\n\n")
        f.write(scaling_md + "\n\n")
        f.write(cost_md + "\n")

    print(f"  Markdown: {md_path}")

    # LaTeX-Datei schreiben
    tex_path = args.output + ".tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% GNN-Modellvergleich — automatisch generiert\n")
        f.write("% Einbinden mit: \\input{" + args.output + "}\n\n")
        f.write(overview_tex + "\n\n")
        f.write(field_tex + "\n\n")
        f.write(rl2_tex + "\n")

    print(f"  LaTeX:    {tex_path}")
    print("  Fertig.\n")


if __name__ == "__main__":
    main()