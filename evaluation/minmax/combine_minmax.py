#!/usr/bin/env python3
"""Kombiniere Ground-Truth- und Vorhersage-Min/Max zu einer Markdown-Tabelle.

Liest die im Long-Format vorliegenden CSVs (Spalten: sim_id, level, field,
min, max) von Ground Truth und beliebig vielen Modellen und erzeugt eine
gemeinsame, lesbar formatierte Markdown-Tabelle (Feld x Level x Quelle) fuer
den Ergebnisteil der Masterarbeit.

Verwendung:
    combine_minmax.py --sim-id sim_013 --output gt_vs_pred_sim_013.md \\
        --source "Ground Truth" gt_minmax_sim_013.csv \\
        --source "GCN-Rerun"    pred_minmax_sim_013.csv \\
        --source "GATv2"        gatv2/pred_minmax_sim_013.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

FIELDS = ["|U|", "p", "k", "epsilon"]


def read_csv(path: str) -> dict[tuple[str, str], tuple[float, float]]:
    """Lies eine Min/Max-CSV im Long-Format.

    Returns
    -------
    dict
        ``{(level, field): (min, max)}``.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV nicht gefunden: {path}")
    out: dict[tuple[str, str], tuple[float, float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[(row["level"], row["field"])] = (
                float(row["min"]), float(row["max"]),
            )
    return out


def write_markdown(sim_id: str, sources: list[tuple[str, dict]],
                   md_path: str) -> None:
    """Schreibe die kombinierte Markdown-Tabelle.

    Parameters
    ----------
    sources : list of (label, data)
        Quellenname und gelesene CSV-Daten; Reihenfolge = Spaltenreihenfolge.
    """
    # Level-Reihenfolge aus der ersten Quelle uebernehmen.
    levels = []
    for (level, _field) in sources[0][1].keys():
        if level not in levels:
            levels.append(level)

    head = ["Feld", "Level"] + [f"{label} (min / max)" for label, _ in sources]
    lines = [
        f"# Min/Max Ground Truth vs. Vorhersage — {sim_id}\n",
        "Wertebereiche der Zielfelder je Subsampling-Level. `|U|` ist die "
        "knotenweise Geschwindigkeitsmagnitude (min/max der Magnitude).\n",
        "| " + " | ".join(head) + " |",
        "|" + "|".join(["---"] * len(head)) + "|",
    ]
    for field in FIELDS:
        for level in levels:
            cells = [field.replace("|", "\\|"), level]
            for _label, data in sources:
                mn, mx = data[(level, field)]
                cells.append(f"{mn:.4g} / {mx:.4g}")
            lines.append("| " + " | ".join(cells) + " |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"MD geschrieben: {md_path}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kombiniere GT- und Vorhersage-Min/Max-CSVs zu einer "
                    "Markdown-Tabelle."
    )
    parser.add_argument("--sim-id", default="sim_013",
                        help="Simulations-Label fuer die Ueberschrift.")
    parser.add_argument("--output", required=True,
                        help="Pfad der zu schreibenden Markdown-Datei.")
    parser.add_argument("--source", nargs=2, action="append", required=True,
                        metavar=("LABEL", "CSV"),
                        help="Quellenname und CSV-Pfad. Mehrfach angebbar; "
                             "Reihenfolge = Spaltenreihenfolge.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        sources = [(label, read_csv(path)) for label, path in args.source]
    except FileNotFoundError as exc:
        print(f"\nFEHLER: {exc}", file=sys.stderr)
        return 1
    write_markdown(args.sim_id, sources, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
