#!/usr/bin/env python3
"""Extraktion der Min/Max-Werte der Vorhersagen je Subsampling-Level.

Pendant zu ``extract_gt_minmax.py``: statt der Ground Truth aus den
Graph-Datensaetzen werden hier die Modell-Vorhersagen aus den sparse
VTU-Dateien (``vorhersage.vtu``) ausgewertet. Diese liegen auf denselben
subgesampleten Knoten wie die Ground Truth vor.

Felder:
    |U| = sqrt(Ux^2 + Uy^2 + Uz^2)   (Geschwindigkeitsmagnitude)
    p                                 (Druck)
    k                                 (turbulente kinetische Energie)
    epsilon                           (Dissipationsrate)

Die ``|U|``-Magnitude wird knotenweise aus den Komponenten ``U`` berechnet
(min/max der Magnitude, nicht Magnitude der Komponenten-Extrema) — konsistent
zur Ground-Truth-Auswertung.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import pyvista as pv

# Ausgabereihenfolge der Felder.
FIELDS = ["|U|", "p", "k", "epsilon"]

# Standard-Pfade der sparse Vorhersage-VTUs (GCN-Rerun, sim_013).
_PRED_BASE = "/home/tbergermann/Python/predictions/prediction_sim_013_gcn_rerun"
DEFAULT_PATHS = {
    "coarse": os.path.join(_PRED_BASE, "gcn_coarse",  "vorhersage.vtu"),
    "medium": os.path.join(_PRED_BASE, "gcn_medium",  "vorhersage.vtu"),
    "bf_25":  os.path.join(_PRED_BASE, "gcn_bf25",    "vorhersage.vtu"),
}


def load_fields(vtu_path: str) -> dict[str, np.ndarray]:
    """Lade die Vorhersagefelder aus einer sparse VTU-Datei.

    Parameters
    ----------
    vtu_path : str
        Pfad zur ``vorhersage.vtu``.

    Returns
    -------
    dict
        ``{"|U|": (N,), "p": (N,), "k": (N,), "epsilon": (N,)}`` als float64.

    Raises
    ------
    FileNotFoundError
        Falls die Datei nicht existiert.
    KeyError
        Falls ein erwartetes Feld fehlt.
    """
    if not os.path.isfile(vtu_path):
        raise FileNotFoundError(f"vorhersage.vtu nicht gefunden: {vtu_path}")

    mesh = pv.read(vtu_path)
    for name in ("U", "p", "k", "epsilon"):
        if name not in mesh.point_data:
            raise KeyError(
                f"Feld '{name}' fehlt in {vtu_path}. "
                f"Vorhanden: {list(mesh.point_data.keys())}"
            )

    u = np.asarray(mesh["U"], dtype=np.float64)
    u_mag = np.sqrt(np.sum(u ** 2, axis=1))  # knotenweise Magnitude (N,)
    return {
        "|U|": u_mag,
        "p": np.asarray(mesh["p"], dtype=np.float64),
        "k": np.asarray(mesh["k"], dtype=np.float64),
        "epsilon": np.asarray(mesh["epsilon"], dtype=np.float64),
    }


def compute_minmax(fields: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Berechne min/max je Feld."""
    return {
        name: {"min": float(np.min(vals)), "max": float(np.max(vals))}
        for name, vals in fields.items()
    }


def collect_results(paths: dict[str, str]):
    """Sammle min/max je Level und Feld."""
    results: dict[str, dict] = {}
    for level, path in paths.items():
        fields = load_fields(path)
        n = len(fields["|U|"])
        results[level] = compute_minmax(fields)
        print(f"  [{level:6s}] {path}  (N={n})")
    return results


def print_table(results: dict, sim_id: str) -> None:
    """Gib die Ergebnisse als formatierte Tabelle auf der Konsole aus."""
    levels = list(results.keys())
    print(f"\nMin/Max der Vorhersagen fuer {sim_id}\n")

    header = f"{'Feld':<10}" + "".join(
        f"{lvl + ' min':>16}{lvl + ' max':>16}" for lvl in levels
    )
    print(header)
    print("-" * len(header))
    for field in FIELDS:
        row = f"{field:<10}"
        for lvl in levels:
            row += f"{results[lvl][field]['min']:>16.6g}"
            row += f"{results[lvl][field]['max']:>16.6g}"
        print(row)
    print()


def write_csv(results: dict, sim_id: str, csv_path: str) -> None:
    """Schreibe die Ergebnisse im Long-Format mit voller Genauigkeit als CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sim_id", "level", "field", "min", "max"])
        for level, fields in results.items():
            for field in FIELDS:
                writer.writerow([
                    sim_id, level, field,
                    repr(fields[field]["min"]),
                    repr(fields[field]["max"]),
                ])
    print(f"CSV  geschrieben: {csv_path}")


def write_markdown(results: dict, sim_id: str, md_path: str,
                   model_label: str) -> None:
    """Schreibe eine lesbar formatierte Markdown-Tabelle (Feld x Level)."""
    levels = list(results.keys())
    with open(md_path, "w") as f:
        f.write(f"# Min/Max der Vorhersagen ({model_label}) — {sim_id}\n\n")
        f.write(
            "Vergleich der vorhergesagten Wertebereiche der Zielfelder ueber "
            "die drei Subsampling-Level.\n\n"
        )

        head = ["Feld"] + [f"{lvl} min / max" for lvl in levels]
        f.write("| " + " | ".join(head) + " |\n")
        f.write("|" + "|".join(["---"] * len(head)) + "|\n")
        for field in FIELDS:
            # Pipe-Zeichen im Feldnamen (z. B. |U|) escapen.
            cells = [field.replace("|", "\\|")]
            for lvl in levels:
                mn = results[lvl][field]["min"]
                mx = results[lvl][field]["max"]
                cells.append(f"{mn:.4g} / {mx:.4g}")
            f.write("| " + " | ".join(cells) + " |\n")
    print(f"MD   geschrieben: {md_path}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Min/Max der Vorhersagen je Subsampling-Level (sparse VTU)."
    )
    parser.add_argument("--coarse", default=DEFAULT_PATHS["coarse"],
                        help="Pfad zur vorhersage.vtu des Coarse-Levels.")
    parser.add_argument("--medium", default=DEFAULT_PATHS["medium"],
                        help="Pfad zur vorhersage.vtu des Medium-Levels.")
    parser.add_argument("--bf25", default=DEFAULT_PATHS["bf_25"],
                        help="Pfad zur vorhersage.vtu des bf_25-Levels.")
    parser.add_argument("--sim-id", default="sim_013",
                        help="Label der Simulation fuer Ausgaben (Default: sim_013).")
    parser.add_argument("--model-label", default="GCN-Rerun",
                        help="Modell-Label fuer die Markdown-Ueberschrift.")
    parser.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Ausgabeverzeichnis fuer CSV und Markdown.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    paths = {
        "coarse": args.coarse,
        "medium": args.medium,
        "bf_25": args.bf25,
    }

    print(f"Lade Vorhersage-VTUs fuer {args.sim_id}:")
    try:
        results = collect_results(paths)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"\nFEHLER: {exc}", file=sys.stderr)
        return 1

    print_table(results, args.sim_id)

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, f"pred_minmax_{args.sim_id}.csv")
    md_path = os.path.join(args.outdir, f"pred_minmax_{args.sim_id}.md")
    write_csv(results, args.sim_id, csv_path)
    write_markdown(results, args.sim_id, md_path, args.model_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
