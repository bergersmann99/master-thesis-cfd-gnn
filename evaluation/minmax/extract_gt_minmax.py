#!/usr/bin/env python3
"""Extraktion der Min/Max-Werte der subgesampleten Ground Truth.

Fuer eine gegebene Simulation (Default: sim_013) werden aus den drei
Subsampling-Leveln (Coarse, Medium, bf_25) die Wertebereiche der
physikalischen Zielfelder bestimmt:

    |U| = sqrt(Ux^2 + Uy^2 + Uz^2)   (Geschwindigkeitsmagnitude)
    p                                 (Druck)
    k                                 (turbulente kinetische Energie)
    epsilon                           (Dissipationsrate)

Die Werte dienen im Ergebnisteil der Masterarbeit dem Vergleich, welches
Subsampling-Level fuer die Aufloesung der Felder benoetigt wird.

Datengrundlage
--------------
Jedes Level liegt als eigener Graph-Datensatz vor (train/val/test.pt), erzeugt
mit ``createGraphDataset.py``. Die ``test.pt`` ist eine Liste von
``torch_geometric.data.Data``-Objekten; jeder Graph traegt das Attribut
``data.sim_id`` und den Ziel-Tensor ``data.y`` der Form ``(N, 6)``.

Spaltenkonvention von ``y`` (verifiziert an ``createGraphDataset.py``,
Funktion ``process_simulation``, Zeilen 803-809):

    0 = Ux, 1 = Uy, 2 = Uz, 3 = p, 4 = k, 5 = epsilon

Der gesuchte Graph wird ueber ``data.sim_id`` gefiltert (nicht ueber den
Index), da die Reihenfolge der Graphen je Level variiert.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

# Spaltenindizes des Ziel-Tensors y, verifiziert am Quellcode von
# createGraphDataset.py (process_simulation). NICHT aus dem Gedaechtnis.
COL_UX, COL_UY, COL_UZ = 0, 1, 2
COL_P, COL_K, COL_EPS = 3, 4, 5

# Ausgabereihenfolge der Felder.
FIELDS = ["|U|", "p", "k", "epsilon"]

# Standard-Pfade der drei lokal vorliegenden Level-Datensaetze (Rerun).
DEFAULT_PATHS = {
    "coarse": "/home/tbergermann/Python/GNN/graph_dataset_coarse_rerun/test.pt",
    "medium": "/home/tbergermann/Python/GNN/graph_dataset_medium_rerun/test.pt",
    "bf_25":  "/home/tbergermann/Python/GNN/graph_dataset_bf25_rerun/test.pt",
}


def find_graph(test_path: str, sim_id: str):
    """Lade eine test.pt und liefere den Graphen mit passender sim_id.

    Parameters
    ----------
    test_path : str
        Pfad zur ``test.pt`` (Liste von ``Data``-Objekten).
    sim_id : str
        Gesuchte Simulations-ID, z. B. ``"sim_013"``.

    Returns
    -------
    torch_geometric.data.Data
        Der zur ``sim_id`` gehoerende Graph.

    Raises
    ------
    FileNotFoundError
        Falls die Datei nicht existiert.
    KeyError
        Falls kein Graph mit der gesuchten ``sim_id`` enthalten ist.
    """
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"test.pt nicht gefunden: {test_path}")

    # weights_only=False ist fuer PyG-Data-Objekte erforderlich, da diese
    # keine reinen Tensoren sind.
    dataset = torch.load(test_path, weights_only=False)

    matches = [g for g in dataset if getattr(g, "sim_id", None) == sim_id]
    if not matches:
        available = sorted(getattr(g, "sim_id", "?") for g in dataset)
        raise KeyError(
            f"'{sim_id}' nicht in {test_path} gefunden. "
            f"Enthaltene sim_ids: {available}"
        )
    return matches[0]


def compute_minmax(y: np.ndarray) -> dict[str, dict[str, float]]:
    """Berechne min/max je Feld aus dem Ziel-Tensor.

    Wichtig: Fuer ``|U|`` werden min/max der knotenweisen Magnitude gebildet,
    nicht die Magnitude der Komponenten-Extrema.

    Parameters
    ----------
    y : np.ndarray
        Ziel-Tensor der Form ``(N, 6)`` in der Spaltenordnung
        ``[Ux, Uy, Uz, p, k, epsilon]``.

    Returns
    -------
    dict
        ``{feldname: {"min": float, "max": float}}`` fuer ``|U|, p, k, epsilon``.
    """
    u = y[:, [COL_UX, COL_UY, COL_UZ]]
    u_mag = np.sqrt(np.sum(u ** 2, axis=1))  # knotenweise Magnitude (N,)

    columns = {
        "|U|": u_mag,
        "p": y[:, COL_P],
        "k": y[:, COL_K],
        "epsilon": y[:, COL_EPS],
    }
    return {
        name: {"min": float(np.min(vals)), "max": float(np.max(vals))}
        for name, vals in columns.items()
    }


def collect_results(paths: dict[str, str], sim_id: str):
    """Sammle min/max je Level und Feld.

    Returns
    -------
    dict
        ``{level: {feld: {"min": float, "max": float}}}`` in der Level-
        Reihenfolge von ``paths``.
    """
    results: dict[str, dict] = {}
    for level, path in paths.items():
        graph = find_graph(path, sim_id)
        y = graph.y.detach().cpu().numpy().astype(np.float64)
        if y.shape[1] != 6:
            raise ValueError(
                f"Level '{level}': unerwartete y-Form {tuple(y.shape)}, "
                f"erwartet (N, 6)."
            )
        results[level] = compute_minmax(y)
        print(f"  [{level:6s}] {path}  (N={y.shape[0]})")
    return results


def print_table(results: dict, sim_id: str) -> None:
    """Gib die Ergebnisse als formatierte Tabelle auf der Konsole aus."""
    levels = list(results.keys())
    print(f"\nMin/Max der Ground Truth fuer {sim_id}\n")

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


def write_markdown(results: dict, sim_id: str, md_path: str) -> None:
    """Schreibe eine lesbar formatierte Markdown-Tabelle (Feld x Level)."""
    levels = list(results.keys())
    with open(md_path, "w") as f:
        f.write(f"# Min/Max der subgesampleten Ground Truth — {sim_id}\n\n")
        f.write(
            "Vergleich der Wertebereiche der Zielfelder ueber die drei "
            "Subsampling-Level.\n\n"
        )

        head = ["Feld"] + [f"{lvl} min / max" for lvl in levels]
        f.write("| " + " | ".join(head) + " |\n")
        f.write("|" + "|".join(["---"] * len(head)) + "|\n")
        for field in FIELDS:
            # Pipe-Zeichen im Feldnamen (z. B. |U|) escapen, damit die
            # Markdown-Tabellenspalten nicht zerbrechen.
            cells = [field.replace("|", "\\|")]
            for lvl in levels:
                mn = results[lvl][field]["min"]
                mx = results[lvl][field]["max"]
                cells.append(f"{mn:.4g} / {mx:.4g}")
            f.write("| " + " | ".join(cells) + " |\n")
    print(f"MD   geschrieben: {md_path}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Min/Max der subgesampleten Ground Truth je Subsampling-Level."
    )
    parser.add_argument("--coarse", default=DEFAULT_PATHS["coarse"],
                        help="Pfad zur test.pt des Coarse-Levels.")
    parser.add_argument("--medium", default=DEFAULT_PATHS["medium"],
                        help="Pfad zur test.pt des Medium-Levels.")
    parser.add_argument("--bf25", default=DEFAULT_PATHS["bf_25"],
                        help="Pfad zur test.pt des bf_25-Levels.")
    parser.add_argument("--sim-id", default="sim_013",
                        help="Zu untersuchende Simulations-ID (Default: sim_013).")
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

    print(f"Lade Level-Datensaetze und filtere auf sim_id == '{args.sim_id}':")
    try:
        results = collect_results(paths, args.sim_id)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"\nFEHLER: {exc}", file=sys.stderr)
        return 1

    print_table(results, args.sim_id)

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, f"gt_minmax_{args.sim_id}.csv")
    md_path = os.path.join(args.outdir, f"gt_minmax_{args.sim_id}.md")
    write_csv(results, args.sim_id, csv_path)
    write_markdown(results, args.sim_id, md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
