"""
meshStudy.py
============
Mesh-Unabhaengigkeitsstudie mit 3 Verfeinerungsstufen.

Fuehrt fuer jede Stufe (Coarse, Medium, Fine) eine Simulation mit
festen Windparametern (angle=0, U_ref=7.0 m/s) durch und berechnet
den Grid Convergence Index (GCI) nach Roache (1997).

Die Studie nutzt die bestehende Pipeline (createGeometry + runSimulation).

Verwendung:
    python meshStudy.py
    python meshStudy.py --config config.yaml

Referenzen:
    - Roache (1997): Quantification of uncertainty in CFD
    - Celik et al. (2008): Procedure for estimation of discretization error
    - AIJ-Guidelines Abschnitt 7 (Mesh-Qualitaet)

Siehe QUELLEN.md im Ausgabeordner fuer vollstaendige Quellenangaben.
"""

import os
import sys
import csv
import copy
import time
import re
import math
import argparse
from datetime import datetime

import yaml
import numpy as np

import test_createGeometry as createGeometry
import test_runSimulation as runSimulation


# ======================================================================
# Mesh-Stufen Definition
# ======================================================================

MESH_LEVELS = [
    {
        "name": "Coarse",
        "cells_per_H": 7,
        "refinement": {
            "surface_level": [4, 4],
            "feature_level": 4,
            "wake_level": 1,
            "near_level": 2,
            "close_level": 3,
        },
    },
    {
        "name": "Medium",
        "cells_per_H": 10,
        "refinement": {
            "surface_level": [5, 5],
            "feature_level": 5,
            "wake_level": 2,
            "near_level": 3,
            "close_level": 4,
        },
    },
    {
        "name": "Fine",
        "cells_per_H": 14,
        "refinement": {
            "surface_level": [5, 5],
            "feature_level": 5,
            "wake_level": 2,
            "near_level": 3,
            "close_level": 4,
        },
    },
]

# Standard-Windparameter fuer die Mesh-Studie
DEFAULT_U_REF = 7.0
DEFAULT_ANGLE = 0.0


# ======================================================================
# Log-Parsing
# ======================================================================

def parse_total_cells(log_path):
    """Parst die Gesamtzellzahl aus log.snappyHexMesh."""
    if not os.path.exists(log_path):
        return 0
    pattern = re.compile(r"Layer mesh\s*:\s*cells:(\d+)")
    fallback = re.compile(r"Snapped mesh\s*:\s*cells:(\d+)")
    total = 0
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                total = int(m.group(1))
            else:
                m = fallback.search(line)
                if m:
                    total = int(m.group(1))
    return total


# ======================================================================
# GCI-Berechnung (Richardson-Extrapolation)
# ======================================================================

def compute_gci(n_cells, values, safety_factor=1.25):
    """
    Berechnet den Grid Convergence Index nach Celik et al. (2008).

    Parameter
    ---------
    n_cells : array-like, shape (3,)
        Zellanzahl fuer [coarse, medium, fine].
    values : array-like, shape (3,)
        Zielgroesse fuer [coarse, medium, fine].
    safety_factor : float
        Sicherheitsfaktor (1.25 fuer 3+ Gitter, 3.0 fuer 2 Gitter).

    Rueckgabe
    ---------
    dict mit: p (Ordnung), phi_ext (extrapolierter Wert),
              gci_fine (GCI fein in %), gci_coarse (GCI grob in %),
              asymptotic_ratio.
    """
    n = np.array(n_cells, dtype=float)
    phi = np.array(values, dtype=float)

    # Repraesentative Gitterweite (3D: N^(1/3))
    h = n ** (-1.0 / 3.0)

    # Verfeinerungsverhaeltnisse
    r21 = h[0] / h[1]  # coarse/medium
    r32 = h[1] / h[2]  # medium/fine

    # Differenzen
    eps32 = phi[2] - phi[1]
    eps21 = phi[1] - phi[0]

    if abs(eps32) < 1e-15 or abs(eps21) < 1e-15:
        return {
            "p": float("nan"),
            "phi_ext": phi[2],
            "gci_fine_pct": 0.0,
            "gci_coarse_pct": 0.0,
            "asymptotic_ratio": float("nan"),
        }

    # Scheinbare Konvergenzordnung (iterativ nach Celik et al. 2008)
    s = np.sign(eps32 / eps21)
    if s < 0:
        # Oszillatorische Konvergenz
        p_est = float("nan")
        phi_ext = phi[2]
        gci_fine = float("nan")
        gci_coarse = float("nan")
        ar = float("nan")
    else:
        # Monotone Konvergenz: iterative Bestimmung von p
        p_est = abs(math.log(abs(eps21 / eps32))) / math.log(r21)

        # Extrapolierter Wert (Richardson)
        phi_ext = (r32 ** p_est * phi[2] - phi[1]) / (r32 ** p_est - 1.0)

        # Relative Fehler
        e_fine = abs((phi[2] - phi[1]) / phi[2]) if abs(phi[2]) > 1e-15 else 0.0
        e_coarse = abs((phi[1] - phi[0]) / phi[1]) if abs(phi[1]) > 1e-15 else 0.0

        # GCI
        gci_fine = safety_factor * e_fine / (r32 ** p_est - 1.0) * 100.0
        gci_coarse = safety_factor * e_coarse / (r21 ** p_est - 1.0) * 100.0

        # Asymptotic ratio (sollte ~1.0 sein)
        if gci_fine > 1e-15:
            ar = (gci_coarse / gci_fine) * (1.0 / r21 ** p_est)
        else:
            ar = float("nan")

    return {
        "p": p_est,
        "phi_ext": phi_ext,
        "gci_fine_pct": gci_fine,
        "gci_coarse_pct": gci_coarse,
        "asymptotic_ratio": ar,
    }


# ======================================================================
# Report-Schreiber
# ======================================================================

def write_report(output_dir, rows, gci_drag, gci_lift, u_ref=7.0, angle=0.0):
    """Schreibt den Mesh-Studie-Report als Textdatei."""
    path = os.path.join(output_dir, "mesh_study_report.txt")
    with open(path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write("  MESH-UNABHAENGIGKEITSSTUDIE\n")
        f.write(f"  Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Windparameter: U_ref={u_ref} m/s, angle={angle} Grad\n")
        f.write("=" * 72 + "\n\n")

        # Ergebnistabelle
        f.write("  ERGEBNISSE\n")
        f.write("  " + "-" * 68 + "\n")
        header = (
            f"  {'Stufe':<10s} {'cells_H':>7s} {'Zellen':>12s} "
            f"{'Drag [N]':>12s} {'Lift [N]':>12s} {'Status':<14s} {'Zeit [s]':>8s}\n"
        )
        f.write(header)
        f.write("  " + "-" * 68 + "\n")
        for r in rows:
            f.write(
                f"  {r['name']:<10s} {r['cells_per_H']:>7d} {r['total_cells']:>12,d} "
                f"{r['drag']:>12.2f} {r['lift']:>12.2f} {r['status']:<14s} {r['runtime']:>8.0f}\n"
            )
        f.write("  " + "-" * 68 + "\n\n")

        # Residuen
        f.write("  FINALE RESIDUEN\n")
        f.write("  " + "-" * 68 + "\n")
        f.write(f"  {'Stufe':<10s} {'p':>10s} {'Ux':>10s} {'Uy':>10s} "
                f"{'Uz':>10s} {'k':>10s} {'eps':>10s}\n")
        f.write("  " + "-" * 68 + "\n")
        for r in rows:
            res = r["residuals"]
            f.write(
                f"  {r['name']:<10s} "
                f"{res.get('p', float('nan')):>10.2e} "
                f"{res.get('Ux', float('nan')):>10.2e} "
                f"{res.get('Uy', float('nan')):>10.2e} "
                f"{res.get('Uz', float('nan')):>10.2e} "
                f"{res.get('k', float('nan')):>10.2e} "
                f"{res.get('epsilon', float('nan')):>10.2e}\n"
            )
        f.write("  " + "-" * 68 + "\n\n")

        # GCI
        f.write("  GRID CONVERGENCE INDEX (GCI)\n")
        f.write("  Methode: Richardson-Extrapolation nach Celik et al. (2008)\n")
        f.write("  Sicherheitsfaktor: Fs = 1.25\n")
        f.write("  " + "-" * 68 + "\n")

        for label, gci in [("Drag", gci_drag), ("Lift", gci_lift)]:
            f.write(f"\n  {label}:\n")
            if gci is None:
                f.write("    Berechnung nicht moeglich (fehlende Daten)\n")
            elif math.isnan(gci["p"]):
                f.write("    Oszillatorische Konvergenz erkannt.\n")
                f.write("    GCI nicht berechenbar (nicht-monoton).\n")
                f.write(f"    Werte: {[r[label.lower()] for r in rows]}\n")
            else:
                f.write(f"    Scheinbare Ordnung p     = {gci['p']:.2f}\n")
                f.write(f"    Extrapolierter Wert      = {gci['phi_ext']:.4f}\n")
                f.write(f"    GCI (fein)               = {gci['gci_fine_pct']:.3f} %\n")
                f.write(f"    GCI (grob)               = {gci['gci_coarse_pct']:.3f} %\n")
                f.write(f"    Asymptotic Ratio         = {gci['asymptotic_ratio']:.4f}\n")

        f.write("\n  " + "-" * 68 + "\n\n")

        # Empfehlung
        f.write("  EMPFEHLUNG\n")
        f.write("  " + "-" * 68 + "\n")
        all_converged = all(r["status"] == "Converged" for r in rows)
        if all_converged:
            if gci_drag and not math.isnan(gci_drag.get("gci_fine_pct", float("nan"))):
                if gci_drag["gci_fine_pct"] < 5.0:
                    f.write("  Das Medium-Mesh ist ausreichend (GCI_fine < 5%).\n")
                    f.write("  Es kann fuer die Produktionslaeufe verwendet werden.\n")
                else:
                    f.write("  ACHTUNG: GCI_fine > 5% — das Fine-Mesh sollte\n")
                    f.write("  als Produktions-Mesh verwendet werden.\n")
            else:
                f.write("  GCI nicht berechenbar. Visuelle Pruefung der Werte empfohlen.\n")
        else:
            crashed = [r["name"] for r in rows if r["status"] not in ("Converged", "NotConverged")]
            if crashed:
                f.write(f"  WARNUNG: Stufen {crashed} sind gecrasht.\n")
                f.write("  Mesh-Studie muss wiederholt werden.\n")
            else:
                f.write("  Nicht alle Stufen konvergiert. Ergebnisse mit Vorsicht verwenden.\n")

        f.write("\n" + "=" * 72 + "\n")
        f.write("  FERTIG\n")
        f.write("=" * 72 + "\n")

    return path


# ======================================================================
# Hauptprogramm
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mesh-Unabhaengigkeitsstudie (3 Stufen + GCI)"
    )
    parser.add_argument(
        "--config", default="test_config.yaml",
        help="Pfad zur Konfigurationsdatei (Standard: config.yaml)"
    )
    parser.add_argument(
        "--angle", type=float, default=None,
        help="Windwinkel in Grad (Standard: 0.0)"
    )
    parser.add_argument(
        "--u-ref", type=float, default=None,
        help="Referenzgeschwindigkeit in m/s (Standard: 7.0)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Ausgabeordner (Standard: 'Mesh study (4, 4)')"
    )
    args = parser.parse_args()

    # Konfiguration laden
    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)

    geo = base_cfg["geometry"]
    H = geo["wall_height"] + geo["roof_height"]
    base_path = os.getcwd()

    # Windparameter
    FIXED_U_REF = args.u_ref if args.u_ref is not None else DEFAULT_U_REF
    FIXED_ANGLE = args.angle if args.angle is not None else DEFAULT_ANGLE

    # Ausgabeordner
    if args.output_dir:
        output_dir = os.path.join(base_path, args.output_dir)
    else:
        output_dir = os.path.join(base_path, "Mesh study (4, 4)")
    os.makedirs(output_dir, exist_ok=True)

    # STL erzeugen (einmalig)
    stl_path = os.path.join(base_path, "geometry", "mesh_study.stl")
    bounds = createGeometry.create_building(
        geo["width"], geo["depth"],
        geo["wall_height"], geo["roof_height"],
        FIXED_ANGLE, stl_path,
    )

    params = {"U_ref": FIXED_U_REF, "angle": FIXED_ANGLE}

    print(f"\n{'=' * 60}")
    print(f"   MESH-UNABHAENGIGKEITSSTUDIE")
    print(f"{'=' * 60}")
    print(f"   U_ref = {FIXED_U_REF} m/s, angle = {FIXED_ANGLE} Grad")
    print(f"   Gebaeude: {geo['width']} x {geo['depth']} x {H} m")
    print(f"   Stufen: {len(MESH_LEVELS)}")
    print(f"{'=' * 60}\n")

    rows = []

    for i, level in enumerate(MESH_LEVELS):
        print(f"\n--- [{i+1}/{len(MESH_LEVELS)}] {level['name']} "
              f"(cells_per_H={level['cells_per_H']}) ---")

        # Config deepcopy und Mesh-Parameter ueberschreiben
        cfg = copy.deepcopy(base_cfg)
        cfg["mesh"]["cells_per_H"] = level["cells_per_H"]
        cfg["mesh"]["refinement"] = level["refinement"]
        cfg["general"]["save_vtk"] = False  # kein VTK bei Mesh-Studie
        cfg["general"]["simulation_dir"] = "simulation"

        # Angepasste max_global_cells fuer Fine-Mesh
        if level["cells_per_H"] >= 14:
            cfg["mesh"]["max_global_cells"] = 20000000
            cfg["mesh"]["max_local_cells"] = 5000000

        case_name = f"mesh_{level['name'].lower()}"

        def status_cb(name, step, total, desc, _i=i, _n=len(MESH_LEVELS)):
            msg = f"\r  [{_i+1}/{_n}] [{name}] Schritt {step}/{total}: {desc}"
            sys.stdout.write(f"{msg:<80}")
            sys.stdout.flush()

        sim_start = time.time()

        result = runSimulation.run_case(
            case_name=case_name,
            stl_source=stl_path,
            params=params,
            bounds=bounds,
            cfg=cfg,
            status_callback=status_cb,
        )

        runtime = time.time() - sim_start

        # Zellzahl aus Log parsen
        log_snappy = os.path.join(
            base_path, "simulation", case_name, "log.snappyHexMesh")
        total_cells = parse_total_cells(log_snappy)

        row = {
            "name": level["name"],
            "cells_per_H": level["cells_per_H"],
            "total_cells": total_cells,
            "drag": result.get("drag", float("nan")),
            "lift": result.get("lift", float("nan")),
            "residuals": result.get("residuals", {}),
            "status": result.get("status", "Unknown"),
            "runtime": runtime,
        }
        rows.append(row)

        status_icon = "OK" if result["status"] == "Converged" else (
            "!!" if result["status"] == "NotConverged" else "XX")
        print(f"\r  [{status_icon}] {level['name']}: {result['status']} "
              f"({runtime:.0f}s, {total_cells:,} Zellen, "
              f"Drag={result.get('drag', float('nan')):.1f} N)"
              f"{'':<20}")

    # ==================================================================
    # CSV speichern
    # ==================================================================
    csv_path = os.path.join(output_dir, "mesh_study_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Stufe", "cells_per_H", "Gesamtzellen",
            "Drag", "Lift",
            "Res_p", "Res_Ux", "Res_Uy", "Res_Uz", "Res_k", "Res_epsilon",
            "Status", "Laufzeit_s",
        ])
        for r in rows:
            res = r["residuals"]
            writer.writerow([
                r["name"], r["cells_per_H"], r["total_cells"],
                r["drag"], r["lift"],
                res.get("p", "NaN"), res.get("Ux", "NaN"),
                res.get("Uy", "NaN"), res.get("Uz", "NaN"),
                res.get("k", "NaN"), res.get("epsilon", "NaN"),
                r["status"], f"{r['runtime']:.0f}",
            ])
    print(f"\n   CSV: {csv_path}")

    # ==================================================================
    # GCI berechnen
    # ==================================================================
    gci_drag = None
    gci_lift = None

    valid = [r for r in rows if r["status"] in ("Converged", "NotConverged")
             and not math.isnan(r["drag"])]
    if len(valid) == 3:
        n_cells = [r["total_cells"] for r in valid]
        drags = [r["drag"] for r in valid]
        lifts = [r["lift"] for r in valid]

        gci_drag = compute_gci(n_cells, drags)
        gci_lift = compute_gci(n_cells, lifts)

        print(f"\n   GCI Drag:  p={gci_drag['p']:.2f}, "
              f"GCI_fine={gci_drag['gci_fine_pct']:.3f}%")
        print(f"   GCI Lift:  p={gci_lift['p']:.2f}, "
              f"GCI_fine={gci_lift['gci_fine_pct']:.3f}%")
    else:
        print(f"\n   GCI: Nur {len(valid)}/3 gueltige Ergebnisse, "
              f"GCI nicht berechenbar.")

    # ==================================================================
    # Report schreiben
    # ==================================================================
    report_path = write_report(output_dir, rows, gci_drag, gci_lift,
                               u_ref=FIXED_U_REF, angle=FIXED_ANGLE)
    print(f"   Report: {report_path}")

    # ==================================================================
    # QUELLEN.md
    # ==================================================================
    quellen_path = os.path.join(output_dir, "QUELLEN.md")
    with open(quellen_path, "w") as f:
        f.write("# Quellen: Mesh-Unabhaengigkeitsstudie\n\n")
        f.write("## Richardson-Extrapolation & GCI\n\n")
        f.write("- Roache, P.J. (1997): *Quantification of Uncertainty in "
                "Computational Fluid Dynamics*. Annual Review of Fluid "
                "Mechanics, 29, 123-160.\n\n")
        f.write("- Celik, I.B., Ghia, U., Roache, P.J., Freitas, C.J., "
                "Coleman, H., Raad, P.E. (2008): *Procedure for Estimation "
                "and Reporting of Uncertainty Due to Discretization in CFD "
                "Applications*. Journal of Fluids Engineering, 130(7), "
                "078001.\n\n")
        f.write("## Mesh-Qualitaet\n\n")
        f.write("- Tominaga, Y. et al. (2008): *AIJ guidelines for practical "
                "applications of CFD to pedestrian wind environment around "
                "buildings*. Journal of Wind Engineering and Industrial "
                "Aerodynamics, 96(10-11), 1749-1761.\n\n")
        f.write("## OpenFOAM\n\n")
        f.write("- OpenFOAM v2506 User Guide: snappyHexMesh, simpleFoam\n")
        f.write("- OpenFOAM Wiki: Mesh quality metrics\n")
    print(f"   Quellen: {quellen_path}")

    print(f"\n{'=' * 60}")
    print(f"   MESH-STUDIE ABGESCHLOSSEN")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
