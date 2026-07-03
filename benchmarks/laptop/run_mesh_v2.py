"""
Mesh-Driver fuer meshstudy_v2 — repliziert Mesh-Schritte 1-6 aus
runSimulation.run_case + checkMesh, persistent unter ~/laptop_timing/{level}/case/.
"""
import os
import sys
import time
import shutil
import argparse
import re

REAL_DIR = "/home/tim-bergermann/laptop_timing/scripts"
sys.path.insert(0, REAL_DIR)

import yaml
import createGeometry
import runSimulation as RS

BASE = "/home/tim-bergermann/laptop_timing"


def parse_snappy_stats(case_dir):
    """Liefert finale cells/points/faces und Cap-Trefferzahl aus snappy-Log."""
    log_path = os.path.join(case_dir, "log.snappyHexMesh")
    cells = points = faces = None
    cap_hits = 0
    with open(log_path) as f:
        for line in f:
            m = re.search(r"Layer mesh\s*:\s*cells:(\d+)\s+faces:(\d+)\s+points:(\d+)", line)
            if m:
                cells, faces, points = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if "reached limit" in line:
                cap_hits += 1
    return cells, points, faces, cap_hits


def parse_checkmesh(case_dir):
    """Bestaetigt checkMesh OK + Anzahl Zellen/Punkte/Flaechen aus dem checkMesh-Log."""
    log_path = os.path.join(case_dir, "log.checkMesh")
    ok = False
    cells = points = faces = None
    with open(log_path) as f:
        for line in f:
            if line.strip().startswith("Mesh OK"):
                ok = True
            m = re.match(r"\s*(cells|points|faces):\s+(\d+)", line)
            if m:
                key, val = m.group(1), int(m.group(2))
                if key == "cells": cells = val
                elif key == "points": points = val
                elif key == "faces": faces = val
    return ok, cells, points, faces


def run(level, config_path, num_cores=4):
    print(f"\n{'='*70}\n{level.upper()} — config: {config_path}\n{'='*70}", flush=True)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["hardware"]["num_cores"] = num_cores

    geo = cfg["geometry"]
    angle = cfg["wind"]["angle_min"]
    U_ref = cfg["wind"]["speed_min"]

    stl_path = os.path.join(BASE, "geometry", f"building_a{angle:.2f}.stl")
    os.makedirs(os.path.dirname(stl_path), exist_ok=True)
    bounds = createGeometry.create_building(
        geo["width"], geo["depth"], geo["wall_height"], geo["roof_height"],
        angle, stl_path,
    )
    print(f"STL: {stl_path}", flush=True)

    case_dir = os.path.join(BASE, level, "case")
    params = {"id": f"{level}_v2", "U_ref": U_ref, "angle": angle}

    # --- Setup ----------------------------------------------------------
    t_total = time.time()
    t0 = time.time()
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)
    for d in ["constant/triSurface", "system", "0"]:
        os.makedirs(os.path.join(case_dir, d), exist_ok=True)
    shutil.copy(stl_path, os.path.join(case_dir, "constant/triSurface/building.stl"))
    RS._write_system_dictionaries(case_dir, bounds, params, cfg)
    RS._write_constant_dictionaries(case_dir, cfg)
    RS._write_boundary_fields(case_dir, params, cfg, include_building=False)
    t_setup = time.time() - t0

    base_path = os.getcwd()
    os.chdir(case_dir)
    os.environ["PWD"] = os.getcwd()
    timings = {"setup": t_setup}
    try:
        for step_name, cmd in [
            ("blockMesh",            ["blockMesh"]),
            ("surfaceFeatureExtract", ["surfaceFeatureExtract"]),
            ("decomposePar",         ["decomposePar", "-force"]),
            ("snappyHexMesh",        ["mpirun", "-np", str(num_cores),
                                      "snappyHexMesh", "-overwrite", "-parallel"]),
            ("reconstructParMesh",   ["reconstructParMesh", "-constant", "-mergeTol", "1e-6"]),
            ("checkMesh",            ["checkMesh", "-constant"]),
        ]:
            t0 = time.time()
            log = f"log.{step_name}"
            RS._run_of_command(cmd, log)
            timings[step_name] = time.time() - t0
            print(f"  {step_name}: {timings[step_name]:.2f}s", flush=True)
    finally:
        os.chdir(base_path)
        os.environ["PWD"] = os.getcwd()

    t_run = time.time() - t_total
    cells, points, faces, cap_hits = parse_snappy_stats(case_dir)
    ok, _, _, _ = parse_checkmesh(case_dir)

    print(f"\n--- {level.upper()} result ---", flush=True)
    print(f"  cells:        {cells:,}", flush=True)
    print(f"  points:       {points:,}", flush=True)
    print(f"  faces:        {faces:,}", flush=True)
    print(f"  cap hits:     {cap_hits}", flush=True)
    print(f"  checkMesh OK: {ok}", flush=True)
    print(f"  total run:    {t_run:.1f}s", flush=True)

    return {
        "level": level, "cells_per_H": cfg["mesh"]["cells_per_H"],
        "cells": cells, "points": points, "faces": faces,
        "cap_hits": cap_hits, "check_ok": ok,
        "timings": timings, "total_seconds": t_run, "case_dir": case_dir,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, choices=["coarse", "medium", "fine"])
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--config", type=str, default=None,
                    help="Direkter Config-Pfad (ueberschreibt --level-basierten Default)")
    args = ap.parse_args()

    if args.config:
        config_path = args.config
        stem = os.path.splitext(os.path.basename(config_path))[0].replace("config_", "")
    else:
        config_path = os.path.join(BASE, "scripts", "configs",
                                   f"config_{args.level}_v2.yaml")
        stem = args.level

    result = run(args.level, config_path, num_cores=args.cores)

    os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)
    out = os.path.join(BASE, "logs", f"mesh_{stem}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(result, f, sort_keys=False)
    print(f"\n[OK] Result yaml: {out}", flush=True)


if __name__ == "__main__":
    main()
