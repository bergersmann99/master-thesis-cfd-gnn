"""
Solver-Driver fuer meshstudy_v2 — repliziert Schritte 7-11 aus
runSimulation.run_case ab vorhandenem Mesh in ~/laptop_timing/{level}/case/.

Schritte: 7) 0/-BCs mit building, 8) potentialFoam, 9) decomposePar,
10) simpleFoam (Phase 1 First-Order + Phase 2 Second-Order),
11) reconstructPar + Drag/Lift + Konvergenzcheck.

Keine VTK-Erzeugung (save_vtk: false in Config). Keine processor-Cleanup
(behalten fuer Dokumentation).
"""
import os
import sys
import time
import shutil
import argparse

REAL_DIR = "/home/tim-bergermann/laptop_timing/scripts"
sys.path.insert(0, REAL_DIR)

import yaml
import runSimulation as RS

BASE = "/home/tim-bergermann/laptop_timing"


def write_controlDict(case_dir, start_from, end_time, write_interval, purge):
    """Schreibt system/controlDict fuer den simpleFoam-Lauf."""
    RS._write_file(case_dir, "system/controlDict", f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     simpleFoam;

startFrom       {start_from};
startTime       0;

stopAt          endTime;
endTime         {end_time};

deltaT          1;

writeControl    timeStep;
writeInterval   {write_interval};
purgeWrite      {purge};

writeFormat     binary;
writePrecision  8;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable true;

libs            (atmosphericModels);

functions
{{
    forces
    {{
        type            forces;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        log             yes;
        patches         (building);
        rho             rhoInf;
        rhoInf          1.225;
        CofR            (0 0 0);
    }}
}}

// ************************************************************************* //
""")


def run(level, config_path, num_cores=4):
    """Fuehrt die Solver-Schritte 7-11 fuer eine Stufe aus und misst die Zeiten je Schritt."""
    print(f"\n{'='*70}\nSOLVER {level.upper()} — config: {config_path}\n{'='*70}", flush=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["hardware"]["num_cores"] = num_cores

    case_dir = os.path.join(BASE, level, "case")
    if not os.path.isdir(case_dir):
        raise FileNotFoundError(f"Case-Verzeichnis fehlt: {case_dir}")

    angle = cfg["wind"]["angle_min"]
    U_ref = cfg["wind"]["speed_min"]
    params = {"id": f"{level}_v2b", "U_ref": U_ref, "angle": angle}
    scfg = cfg["solver"]
    first_order_iters = scfg.get("first_order_iterations", 0)

    timings = {}
    base_path = os.getcwd()
    os.chdir(case_dir)
    os.environ["PWD"] = os.getcwd()

    try:
        # --- Schritt 7: 0/ mit building-Patch ---------------------------
        t0 = time.time()
        for d in os.listdir(case_dir):
            if d.startswith("processor"):
                shutil.rmtree(os.path.join(case_dir, d))
        RS._write_boundary_fields(case_dir, params, cfg, include_building=True)
        timings["7_bcs"] = time.time() - t0
        print(f"  [7] BCs aktualisiert: {timings['7_bcs']:.2f}s", flush=True)

        # --- Schritt 8: potentialFoam (seriell) -------------------------
        t0 = time.time()
        RS._run_of_command(["potentialFoam", "-initialiseUBCs"],
                           "log.potentialFoam")
        timings["8_potentialFoam"] = time.time() - t0
        print(f"  [8] potentialFoam: {timings['8_potentialFoam']:.2f}s", flush=True)

        # --- Schritt 9: decomposePar (mit Init-Feldern) -----------------
        t0 = time.time()
        RS._run_of_command(["decomposePar", "-force"], "log.decomposePar2")
        timings["9_decomposePar"] = time.time() - t0
        print(f"  [9] decomposePar: {timings['9_decomposePar']:.2f}s", flush=True)

        # --- Schritt 10a: simpleFoam Phase 1 (First-Order) --------------
        if first_order_iters and first_order_iters > 0:
            t0 = time.time()
            RS._write_fvSchemes(case_dir, second_order=False)
            write_controlDict(case_dir, "startTime", first_order_iters,
                              first_order_iters, scfg["purge_write"])
            RS._run_of_command(
                ["mpirun", "-np", str(num_cores),
                 "simpleFoam", "-parallel"],
                "log.simpleFoam",
                monitor_divergence=True,
            )
            timings["10a_simpleFoam_phase1"] = time.time() - t0
            print(f"  [10a] simpleFoam Phase 1 ({first_order_iters} iters): "
                  f"{timings['10a_simpleFoam_phase1']:.2f}s", flush=True)

            # --- Schritt 10b: simpleFoam Phase 2 (Second-Order) ----------
            t0 = time.time()
            RS._write_fvSchemes(case_dir, second_order=True)
            write_controlDict(case_dir, "latestTime", scfg["end_time"],
                              scfg["write_interval"], scfg["purge_write"])
            RS._run_of_command(
                ["mpirun", "-np", str(num_cores),
                 "simpleFoam", "-parallel"],
                "log.simpleFoam",
                monitor_divergence=True, append=True,
            )
            timings["10b_simpleFoam_phase2"] = time.time() - t0
            print(f"  [10b] simpleFoam Phase 2: {timings['10b_simpleFoam_phase2']:.2f}s",
                  flush=True)
        else:
            t0 = time.time()
            RS._run_of_command(
                ["mpirun", "-np", str(num_cores),
                 "simpleFoam", "-parallel"],
                "log.simpleFoam", monitor_divergence=True,
            )
            timings["10_simpleFoam"] = time.time() - t0

        # --- Schritt 11: reconstructPar + Konvergenz + Forces -----------
        t0 = time.time()
        RS._run_of_command(["reconstructPar", "-latestTime"], "log.reconstructPar")
        residuals = RS._parse_residuals("log.simpleFoam")
        status = RS._check_convergence(residuals, scfg["convergence"], "log.simpleFoam")
        drag, lift = RS._extract_forces(case_dir)
        timings["11_post"] = time.time() - t0
        print(f"  [11] reconstructPar + post: {timings['11_post']:.2f}s", flush=True)
    finally:
        os.chdir(base_path)
        os.environ["PWD"] = os.getcwd()

    total = sum(timings.values())
    result = {
        "level": level,
        "case_dir": case_dir,
        "status": status,
        "residuals": residuals,
        "drag": drag, "lift": lift,
        "timings": timings,
        "total_seconds": total,
    }

    print(f"\n--- {level.upper()} SOLVER result ---", flush=True)
    print(f"  status:    {status}", flush=True)
    print(f"  residuen:  {residuals}", flush=True)
    print(f"  drag:      {drag}", flush=True)
    print(f"  lift:      {lift}", flush=True)
    print(f"  total:     {total:.1f}s ({total/60:.1f} min)", flush=True)
    return result


def main():
    """Parst CLI-Argumente, startet den Solver-Lauf und schreibt das Ergebnis-YAML."""
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
    out = os.path.join(BASE, "logs", f"solver_{stem}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(result, f, sort_keys=False)
    print(f"\n[OK] Result yaml: {out}", flush=True)


if __name__ == "__main__":
    main()
