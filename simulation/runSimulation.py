"""
runSimulation.py
================
Erzeugt einen vollstaendigen OpenFOAM-Case (simpleFoam, kEpsilon),
fuehrt die Simulation durch und prueft die Konvergenz.

Reihenfolge (paralleles snappyHexMesh mit Patch-Handling):
    1.  Case-Setup (Dictionaries OHNE building-Patch in 0/)
    2.  blockMesh (seriell)
    3.  surfaceFeatureExtract (seriell)
    4.  decomposePar (seriell, zerlegt Basis-Mesh)
    5.  snappyHexMesh -parallel (erzeugt building-Patch)
    6.  reconstructParMesh (seriell, fuehrt Mesh zusammen)
    7.  0/-Dateien MIT building-Patch neu schreiben
    8.  potentialFoam -initialiseUBCs (seriell, Initialisierung)
    9.  decomposePar (mit initialisierten Feldern)
    10. simpleFoam -parallel
    11. reconstructPar + Post-Processing

Hintergrund: decomposePar im Schritt 4 kennt den building-Patch noch
nicht (existiert erst nach snappyHexMesh). Deshalb werden die 0/-Dateien
in zwei Phasen geschrieben: erst ohne, dann mit building-BCs.

ABL-Wandfunktionen:
    Verwendet atmNutkWallFunction und atmEpsilonWallFunction (aus
    libatmosphericModels) am ground-Patch. Diese verwenden z0 direkt
    und sind unabhaengig von der lokalen Zellhoehe (kein yp > Ks Problem).

Referenzen:
    - AIJ-Guidelines: cfd-spec-aij.md
    - OpenFOAM v2506 User Guide
    - Blocken et al. (2007): CFD simulation of the ABL
    - Richards & Hoxey (1993): Appropriate BCs for ABL simulation
    - Hargreaves & Wright (2007): On the use of k-epsilon in CFD
    - Nikuradse (1933): Ks = 20*z0 Beziehung

"""

import os
import re
import math
import shutil
import subprocess
import sys
import traceback


# ======================================================================
# Oeffentliche API
# ======================================================================

def run_case(case_name, stl_source, params, bounds, cfg, status_callback=None):
    """
    Fuehrt eine vollstaendige OpenFOAM-Simulation durch.

    Parameter
    ---------
    case_name : str
        Eindeutiger Name des Falls (z.B. "sim_003").
    stl_source : str
        Pfad zur STL-Datei des Gebaeudes.
    params : dict
        Simulationsparameter mit Schluesseln: U_ref, angle.
    bounds : tuple
        Bounding Box (min_x, max_x, min_y, max_y, min_z, max_z).
    cfg : dict
        Gesamte Konfiguration (aus config.yaml).
    status_callback : callable, optional
        Funktion(case_name, step, total_steps, description) fuer Statusanzeige.

    Rueckgabe
    ---------
    dict : Ergebnis mit Status, Residuen, Kraeften, Fehlerdetails.
    """
    base_path = os.getcwd()
    sim_dir = cfg["general"]["simulation_dir"]
    res_dir = cfg["general"]["results_dir"]
    case_dir = os.path.join(base_path, sim_dir, case_name)
    vtk_dir = os.path.join(base_path, res_dir, "vtks")
    num_cores = cfg["hardware"]["num_cores"]

    total_steps = 11
    result = {
        "id": case_name,
        "status": "Unknown",
        "residuals": {},
        "drag": float("nan"),
        "lift": float("nan"),
        "failed_step": None,
        "error_message": None,
        "error_log_tail": None,
        "error_logfile": None,
        "error_traceback": None,
    }

    step_info = {
        1:  ("Case-Setup",                   None),
        2:  ("blockMesh",                     "log.blockMesh"),
        3:  ("surfaceFeatureExtract",         "log.surfaceFeatureExtract"),
        4:  ("decomposePar (Basis-Mesh)",     "log.decomposePar"),
        5:  ("snappyHexMesh (parallel)",      "log.snappyHexMesh"),
        6:  ("reconstructParMesh",            "log.reconstructParMesh"),
        7:  ("BCs aktualisieren",             None),
        8:  ("potentialFoam (Initialisierung)", "log.potentialFoam"),
        9:  ("decomposePar (initialisiert)",  "log.decomposePar2"),
        10: ("simpleFoam (parallel)",          "log.simpleFoam"),
        11: ("Post-Processing",                None),
    }

    current_step = 0

    def _status(step, desc):
        nonlocal current_step
        current_step = step
        if status_callback:
            status_callback(case_name, step, total_steps, desc)

    def _fail(step, exception, logfile=None):
        step_name = step_info.get(step, ("Unbekannt", None))[0]
        result["status"] = "Crashed"
        result["failed_step"] = f"Schritt {step}/{total_steps}: {step_name}"
        result["error_message"] = str(exception)
        result["error_traceback"] = traceback.format_exc()
        if logfile is None:
            logfile = step_info.get(step, (None, None))[1]
        if logfile and os.path.exists(logfile):
            result["error_log_tail"] = _get_log_tail(logfile, n_lines=20)
            result["error_logfile"] = os.path.join(case_dir, logfile)
        elif logfile:
            result["error_logfile"] = f"{logfile} (nicht vorhanden)"

    try:
        # ==============================================================
        # Schritt 1/11: Case-Setup (OHNE building-Patch in 0/)
        # ==============================================================
        _status(1, "Case-Setup")
        try:
            if os.path.exists(case_dir):
                shutil.rmtree(case_dir)
            for d in ["constant/triSurface", "system", "0"]:
                os.makedirs(os.path.join(case_dir, d), exist_ok=True)
            if cfg["general"]["save_vtk"]:
                os.makedirs(vtk_dir, exist_ok=True)
            shutil.copy(stl_source, os.path.join(
                case_dir, "constant/triSurface/building.stl"))
            _write_system_dictionaries(case_dir, bounds, params, cfg)
            _write_constant_dictionaries(case_dir, cfg)
            _write_boundary_fields(case_dir, params, cfg, include_building=False)
        except Exception as e:
            _fail(1, e)
            return result

        os.chdir(case_dir)
        os.environ["PWD"] = os.getcwd()

        # ==============================================================
        # Schritt 2/11: blockMesh (seriell)
        # ==============================================================
        _status(2, "blockMesh")
        try:
            _run_of_command(["blockMesh"], "log.blockMesh")
        except Exception as e:
            _fail(2, e)
            os.chdir(base_path)
            os.environ["PWD"] = os.getcwd()
            return result

        # ==============================================================
        # Schritt 3/11: surfaceFeatureExtract (seriell)
        # ==============================================================
        _status(3, "surfaceFeatureExtract")
        try:
            _run_of_command(["surfaceFeatureExtract"], "log.surfaceFeatureExtract")
        except Exception as e:
            _fail(3, e)
            os.chdir(base_path)
            os.environ["PWD"] = os.getcwd()
            return result

        # ==============================================================
        # Schritt 4/11: decomposePar (Basis-Mesh, ohne building-Patch)
        # ==============================================================
        _status(4, "decomposePar (Basis-Mesh)")
        try:
            _run_of_command(["decomposePar", "-force"], "log.decomposePar")
        except Exception as e:
            _fail(4, e)
            os.chdir(base_path)
            os.environ["PWD"] = os.getcwd()
            return result

        # ==============================================================
        # Schritt 5/11: snappyHexMesh -parallel (erzeugt building-Patch)
        # ==============================================================
        _status(5, "snappyHexMesh (parallel)")
        try:
            _run_of_command(
                ["mpirun", "-np", str(num_cores),
                 "snappyHexMesh", "-overwrite", "-parallel"],
                "log.snappyHexMesh"
            )
        except Exception as e:
            _fail(5, e)
            os.chdir(base_path)
            os.environ["PWD"] = os.getcwd()
            return result

        # ==============================================================
        # Schritt 6/11: reconstructParMesh (Mesh zusammenfuehren)
        # ==============================================================
        _status(6, "reconstructParMesh")
        try:
            _run_of_command(
                ["reconstructParMesh", "-constant", "-mergeTol", "1e-6"],
                "log.reconstructParMesh"
            )
        except Exception as e:
            _fail(6, e)
            os.chdir(base_path)
            os.environ["PWD"] = os.getcwd()
            return result

        # ==============================================================
        # Schritt 7/11: 0/-Dateien MIT building-Patch neu schreiben
        # ==============================================================
        _status(7, "BCs aktualisieren")
        try:
            for d in os.listdir(case_dir):
                if d.startswith("processor"):
                    shutil.rmtree(os.path.join(case_dir, d))
            _write_boundary_fields(case_dir, params, cfg, include_building=True)
        except Exception as e:
            _fail(7, e)
            os.chdir(base_path)
            os.environ["PWD"] = os.getcwd()
            return result

        # ==============================================================
        # Schritt 8/11: potentialFoam (seriell, Initialisierung)
        # ==============================================================
        _status(8, "potentialFoam (Initialisierung)")
        try:
            _run_of_command(
                ["potentialFoam", "-initialiseUBCs"],
                "log.potentialFoam"
            )
        except Exception as e:
            _fail(8, e, "log.potentialFoam")
            os.chdir(base_path)
            os.environ["PWD"] = os.getcwd()
            return result

        # ==============================================================
        # Schritt 9/11: decomposePar (mit initialisierten Feldern)
        # ==============================================================
        _status(9, "decomposePar (initialisiert)")
        try:
            _run_of_command(["decomposePar", "-force"], "log.decomposePar2")
        except Exception as e:
            _fail(9, e)
            os.chdir(base_path)
            os.environ["PWD"] = os.getcwd()
            return result

        # ==============================================================
        # Schritt 10/11: simpleFoam -parallel (Zwei-Phasen-Strategie)
        # ==============================================================
        first_order_iters = cfg["solver"].get("first_order_iterations", 0)

        # --- Phase 1: First-Order Upwind (Stabilisierung) ---
        if first_order_iters and first_order_iters > 0:
            _status(10, "simpleFoam Phase 1 (Upwind)")
            try:
                # fvSchemes auf First-Order umschreiben
                _write_fvSchemes(case_dir, second_order=False)

                # controlDict: endTime und writeInterval auf first_order_iters
                _write_file(case_dir, "system/controlDict", f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     simpleFoam;

startFrom       startTime;
startTime       0;

stopAt          endTime;
endTime         {first_order_iters};

deltaT          1;

writeControl    timeStep;
writeInterval   {first_order_iters};
purgeWrite      {cfg['solver']['purge_write']};

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

                _run_of_command(
                    ["mpirun", "-np", str(num_cores),
                     "simpleFoam", "-parallel"],
                    "log.simpleFoam",
                    monitor_divergence=True
                )
            except Exception as e:
                _fail(10, e, "log.simpleFoam")
                os.chdir(base_path)
                os.environ["PWD"] = os.getcwd()
                return result

            # --- Phase 2: Second-Order (Produktionslauf) ---
            _status(10, "simpleFoam Phase 2 (Second Order)")
            try:
                # fvSchemes auf Second-Order zurueckschreiben
                _write_fvSchemes(case_dir, second_order=True)

                # controlDict: startFrom latestTime, originale endTime/writeInterval
                _write_file(case_dir, "system/controlDict", f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     simpleFoam;

startFrom       latestTime;

stopAt          endTime;
endTime         {cfg['solver']['end_time']};

deltaT          1;

writeControl    timeStep;
writeInterval   {cfg['solver']['write_interval']};
purgeWrite      {cfg['solver']['purge_write']};

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

                _run_of_command(
                    ["mpirun", "-np", str(num_cores),
                     "simpleFoam", "-parallel"],
                    "log.simpleFoam",
                    monitor_divergence=True,
                    append=True
                )
            except Exception as e:
                _fail(10, e, "log.simpleFoam")
                os.chdir(base_path)
                os.environ["PWD"] = os.getcwd()
                return result

        else:
            # Keine Zwei-Phasen-Strategie: direkt mit Second-Order
            _status(10, "simpleFoam (parallel)")
            try:
                _run_of_command(
                    ["mpirun", "-np", str(num_cores),
                     "simpleFoam", "-parallel"],
                    "log.simpleFoam",
                    monitor_divergence=True
                )
            except Exception as e:
                _fail(10, e, "log.simpleFoam")
                os.chdir(base_path)
                os.environ["PWD"] = os.getcwd()
                return result

        # ==============================================================
        # Schritt 11/11: reconstructPar + Post-Processing
        # ==============================================================
        _status(11, "Post-Processing")
        try:
            _run_of_command(
                ["reconstructPar", "-latestTime"], "log.reconstructPar")

            result["residuals"] = _parse_residuals("log.simpleFoam")
            result["status"] = _check_convergence(
                result["residuals"], cfg["solver"]["convergence"],
                "log.simpleFoam"
            )

            drag, lift = _extract_forces(case_dir)
            result["drag"] = drag
            result["lift"] = lift

            if cfg["general"]["save_vtk"] and result["status"] in ("Converged", "NotConverged"):
                _run_of_command(
                    ["foamToVTK", "-latestTime"], "log.foamToVTK")
                vtk_src = os.path.join(case_dir, "VTK")
                if os.path.exists(vtk_src):
                    vtk_dest = os.path.join(vtk_dir, f"{case_name}_VTK")
                    if os.path.exists(vtk_dest):
                        shutil.rmtree(vtk_dest)
                    shutil.copytree(vtk_src, vtk_dest)

            if cfg["cleanup"]["delete_processor_dirs"]:
                for d in os.listdir(case_dir):
                    if d.startswith("processor"):
                        shutil.rmtree(os.path.join(case_dir, d))

            if cfg["cleanup"]["delete_intermediate_times"]:
                _cleanup_time_dirs(case_dir, cfg["solver"]["end_time"])

        except Exception as e:
            _fail(11, e, "log.reconstructPar")
            os.chdir(base_path)
            os.environ["PWD"] = os.getcwd()
            return result

    except Exception as e:
        result["status"] = "Error"
        result["failed_step"] = (
            f"Unerwarteter Fehler (letzter Schritt: {current_step})")
        result["error_message"] = str(e)
        result["error_traceback"] = traceback.format_exc()
    finally:
        os.chdir(base_path)
        os.environ["PWD"] = os.getcwd()

    return result


# ======================================================================
# OpenFOAM-Kommandos
# ======================================================================

def _run_of_command(cmd, logfile, monitor_divergence=False, append=False):
    """Fuehrt ein OpenFOAM-Kommando aus und schreibt stdout/stderr in logfile.

    Bei monitor_divergence=True wird der Prozess abgebrochen, wenn
    die Residuen explodieren (Initial residual > 1e10).
    Bei append=True wird an eine bestehende Logdatei angehaengt.
    """
    mode = "a" if append else "w"
    with open(logfile, mode) as log:
        if not monitor_divergence:
            subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            return

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        divergence_pattern = re.compile(
            r"Initial residual = ([\d.eE\+\-]+)")
        try:
            for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace")
                log.write(line)
                m = divergence_pattern.search(line)
                if m:
                    try:
                        val = float(m.group(1))
                        if val > 1e10 or math.isnan(val) or math.isinf(val):
                            proc.terminate()
                            proc.wait(timeout=30)
                            raise RuntimeError(
                                f"Divergenz erkannt: Residuum = {val:.2e} "
                                f"(Abbruch, um SIGFPE zu vermeiden)")
                    except ValueError:
                        pass
            proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
        except Exception:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)
            raise


# ======================================================================
# Log-Analyse
# ======================================================================

def _get_log_tail(logfile, n_lines=20):
    """Liest die letzten n_lines Zeilen einer Log-Datei."""
    if not os.path.exists(logfile):
        return f"[Datei nicht vorhanden: {logfile}]"
    try:
        with open(logfile, "r") as f:
            lines = f.readlines()
        tail = lines[-n_lines:] if len(lines) >= n_lines else lines
        return "".join(tail)
    except Exception as e:
        return f"[Fehler beim Lesen: {e}]"


def _parse_residuals(logfile):
    """Parst die letzten Residuen aus log.simpleFoam."""
    residuals = {}
    pattern = re.compile(
        r"Solving for (\w+),\s+Initial residual = ([\d.eE\+\-]+)")
    if not os.path.exists(logfile):
        return residuals
    with open(logfile, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                residuals[match.group(1)] = float(match.group(2))
    return residuals


def _check_convergence(residuals, thresholds, logfile):
    """Prueft Konvergenz anhand der AIJ-Kriterien."""
    if not os.path.exists(logfile):
        return "Crashed"
    with open(logfile, "r") as f:
        content = f.read()
    tail = content[-1000:]
    if "FOAM FATAL ERROR" in content:
        return "Crashed"
    if not (("End" in tail) or ("Finalising parallel run" in tail)):
        return "Crashed"
    for val in residuals.values():
        if math.isnan(val) or math.isinf(val):
            return "Diverged"
    checks = {
        "p": thresholds.get("p", 1e-4),
        "Ux": thresholds.get("U", 1e-5),
        "Uy": thresholds.get("U", 1e-5),
        "Uz": thresholds.get("U", 1e-5),
        "k": thresholds.get("k", 1e-5),
        "epsilon": thresholds.get("epsilon", 1e-5),
    }
    for field, threshold in checks.items():
        if field in residuals and residuals[field] > threshold:
            return "NotConverged"
    return "Converged"


# ======================================================================
# Kraefte-Extraktion
# ======================================================================

def _extract_forces(case_dir):
    """Extrahiert Drag und Lift aus postProcessing-Ergebnissen."""
    import glob
    patterns = [
        os.path.join(case_dir, "postProcessing/forces/*/force.dat"),
        os.path.join(case_dir, "postProcessing/forces/*/forces.dat"),
        os.path.join(
            case_dir, "processor0/postProcessing/forces/*/force*.dat"),
    ]
    forces_files = []
    for p in patterns:
        forces_files = glob.glob(p)
        if forces_files:
            break
    if not forces_files:
        return float("nan"), float("nan")
    try:
        with open(sorted(forces_files)[-1], "r") as f:
            lines = f.readlines()
        for line in reversed(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                numbers = re.findall(
                    r"[-+]?[\d]*\.?\d+(?:[eE][-+]?\d+)?", stripped)
                if len(numbers) >= 7:
                    nums = [float(n) for n in numbers]
                    return nums[1] + nums[4], nums[3] + nums[6]
                break
    except Exception:
        pass
    return float("nan"), float("nan")


# ======================================================================
# Cleanup
# ======================================================================

def _cleanup_time_dirs(case_dir, end_time):
    """Loescht Zeitschritt-Ordner ausser 0 und dem letzten Schritt."""
    for item in os.listdir(case_dir):
        item_path = os.path.join(case_dir, item)
        if not os.path.isdir(item_path):
            continue
        try:
            t = float(item)
            if t != 0 and t != end_time:
                shutil.rmtree(item_path)
        except ValueError:
            pass


# ======================================================================
# Dictionary-Schreiber: System-Dateien
# ======================================================================

def _write_system_dictionaries(case_dir, bounds, params, cfg):
    """Schreibt system/ Dictionaries (controlDict, fvSchemes, etc.)."""
    (min_x, max_x, min_y, max_y, min_z, max_z) = bounds
    H = max_z

    # --- Domain-Berechnung (AIJ: 5H upstream, 15H downstream, 5H lateral) ---
    dom = cfg["domain"]
    x_min = min_x - dom["upstream"] * H
    x_max = max_x + dom["downstream"] * H
    y_min = min_y - dom["lateral"] * H
    y_max = max_y + dom["lateral"] * H
    z_max = max_z + dom["top"] * H

    # Blockage-Check
    frontal_area = (max_x - min_x) * max_z
    channel_cross = (y_max - y_min) * z_max
    blockage = frontal_area / channel_cross
    if blockage > dom["max_blockage"]:
        print(f"   [WARNUNG] Blockage = {blockage:.4f} > {dom['max_blockage']}")

    # --- blockMesh Zellberechnung ---
    cells_per_H = cfg["mesh"]["cells_per_H"]
    cell_size = H / cells_per_H
    nx = max(10, int(round((x_max - x_min) / cell_size)))
    ny = max(10, int(round((y_max - y_min) / cell_size)))
    nz = max(10, int(round(z_max / cell_size)))

    # Vertikales Grading (grosse Zellen am Boden fuer Ks-Kompatibilitaet)
    z_grading = cfg["mesh"].get("z_grading", 1.0)

    mcfg = cfg["mesh"]
    rcfg = mcfg["refinement"]
    lcfg = mcfg["layers"]
    qcfg = mcfg["quality"]
    scfg = cfg["solver"]
    num_cores = cfg["hardware"]["num_cores"]

    # --- Refinement-Boxen ---
    # WICHTIG: Alle Boxen starten bei box_z_min statt z=0
    # Damit bleiben die Bodenzellen im Basislevel (yp > Ks)
    box_z_min = mcfg.get("box_z_min", 0.0)

    # wakeBox: grosse Nachlaufzone
    wake_min_x = min_x - 1.5 * H
    wake_max_x = max_x + 8.0 * H
    wake_min_y = min_y - 2.0 * H
    wake_max_y = max_y + 2.0 * H
    wake_max_z = max_z + 2.0 * H

    # nearBox: mittlere Zone (AIJ-konform, H-basiert)
    near_offset = 3.0 * H
    near_min_x = min_x - near_offset
    near_max_x = max_x + near_offset
    near_min_y = min_y - near_offset
    near_max_y = max_y + near_offset
    near_max_z = max_z + near_offset

    # closeBox: enge Box direkt am Gebaeude (feste Meter)
    close_offset = 1.5  # [m] fest, nicht H-abhaengig
    close_min_x = min_x - close_offset
    close_max_x = max_x + close_offset
    close_min_y = min_y - close_offset
    close_max_y = max_y + close_offset
    close_max_z = max_z + close_offset

    # locationInMesh (muss AUSSERHALB des Gebaeudes liegen)
    loc_x = x_min + 0.1 * (x_max - x_min)
    loc_y = (y_min + y_max) / 2.0 + 0.01  # leicht off-center, verhindert Partition-Grenze bei y=0
    loc_z = z_max * 0.9

    surf_min, surf_max = rcfg["surface_level"]

    # --- system/controlDict ---
    _write_file(case_dir, "system/controlDict", f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     simpleFoam;

startFrom       startTime;
startTime       0;

stopAt          endTime;
endTime         {scfg['end_time']};

deltaT          1;

writeControl    timeStep;
writeInterval   {scfg['write_interval']};
purgeWrite      {scfg['purge_write']};

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

    # --- system/fvSchemes ---
    # Quelle: OpenFOAM v2506 windAroundBuildings Tutorial
    # AIJ-Spec Abschnitt 8.1:
    #   div(phi,U): zweite Ordnung mit Limiter (linearUpwind)
    #   div(phi,k), div(phi,epsilon): limitedLinear 1 (oder upwind falls noetig)
    _write_fvSchemes(case_dir, second_order=True)

    # --- system/fvSolution ---
    # Quelle: OpenFOAM v2506 windAroundBuildings Tutorial
    # AIJ-Spec Abschnitt 8.2:
    #   Relaxation p: 0.2-0.3, U: 0.5-0.7, k/eps: 0.5-0.7
    _write_file(case_dir, "system/fvSolution", f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{{
    p
    {{
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
        processorAgglomerator masterCoarsest;
    }}

    "(U|k|epsilon)"
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
    }}

    Phi
    {{
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-6;
        relTol          0.01;
        processorAgglomerator masterCoarsest;
    }}
}}

SIMPLE
{{
    consistent      yes;
    nNonOrthogonalCorrectors 1;

    residualControl
    {{
        p               {scfg['convergence']['p']};
        U               {scfg['convergence']['U']};
        k               {scfg['convergence']['k']};
        "(epsilon)"     {scfg['convergence']['epsilon']};
    }}

    pRefCell        0;
    pRefValue       0;
}}

potentialFlow
{{
    nNonOrthogonalCorrectors 10;
}}

relaxationFactors
{{
    fields
    {{
        p               0.9;
    }}
    equations
    {{
        U               0.9;
        "(k|epsilon).*" 0.7;
    }}
}}

// ************************************************************************* //
""")

    # --- system/blockMeshDict ---
    # z-Grading: simpleGrading (1 1 z_grading)
    # z_grading < 1 bedeutet: Zellen am Boden (z=0) groesser als oben
    _write_file(case_dir, "system/blockMeshDict", f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scale   1;

vertices
(
    ({x_min:.4f} {y_min:.4f} 0)
    ({x_max:.4f} {y_min:.4f} 0)
    ({x_max:.4f} {y_max:.4f} 0)
    ({x_min:.4f} {y_max:.4f} 0)
    ({x_min:.4f} {y_min:.4f} {z_max:.4f})
    ({x_max:.4f} {y_min:.4f} {z_max:.4f})
    ({x_max:.4f} {y_max:.4f} {z_max:.4f})
    ({x_min:.4f} {y_max:.4f} {z_max:.4f})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 {z_grading})
);

edges ();

boundary
(
    inlet
    {{
        type patch;
        faces ( (0 4 7 3) );
    }}
    outlet
    {{
        type patch;
        faces ( (1 2 6 5) );
    }}
    ground
    {{
        type wall;
        faces ( (0 3 2 1) );
    }}
    top
    {{
        type patch;
        faces ( (4 5 6 7) );
    }}
    sides
    {{
        type patch;
        faces ( (0 1 5 4) (3 7 6 2) );
    }}
);

mergePatchPairs ();

// ************************************************************************* //
""")

    # --- system/snappyHexMeshDict ---
    # ALLE Boxen starten bei box_z_min statt z=0
    _write_file(case_dir, "system/snappyHexMeshDict", f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

castellatedMesh true;
snap            true;
addLayers       true;

geometry
{{
    building.stl
    {{
        type    triSurfaceMesh;
        name    building;
    }}
    wakeBox
    {{
        type box;
        min ({wake_min_x:.4f} {wake_min_y:.4f} {box_z_min:.4f});
        max ({wake_max_x:.4f} {wake_max_y:.4f} {wake_max_z:.4f});
    }}
    nearBox
    {{
        type box;
        min ({near_min_x:.4f} {near_min_y:.4f} {box_z_min:.4f});
        max ({near_max_x:.4f} {near_max_y:.4f} {near_max_z:.4f});
    }}
    closeBox
    {{
        type box;
        min ({close_min_x:.4f} {close_min_y:.4f} {box_z_min:.4f});
        max ({close_max_x:.4f} {close_max_y:.4f} {close_max_z:.4f});
    }}
}}

castellatedMeshControls
{{
    maxLocalCells   {mcfg['max_local_cells']};
    maxGlobalCells  {mcfg['max_global_cells']};
    minRefinementCells 10;
    nCellsBetweenLevels 3;
    maxLoadUnbalance 0.10;
    features ( {{ file "building.eMesh"; level {rcfg['feature_level']}; }} );
    refinementSurfaces {{ building {{ level ({surf_min} {surf_max}); }} }}
    resolveFeatureAngle 30;
    refinementRegions
    {{
        wakeBox  {{ mode inside; levels ((1E15 {rcfg['wake_level']})); }}
        nearBox  {{ mode inside; levels ((1E15 {rcfg['near_level']})); }}
        closeBox {{ mode inside; levels ((1E15 {rcfg['close_level']})); }}
    }}
    locationInMesh ({loc_x:.4f} {loc_y:.4f} {loc_z:.4f});
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false; explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes true;
    layers {{ building {{ nSurfaceLayers {lcfg['n_surface_layers']}; }} }}
    expansionRatio {lcfg['expansion_ratio']};
    finalLayerThickness {lcfg['final_layer_thickness']};
    minThickness {lcfg['min_thickness']};
    nGrow 0; featureAngle 60; slipFeatureAngle 30; nRelaxIter 3;
    nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
    maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50;
}}

meshQualityControls
{{
    maxNonOrtho {qcfg['max_non_ortho']}; maxBoundarySkewness 20;
    maxInternalSkewness {qcfg['max_skewness']}; maxConcave 80;
    minTetQuality 1e-30; minArea -1; minTwist 0.05; minDeterminant 0.001;
    minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1;
    minVol 1e-13; minFlatness 0.5; nSmoothScale 4; errorReduction 0.75;
}}

mergeTolerance 1E-6;

// ************************************************************************* //
""")

    # --- system/surfaceFeatureExtractDict ---
    _write_file(case_dir, "system/surfaceFeatureExtractDict", """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      surfaceFeatureExtractDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

building.stl
{
    extractionMethod    extractFromSurface;
    includedAngle       150;
    subsetFeatures
    {
        nonManifoldEdges    no;
        openEdges           yes;
    }
    writeObj            yes;
}

// ************************************************************************* //
""")

    # --- system/decomposeParDict ---
    _write_file(case_dir, "system/decomposeParDict", f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

numberOfSubdomains  {num_cores};

method              scotch;

// ************************************************************************* //
""")


# ======================================================================
# Dictionary-Schreiber: constant/
# ======================================================================

def _write_constant_dictionaries(case_dir, cfg):
    """Schreibt constant/ Dictionaries."""

    _write_file(case_dir, "constant/transportProperties", """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

transportModel  Newtonian;

nu              1.5e-05;

// ************************************************************************* //
""")

    _write_file(case_dir, "constant/turbulenceProperties", """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

simulationType  RAS;

RAS
{
    RASModel        kEpsilon;
    turbulence      on;
    printCoeffs     on;
}

// ************************************************************************* //
""")


# ======================================================================
# Dictionary-Schreiber: 0/ Boundary Fields
# ======================================================================

def _write_boundary_fields(case_dir, params, cfg, include_building=False):
    """
    Schreibt die 0/-Dateien (U, p, k, epsilon, nut).

    Parameter
    ---------
    include_building : bool
        False = Erste Phase (vor snappyHexMesh, building-Patch existiert nicht)
        True  = Zweite Phase (nach snappyHexMesh, building-Patch vorhanden)
    """
    U_ref = params["U_ref"]
    abl = cfg["abl"]
    alpha = abl["alpha"]
    z_ref = abl["z_ref"]
    I_turb = abl["turbulence_intensity"]
    Cmu = 0.09
    Lt = abl["length_scale"]
    k_ref = 1.5 * (I_turb * U_ref) ** 2
    eps_ref = Cmu ** 0.75 * k_ref ** 1.5 / Lt
    z0 = abl["z0"]
    Ks = abl["Ks"]
    Cs_rough = abl["Cs"]

    # Building-Patch Eintraege (nur wenn Patch existiert)
    if include_building:
        U_building = """\
    building
    {
        type            noSlip;
    }"""
        p_building = """\
    building    { type zeroGradient; }"""
        k_building = f"""\
    building    {{ type kqRWallFunction; value uniform {k_ref:.6f}; }}"""
        eps_building = f"""\
    building    {{ type epsilonWallFunction; value uniform {eps_ref:.8f}; }}"""
        nut_building = """\
    building
    {
        type            nutkWallFunction;
        value           uniform 0;
    }"""
    else:
        U_building = ""
        p_building = ""
        k_building = ""
        eps_building = ""
        nut_building = ""

    # --- 0/U ---
    _write_file(case_dir, "0/U", f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
    inlet
    {{
        type            codedFixedValue;
        value           uniform ({U_ref} 0 0);
        name            ablVelocityProfile;
        code
        #{{
            // ABL Power-Law: U(z) = U_ref * (z / z_ref)^alpha
            const scalar Uref  = {U_ref};
            const scalar zRef  = {z_ref};
            const scalar alpha = {alpha};
            const scalar zMin  = 0.001;
            const fvPatch& patch = this->patch();
            const vectorField& Cf = patch.Cf();
            vectorField& field = *this;
            forAll(Cf, faceI)
            {{
                scalar z = max(Cf[faceI].z(), zMin);
                scalar Uz = Uref * pow(z / zRef, alpha);
                field[faceI] = vector(Uz, 0, 0);
            }}
        #}};
    }}
    outlet
    {{
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }}
    ground
    {{
        type            noSlip;
    }}
{U_building}
    top
    {{
        type            slip;
    }}
    sides
    {{
        type            slip;
    }}
}}

// ************************************************************************* //
""")

    # --- 0/p ---
    _write_file(case_dir, "0/p", f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    inlet       {{ type zeroGradient; }}
    outlet      {{ type totalPressure; p0 uniform 0; }}
    ground      {{ type zeroGradient; }}
{p_building}
    top         {{ type zeroGradient; }}
    sides       {{ type zeroGradient; }}
}}

// ************************************************************************* //
""")

    # --- 0/k ---
    _write_file(case_dir, "0/k", f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      k;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform {k_ref:.6f};

boundaryField
{{
    inlet
    {{
        type            codedFixedValue;
        value           uniform {k_ref:.6f};
        name            ablKProfile;
        code
        #{{
            // k(z) = 1.5 * (I * U(z))^2
            const scalar Uref  = {U_ref};
            const scalar zRef  = {z_ref};
            const scalar alpha = {alpha};
            const scalar I     = {I_turb};
            const scalar zMin  = 0.001;
            const fvPatch& patch = this->patch();
            const vectorField& Cf = patch.Cf();
            scalarField& field = *this;
            forAll(Cf, faceI)
            {{
                scalar z = max(Cf[faceI].z(), zMin);
                scalar Uz = Uref * pow(z / zRef, alpha);
                field[faceI] = 1.5 * pow(I * Uz, 2);
            }}
        #}};
    }}
    outlet      {{ type zeroGradient; }}
    ground      {{ type kqRWallFunction; value uniform {k_ref:.6f}; }}
{k_building}
    top         {{ type zeroGradient; }}
    sides       {{ type zeroGradient; }}
}}

// ************************************************************************* //
""")

    # --- 0/epsilon ---
    _write_file(case_dir, "0/epsilon", f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      epsilon;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -3 0 0 0 0];

internalField   uniform {eps_ref:.8f};

boundaryField
{{
    inlet
    {{
        type            codedFixedValue;
        value           uniform {eps_ref:.8f};
        name            ablEpsilonProfile;
        code
        #{{
            // epsilon(z) = Cmu^0.75 * k(z)^1.5 / Lt
            const scalar Uref  = {U_ref};
            const scalar zRef  = {z_ref};
            const scalar alpha = {alpha};
            const scalar I     = {I_turb};
            const scalar Cmu   = {Cmu};
            const scalar Lt    = {Lt};
            const scalar zMin  = 0.001;
            const fvPatch& patch = this->patch();
            const vectorField& Cf = patch.Cf();
            scalarField& field = *this;
            forAll(Cf, faceI)
            {{
                scalar z = max(Cf[faceI].z(), zMin);
                scalar Uz = Uref * pow(z / zRef, alpha);
                scalar kz = 1.5 * pow(I * Uz, 2);
                field[faceI] = pow(Cmu, 0.75) * pow(kz, 1.5) / Lt;
            }}
        #}};
    }}
    outlet      {{ type zeroGradient; }}
    ground      {{ type atmEpsilonWallFunction; z0 uniform {z0}; value uniform {eps_ref:.8f}; }}
{eps_building}
    top         {{ type zeroGradient; }}
    sides       {{ type zeroGradient; }}
}}

// ************************************************************************* //
""")

    # --- 0/nut ---
    _write_file(case_dir, "0/nut", f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      nut;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    inlet       {{ type calculated; value uniform 0; }}
    outlet      {{ type calculated; value uniform 0; }}
    ground
    {{
        type            atmNutkWallFunction;
        z0              uniform {z0};
        boundNut        true;
        value           uniform 0;
    }}
{nut_building}
    top         {{ type calculated; value uniform 0; }}
    sides       {{ type calculated; value uniform 0; }}
}}

// ************************************************************************* //
""")


# ======================================================================
# Hilfsfunktion
# ======================================================================

def _write_file(case_dir, rel_path, content):
    """Schreibt eine Datei relativ zum Case-Verzeichnis."""
    filepath = os.path.join(case_dir, rel_path)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def _write_fvSchemes(case_dir, second_order=True):
    """Schreibt system/fvSchemes mit First- oder Second-Order-Schemata.

    Parameter
    ---------
    case_dir : str
        Pfad zum OpenFOAM-Case-Verzeichnis.
    second_order : bool
        True  = Second-Order (linearUpwind / limitedLinear) fuer Produktionslaeufe
        False = First-Order (upwind) fuer numerische Stabilisierung in der Startphase
    """
    if second_order:
        div_U   = "bounded Gauss linearUpwind limited"
        div_turb = "bounded Gauss limitedLinear 1"
    else:
        div_U   = "bounded Gauss upwind"
        div_turb = "bounded Gauss upwind"

    _write_file(case_dir, "system/fvSchemes", f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{{
    default         steadyState;
}}

gradSchemes
{{
    default         Gauss linear;
    limited         cellLimited Gauss linear 1;
    grad(U)         $limited;
    grad(k)         $limited;
    grad(epsilon)   $limited;
}}

divSchemes
{{
    default         none;
    div(phi,U)      {div_U};
    turbulence      {div_turb};
    div(phi,k)      $turbulence;
    div(phi,epsilon) $turbulence;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}

laplacianSchemes
{{
    default         Gauss linear corrected;
}}

interpolationSchemes
{{
    default         linear;
}}

snGradSchemes
{{
    default         corrected;
}}

wallDist
{{
    method          meshWave;
}}

// ************************************************************************* //
""")