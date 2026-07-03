"""
main.py
=======
Steuert die automatisierte CFD-Datensatzgenerierung.

Liest die Konfiguration aus config.yaml, erzeugt fuer jede Simulation
eine rotierte Gebaeude-STL und fuehrt die OpenFOAM-Simulation durch.

Unterstuetzt parallele Ausfuehrung mehrerer Simulationen gleichzeitig
(konfigurierbar ueber hardware.parallel_jobs). Jede Simulation laeuft
in einem eigenen Prozess mit anteiliger Kernzahl.

Verwendung:
    python main.py
    python main.py --config meine_config.yaml
"""

import os
import sys
import csv
import copy
import time
import random
import shutil
import argparse
import tarfile
import subprocess
from datetime import datetime
from multiprocessing import Pool

import yaml

import createGeometry
import runSimulation


# ======================================================================
# Konfiguration laden
# ======================================================================

def load_config(config_path):
    """Laedt die YAML-Konfigurationsdatei."""
    if not os.path.exists(config_path):
        print(f"FEHLER: Konfigurationsdatei '{config_path}' nicht gefunden.")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


# ======================================================================
# Zufaellige Simulationsparameter erzeugen
# ======================================================================

def generate_tasks(cfg):
    """
    Erzeugt eine Liste von Simulationsaufgaben mit zufaelligen
    Windgeschwindigkeiten und -winkeln.
    """
    n = cfg["general"]["num_simulations"]
    w = cfg["wind"]

    tasks = []
    for i in range(n):
        tasks.append({
            "id": f"sim_{i:03d}",
            "U_ref": round(random.uniform(w["speed_min"], w["speed_max"]), 2),
            "angle": round(random.uniform(w["angle_min"], w["angle_max"]), 2),
        })

    return tasks


# ======================================================================
# Statusanzeige
# ======================================================================

def print_status(case_name, step, total, desc, sim_num=0, sim_total=0):
    """Einzeilige, ueberschreibende Statusanzeige."""
    msg = f"\r[{sim_num}/{sim_total}] [{case_name}] Schritt {step}/{total}: {desc}"
    sys.stdout.write(f"{msg:<80}")
    sys.stdout.flush()


# ======================================================================
# CSV-Ergebnisse
# ======================================================================

def write_csv_header(csv_path):
    """Schreibt die CSV-Header-Zeile."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ID", "U_ref", "Angle",
            "Status", "Failed_Step",
            "Res_p", "Res_Ux", "Res_Uy", "Res_Uz", "Res_k", "Res_epsilon",
            "Drag", "Lift",
        ])


def write_csv_row(csv_path, task, result):
    """Fuegt eine Ergebniszeile zur CSV hinzu."""
    res = result.get("residuals", {})
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            task["id"],
            task["U_ref"],
            task["angle"],
            result["status"],
            result.get("failed_step", ""),
            res.get("p", "NaN"),
            res.get("Ux", "NaN"),
            res.get("Uy", "NaN"),
            res.get("Uz", "NaN"),
            res.get("k", "NaN"),
            res.get("epsilon", "NaN"),
            result.get("drag", "NaN"),
            result.get("lift", "NaN"),
        ])


# ======================================================================
# Fehler-Logging
# ======================================================================

def write_error_header(error_log_path):
    """Erstellt die errors.log mit Header."""
    with open(error_log_path, "w") as f:
        f.write(f"{'=' * 72}\n")
        f.write(f"  FEHLERPROTOKOLL - CFD-Datensatzgenerierung\n")
        f.write(f"  Erstellt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 72}\n\n")


def _verify_s3_upload(tar_path, s3_dest):
    """
    Prueft ob die hochgeladene Datei in S3 existiert und die
    korrekte Groesse hat.

    Rueckgabe
    ---------
    bool : True wenn Datei in S3 mit korrekter Groesse vorhanden.
    """
    local_size = os.path.getsize(tar_path)
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", s3_dest],
            capture_output=True, check=True,
        )
        # Ausgabe: "2026-02-22 04:34:44  735822767 sim_001.tar.gz"
        parts = result.stdout.decode().strip().split()
        if len(parts) >= 3:
            remote_size = int(parts[2])
            return remote_size == local_size
    except Exception:
        pass
    return False


def upload_vtk_to_s3(vtk_dir, sim_id, s3_cfg, base_path, max_retries=3):
    """
    Komprimiert den VTK-Ordner als tar.gz und laedt ihn nach S3 hoch.
    Loescht die lokale tar.gz nach erfolgreichem und verifiziertem Upload.

    Bei Netzwerkfehlern wird der Upload bis zu max_retries Mal
    wiederholt (exponentielles Backoff: 10s, 20s, 40s).
    Nach jedem Upload wird die Dateigroesse in S3 gegen die lokale
    Datei geprueft.

    Parameter
    ---------
    vtk_dir : str
        Pfad zum VTK-Ordner (z.B. results/vtks/sim_000_VTK).
    sim_id : str
        Simulationsname (z.B. sim_000).
    s3_cfg : dict
        S3-Konfiguration aus config.yaml.
    base_path : str
        Arbeitsverzeichnis.
    max_retries : int
        Maximale Anzahl Upload-Versuche (Standard: 3).

    Rueckgabe
    ---------
    bool : True bei Erfolg, False bei Fehler.
    """
    tar_path = os.path.join(base_path, f"{sim_id}.tar.gz")
    s3_dest = f"s3://{s3_cfg['bucket']}/{s3_cfg['prefix']}/{sim_id}.tar.gz"

    # Groessere Multipart-Chunks (64 MB statt 8 MB default)
    # Reduziert Anzahl der Parts von ~88 auf ~11 bei 700 MB Dateien
    s3_config = (
        "multipart_threshold=64MB "
        "multipart_chunksize=64MB"
    )

    try:
        # tar.gz nur erstellen, wenn noch nicht vorhanden (Retry-Fall)
        if not os.path.exists(tar_path):
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(vtk_dir, arcname=sim_id)

        for attempt in range(1, max_retries + 1):
            try:
                subprocess.run(
                    ["aws", "s3", "cp", tar_path, s3_dest,
                     "--cli-read-timeout", "120",
                     "--cli-connect-timeout", "30",
                     "--s3-transfer-config", s3_config],
                    check=True,
                    capture_output=True,
                )

                # Verifikation: Dateigroesse in S3 pruefen
                if _verify_s3_upload(tar_path, s3_dest):
                    os.remove(tar_path)
                    return True
                else:
                    raise RuntimeError("Verifikation fehlgeschlagen: "
                                       "Dateigroesse in S3 stimmt nicht")

            except Exception as e:
                if attempt < max_retries:
                    wait = 10 * (2 ** (attempt - 1))
                    print(f"\n   WARNUNG: S3-Upload {sim_id} fehlgeschlagen "
                          f"(Versuch {attempt}/{max_retries}), "
                          f"naechster Versuch in {wait}s...")
                    time.sleep(wait)
                else:
                    err_msg = str(e)
                    if hasattr(e, "stderr") and e.stderr:
                        err_msg = e.stderr.decode(
                            "utf-8", errors="replace").strip()
                    print(f"\n   FEHLER: S3-Upload {sim_id} endgueltig "
                          f"fehlgeschlagen nach {max_retries} Versuchen: "
                          f"{err_msg}")

        # Alle Versuche gescheitert — tar.gz behalten fuer manuellen Retry
        return False
    except Exception as e:
        print(f"\n   WARNUNG: S3-Upload fuer {sim_id} fehlgeschlagen: {e}")
        if os.path.exists(tar_path):
            os.remove(tar_path)
        return False


def _run_single_sim(args):
    """
    Worker-Funktion fuer multiprocessing.Pool.

    Laeuft in einem separaten Prozess — os.chdir() in runSimulation
    beeinflusst andere Worker nicht.

    Parameter
    ---------
    args : tuple
        (task, stl_path, bounds, cfg_worker)

    Rueckgabe
    ---------
    tuple : (task, result, sim_time)
    """
    task, stl_path, bounds, cfg_worker = args
    sim_start = time.time()
    try:
        result = runSimulation.run_case(
            case_name=task["id"],
            stl_source=stl_path,
            params=task,
            bounds=bounds,
            cfg=cfg_worker,
            status_callback=None,
        )
    except Exception as e:
        result = {
            "id": task["id"],
            "status": "Error",
            "failed_step": "run_case",
            "error_message": str(e),
        }
    sim_time = time.time() - sim_start
    return (task, result, sim_time)


def write_error_entry(error_log_path, task, result, sim_time):
    """
    Schreibt einen detaillierten Fehlereintrag in die errors.log.

    Enthaelt: Simulationsname, Parameter, fehlgeschlagener Schritt,
    Fehlermeldung, Python-Traceback und die letzten Log-Zeilen.
    """
    with open(error_log_path, "a") as f:
        f.write(f"{'- ' * 36}\n")
        f.write(f"  FEHLER: {task['id']}\n")
        f.write(f"{'- ' * 36}\n")
        f.write(f"  Zeitstempel:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Laufzeit:        {sim_time:.0f}s\n")
        f.write(f"  U_ref:           {task['U_ref']} m/s\n")
        f.write(f"  Windwinkel:      {task['angle']} Grad\n")
        f.write(f"  Status:          {result['status']}\n")

        if result.get("failed_step"):
            f.write(f"  Abgebrochen bei: {result['failed_step']}\n")

        if result.get("error_message"):
            f.write(f"  Fehlermeldung:   {result['error_message']}\n")

        if result.get("error_logfile"):
            f.write(f"  Log-Datei:       {result['error_logfile']}\n")

        # Python-Traceback
        if result.get("error_traceback"):
            f.write(f"\n  --- Python Traceback ---\n")
            for line in result["error_traceback"].splitlines():
                f.write(f"  {line}\n")

        # Letzte Zeilen aus dem OpenFOAM-Log
        if result.get("error_log_tail"):
            f.write(f"\n  --- Letzte Log-Zeilen ---\n")
            for line in result["error_log_tail"].splitlines():
                f.write(f"  {line}\n")

        f.write(f"\n\n")


# ======================================================================
# Hauptprogramm
# ======================================================================

def main():
    # ------------------------------------------------------------------
    # Argumente parsen
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="CFD-Datensatzgenerierung fuer GNN-Training"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Pfad zur Konfigurationsdatei (Standard: config.yaml)"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Konfiguration laden
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    gen = cfg["general"]
    geo = cfg["geometry"]
    hw = cfg["hardware"]

    # ------------------------------------------------------------------
    # Reproduzierbarkeit
    # ------------------------------------------------------------------
    random.seed(gen["random_seed"])

    # ------------------------------------------------------------------
    # Verzeichnisse erstellen
    # ------------------------------------------------------------------
    base_path = os.getcwd()
    for d in [gen["results_dir"], gen["geometry_dir"], gen["simulation_dir"]]:
        os.makedirs(os.path.join(base_path, d), exist_ok=True)

    # ------------------------------------------------------------------
    # Aufgaben generieren
    # ------------------------------------------------------------------
    tasks = generate_tasks(cfg)
    H = geo["wall_height"] + geo["roof_height"]

    # ------------------------------------------------------------------
    # Parallelisierung konfigurieren
    # ------------------------------------------------------------------
    parallel_jobs = hw.get("parallel_jobs", 1)
    num_cores = hw["num_cores"]
    cores_per_job = num_cores // max(parallel_jobs, 1)

    # ------------------------------------------------------------------
    # Setup-Uebersicht
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"   CFD-DATENSATZ GENERIERUNG")
    print(f"{'=' * 60}")
    print(f"")
    print(f"   Simulationen:    {len(tasks)}")
    print(f"   Gebaeude:        {geo['width']} x {geo['depth']} x {H} m (B x T x H)")
    print(f"   Windbereich:     {cfg['wind']['speed_min']}-{cfg['wind']['speed_max']} m/s")
    print(f"   Winkelbereich:   {cfg['wind']['angle_min']}-{cfg['wind']['angle_max']} Grad")
    print(f"   CPU-Kerne:       {num_cores} ({parallel_jobs} Jobs x {cores_per_job} Kerne)")
    print(f"   ABL-Profil:      Power Law (alpha={cfg['abl']['alpha']})")
    print(f"   Max. Zellen:     {cfg['mesh']['max_global_cells']:,}")
    print(f"   VTK-Export:      {'Ja' if gen['save_vtk'] else 'Nein'}")
    s3_cfg = cfg.get("s3", {})
    s3_enabled = s3_cfg.get("enabled", False)
    if s3_enabled:
        print(f"   S3-Bucket:       s3://{s3_cfg['bucket']}/{s3_cfg['prefix']}/")
    print(f"")
    print(f"{'=' * 60}")
    print(f"   STARTE SIMULATIONEN")
    print(f"{'=' * 60}")
    print(f"")

    # ------------------------------------------------------------------
    # Ausgabedateien vorbereiten
    # ------------------------------------------------------------------
    csv_path = os.path.join(base_path, gen["results_dir"], "dataset_overview.csv")
    error_log_path = os.path.join(base_path, gen["results_dir"], "errors.log")
    write_csv_header(csv_path)
    write_error_header(error_log_path)

    # ------------------------------------------------------------------
    # Geometrien vorab erzeugen (schnell, numpy-only)
    # ------------------------------------------------------------------
    geometry_data = []
    for task in tasks:
        stl_path = os.path.join(
            base_path, gen["geometry_dir"], f"{task['id']}.stl"
        )
        bounds = createGeometry.create_building(
            geo["width"],
            geo["depth"],
            geo["wall_height"],
            geo["roof_height"],
            task["angle"],
            stl_path,
        )
        geometry_data.append((stl_path, bounds))

    # ------------------------------------------------------------------
    # Simulationen ausfuehren
    # ------------------------------------------------------------------
    start_time = time.time()
    results = []
    n_success = 0
    n_not_converged = 0
    n_failed = 0

    if parallel_jobs > 1:
        # ==============================================================
        # PARALLELER MODUS: multiprocessing.Pool
        # ==============================================================
        cfg_worker = copy.deepcopy(cfg)
        cfg_worker["hardware"]["num_cores"] = cores_per_job

        print(f"   Paralleler Modus: {parallel_jobs} Jobs x {cores_per_job} Kerne")
        print(f"")

        # Worker-Argumente vorbereiten
        worker_args = []
        for task, (stl_path, bounds) in zip(tasks, geometry_data):
            worker_args.append((task, stl_path, bounds, cfg_worker))

        # Pool starten und Ergebnisse einsammeln
        pool = Pool(processes=parallel_jobs)
        async_results = []
        for wa in worker_args:
            ar = pool.apply_async(_run_single_sim, (wa,))
            async_results.append(ar)
        pool.close()

        for i, ar in enumerate(async_results):
            task, result, sim_time = ar.get()

            results.append(result)
            write_csv_row(csv_path, task, result)

            status = result["status"]
            if status == "Converged":
                n_success += 1
                status_icon = "OK"
            elif status == "NotConverged":
                n_not_converged += 1
                status_icon = "!!"
            else:
                n_failed += 1
                status_icon = "XX"
                write_error_entry(error_log_path, task, result, sim_time)

            line = (
                f"   [{status_icon}] [{i+1}/{len(tasks)}] {task['id']}: "
                f"{status} ({sim_time:.0f}s)"
            )
            if result.get("failed_step"):
                line += f" -> {result['failed_step']}"

            # S3-Upload bei erfolgreicher Simulation
            if result["status"] in ("Converged", "NotConverged"):
                vtk_dir = os.path.join(base_path, gen["results_dir"], "vtks", f"{task['id']}_VTK")
                if s3_enabled and os.path.isdir(vtk_dir):
                    s3_ok = upload_vtk_to_s3(vtk_dir, task["id"], s3_cfg, base_path)
                    if s3_ok:
                        shutil.rmtree(vtk_dir)
                        line += " [S3 OK, VTK geloescht]"
                    else:
                        line += " [S3 FEHLER, VTK behalten]"

            # Case-Ordner loeschen bei erfolgreicher Simulation
            delete_case = cfg["cleanup"].get("delete_case_after_vtk", False)
            if delete_case and result["status"] in ("Converged", "NotConverged"):
                case_dir = os.path.join(base_path, gen["simulation_dir"], task["id"])
                if os.path.exists(case_dir):
                    shutil.rmtree(case_dir)
                line += " [Case geloescht]"

            print(f"{line:<80}")

        pool.join()

    else:
        # ==============================================================
        # SEQUENZIELLER MODUS (wie bisher, mit status_callback)
        # ==============================================================
        for i, (task, (stl_path, bounds)) in enumerate(zip(tasks, geometry_data)):
            sim_start = time.time()

            # Status-Callback
            def status_cb(name, step, total, desc, _i=i, _n=len(tasks)):
                print_status(name, step, total, desc, _i + 1, _n)

            # Simulation ausfuehren
            result = runSimulation.run_case(
                case_name=task["id"],
                stl_source=stl_path,
                params=task,
                bounds=bounds,
                cfg=cfg,
                status_callback=status_cb,
            )

            sim_time = time.time() - sim_start

            results.append(result)
            write_csv_row(csv_path, task, result)

            status = result["status"]
            if status == "Converged":
                n_success += 1
                status_icon = "OK"
            elif status == "NotConverged":
                n_not_converged += 1
                status_icon = "!!"
            else:
                n_failed += 1
                status_icon = "XX"
                write_error_entry(error_log_path, task, result, sim_time)

            line = (
                f"\r   [{status_icon}] [{i+1}/{len(tasks)}] {task['id']}: "
                f"{status} ({sim_time:.0f}s)"
            )
            if result.get("failed_step"):
                line += f" -> {result['failed_step']}"

            # S3-Upload bei erfolgreicher Simulation
            if result["status"] in ("Converged", "NotConverged"):
                vtk_dir = os.path.join(base_path, gen["results_dir"], "vtks", f"{task['id']}_VTK")
                if s3_enabled and os.path.isdir(vtk_dir):
                    s3_ok = upload_vtk_to_s3(vtk_dir, task["id"], s3_cfg, base_path)
                    if s3_ok:
                        shutil.rmtree(vtk_dir)
                        line += " [S3 OK, VTK geloescht]"
                    else:
                        line += " [S3 FEHLER, VTK behalten]"

            # Case-Ordner loeschen bei erfolgreicher Simulation
            delete_case = cfg["cleanup"].get("delete_case_after_vtk", False)
            if delete_case and result["status"] in ("Converged", "NotConverged"):
                case_dir = os.path.join(base_path, gen["simulation_dir"], task["id"])
                if os.path.exists(case_dir):
                    shutil.rmtree(case_dir)
                line += " [Case geloescht]"

            print(f"{line:<80}")

    # ------------------------------------------------------------------
    # Zusammenfassung
    # ------------------------------------------------------------------
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)

    print(f"\n{'=' * 60}")
    print(f"   ZUSAMMENFASSUNG")
    print(f"{'=' * 60}")
    print(f"")
    print(f"   Converged:       {n_success}/{len(tasks)}")
    print(f"   NotConverged:    {n_not_converged}/{len(tasks)}")
    print(f"   Crashed/Error:   {n_failed}/{len(tasks)}")
    print(f"   Gesamtzeit:      {hours}h {minutes}min")
    print(f"")
    print(f"   Ergebnisse:      {csv_path}")
    if gen["save_vtk"]:
        print(f"   VTK-Dateien:     {os.path.join(gen['results_dir'], 'vtks')}")

    # Fehlgeschlagene Simulationen auflisten
    failed = [r for r in results if r["status"] not in ("Converged", "NotConverged")]
    if failed:
        print(f"\n   FEHLGESCHLAGENE SIMULATIONEN ({len(failed)}):")
        print(f"   {'─' * 52}")
        for r in failed:
            step = r.get("failed_step", "Unbekannt")
            msg = r.get("error_message", "Keine Fehlermeldung")
            # Kuerze die Fehlermeldung auf max 60 Zeichen fuer die Konsole
            if len(msg) > 60:
                msg = msg[:57] + "..."
            print(f"   {r['id']:10s} | {step}")
            print(f"{'':14s} | {msg}")
        print(f"   {'─' * 52}")
        print(f"\n   Detaillierte Fehlerbeschreibungen:")
        print(f"   -> {error_log_path}")

    # NotConverged auflisten (weniger kritisch, aber nützlich)
    not_conv = [r for r in results if r["status"] == "NotConverged"]
    if not_conv:
        print(f"\n   NICHT KONVERGIERT ({len(not_conv)}):")
        print(f"   {'─' * 52}")
        for r in not_conv:
            res = r.get("residuals", {})
            worst_field = ""
            worst_val = 0
            for field, val in res.items():
                if val > worst_val:
                    worst_val = val
                    worst_field = field
            print(f"   {r['id']:10s} | Hoechstes Residuum: {worst_field}={worst_val:.2e}")
        print(f"   {'─' * 52}")

    # ------------------------------------------------------------------
    # CSV nach S3 hochladen
    # ------------------------------------------------------------------
    if s3_enabled:
        try:
            subprocess.run(
                ["aws", "s3", "cp", csv_path,
                 f"s3://{s3_cfg['bucket']}/{s3_cfg['prefix']}/dataset_overview.csv"],
                check=True,
            )
            print(f"\n   CSV hochgeladen: s3://{s3_cfg['bucket']}/{s3_cfg['prefix']}/dataset_overview.csv")
        except Exception as e:
            print(f"\n   WARNUNG: CSV-Upload fehlgeschlagen: {e}")

    print(f"\n{'=' * 60}")
    print(f"   FERTIG")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
