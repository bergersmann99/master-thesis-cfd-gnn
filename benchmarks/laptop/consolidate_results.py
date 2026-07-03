"""
Liest alle Timing-YAMLs und prediction_report.json-Dateien aus ~/laptop_timing/
und schreibt eine konsolidierte run_log.yaml mit Speedup-Berechnung.
"""
import os
import re
import json
import statistics

import yaml

BASE = "/home/tim-bergermann/laptop_timing"
LOGS = os.path.join(BASE, "logs")


def load_yaml(path):
    """Liest eine YAML-Datei; None bei fehlender oder unlesbarer Datei."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def load_inference_times(case_name, n_runs=3):
    """Liest die Inferenzzeiten der einzelnen Läufe aus prediction_report.json."""
    times = []
    for i in range(1, n_runs + 1):
        report = os.path.join(BASE, "predictions", f"{case_name}_run_{i}",
                              "prediction_report.json")
        if os.path.exists(report):
            with open(report) as f:
                d = json.load(f)
            times.append(d["inference_time_s"])
    return times


def load_aux_time(key):
    """Liest einen Zeitwert per Schlüssel aus logs/aux_times.yaml."""
    path = os.path.join(LOGS, "aux_times.yaml")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        content = f.read()
    for line in content.splitlines():
        if line.startswith(key + ":"):
            return float(line.split(":")[1].strip())
    return None


def load_simpleFoam_total(solver_stem):
    """Addiert Morgen-Stop + Abend-Resume aus aux_times, falls vorhanden."""
    solver_data = load_yaml(os.path.join(LOGS, f"solver_{solver_stem}.yaml"))
    sf = 0
    if solver_data:
        t = solver_data["timings"]
        sf = (t.get("10a_simpleFoam_phase1", 0) +
              t.get("10b_simpleFoam_phase2", 0) +
              t.get("10_simpleFoam", 0))
    # Prüfe ob ein Resume-Teil in aux_times vorhanden ist
    # resume_solver.sh schreibt z.B. "simpleFoam_resume_schwachwind"
    case_prefix = solver_stem.split("_")[0]  # "sturm" oder "schwachwind"
    resume_s = load_aux_time(f"simpleFoam_resume_{case_prefix}") or load_aux_time("simpleFoam_resume")
    if resume_s:
        sf += resume_s
    return sf, solver_data


def summarize_case(case_key, mesh_stem, solver_stem, sim_id, graph_yaml_name,
                   mesh_stem_ref=None):
    """mesh_stem_ref: falls gesetzt, wird Mesh-Zeit von diesem Fall übernommen (Schwachwind=Sturm)."""
    result = {"case": case_key}

    # CFD Mesh
    actual_mesh_stem = mesh_stem_ref if mesh_stem_ref else mesh_stem
    mesh_data = load_yaml(os.path.join(LOGS, f"mesh_{actual_mesh_stem}.yaml"))
    if mesh_data:
        snappy_s = mesh_data["timings"].get("snappyHexMesh")
        result["snappyHexMesh_s"] = snappy_s
        result["snappyHexMesh_min"] = round(snappy_s / 60, 2) if snappy_s else None
        result["mesh_total_s"] = mesh_data["total_seconds"]
        result["checkMesh_ok"] = mesh_data.get("check_ok")
        if mesh_stem_ref:
            result["mesh_note"] = f"Mesh identisch mit {mesh_stem_ref} — Zeit übernommen"
    else:
        result["snappyHexMesh_s"] = None
        result["mesh_total_s"] = None

    # CFD Solver
    sf, solver_data = load_simpleFoam_total(solver_stem)
    if solver_data or sf > 0:
        result["simpleFoam_s"] = sf
        result["simpleFoam_min"] = round(sf / 60, 2) if sf else None
        result["solver_total_s"] = solver_data["total_seconds"] if solver_data else sf
        status = solver_data.get("status") if solver_data else "partial_or_skipped"
        # Falls Resume-Log vorhanden: Konvergenzstatus daraus übernehmen
        resume_log = os.path.join(
            solver_data["case_dir"] if solver_data else "",
            "log.simpleFoam_resume")
        if os.path.exists(resume_log):
            with open(resume_log) as f:
                content = f.read()
            if "SIMPLE solution converged" in content:
                m = re.search(r"SIMPLE solution converged in (\d+) iterations", content)
                status = "Converged"
                result["simpleFoam_converged_at_iter"] = int(m.group(1)) if m else None
        result["solver_status"] = status
    else:
        result["simpleFoam_s"] = None
        result["solver_total_s"] = None
        result["solver_status"] = "not_run"

    # CFD Gesamt
    snappy = result.get("snappyHexMesh_s")
    sf = result.get("simpleFoam_s")
    if snappy is not None and sf is not None:
        cfd_total = snappy + sf
        result["cfd_total_s"] = cfd_total
        result["cfd_total_min"] = round(cfd_total / 60, 2)
    else:
        result["cfd_total_s"] = None

    # foamToVTK (nicht in GNN-Kette)
    vtk_time = load_aux_time(f"foamToVTK_{case_key.split('_')[0]}")
    result["foamToVTK_s"] = vtk_time

    # Graph-Konstruktion
    graph_data = load_yaml(os.path.join(LOGS, f"graph_{graph_yaml_name}.yaml"))
    if graph_data and graph_data.get("total_seconds"):
        graph_total = graph_data["total_seconds"]
        result["graph_construction_s"] = graph_total
        result["graph_construction_min"] = round(graph_total / 60, 2)
        result["graph_timings"] = graph_data.get("timings_seconds")
        result["n_original"] = graph_data.get("n_original")
        result["n_subsampled"] = graph_data.get("n_subsampled")
    else:
        # Fallback: aux_times.yaml (wenn YAML fehlt oder korrupt)
        graph_total = load_aux_time(f"graph_{graph_yaml_name}")
        result["graph_construction_s"] = graph_total
        result["graph_construction_min"] = round(graph_total / 60, 2) if graph_total else None

    # Inferenz (10 Läufe)
    inf_times = load_inference_times(case_key)
    if inf_times:
        result["inference_n_runs"] = len(inf_times)
        result["inference_mean_s"] = round(statistics.mean(inf_times), 4)
        result["inference_std_s"] = round(statistics.stdev(inf_times), 4) if len(inf_times) > 1 else 0.0
        result["inference_times_s"] = [round(t, 4) for t in inf_times]
    else:
        result["inference_mean_s"] = None

    # IDW
    idw_key = "idw_sturm" if "sturm" in case_key else "idw_schwachwind"
    idw_time = load_aux_time(idw_key)
    result["idw_s"] = idw_time

    # GNN-Kette Gesamt
    gnn_parts = [snappy, result.get("graph_construction_s"),
                 result.get("inference_mean_s"), idw_time]
    if all(p is not None for p in gnn_parts):
        gnn_total = sum(gnn_parts)
        result["gnn_total_s"] = gnn_total
        result["gnn_total_min"] = round(gnn_total / 60, 2)
    else:
        result["gnn_total_s"] = None

    # Speedup
    cfd = result.get("cfd_total_s")
    gnn = result.get("gnn_total_s")
    if cfd and gnn:
        result["speedup"] = round(cfd / gnn, 2)
    else:
        result["speedup"] = None

    return result


def main():
    """Konsolidiert alle Fälle und schreibt run_log.yaml."""
    cases = [
        {
            "case_key":    "sturm",
            "mesh_stem":   "sturm_v2",
            "solver_stem": "sturm_v2",
            "sim_id":      "sturm_25ms_45deg",
            "graph_yaml":  "sturm",
            "mesh_ref":    None,
        },
        {
            "case_key":    "schwachwind",
            "mesh_stem":   "schwachwind_v2",
            "solver_stem": "schwachwind_v2",
            "sim_id":      "schwachwind_1_5ms_45deg",
            "graph_yaml":  "schwachwind",
            "mesh_ref":    "sturm_v2",  # Mesh identisch mit Sturm
        },
    ]

    results = {}
    for c in cases:
        print(f"\nLese Ergebnisse: {c['case_key']} ...")
        r = summarize_case(
            c["case_key"], c["mesh_stem"], c["solver_stem"],
            c["sim_id"], c["graph_yaml"],
            mesh_stem_ref=c.get("mesh_ref"),
        )
        results[c["case_key"]] = r

        print(f"  snappyHexMesh:      {r.get('snappyHexMesh_min')} min")
        print(f"  simpleFoam:         {r.get('simpleFoam_min')} min")
        print(f"  CFD gesamt:         {r.get('cfd_total_min')} min")
        print(f"  Graphkonstruktion:  {r.get('graph_construction_min')} min")
        print(f"  Inferenz (Mittel):  {r.get('inference_mean_s')} s ± {r.get('inference_std_s')} s")
        print(f"  IDW:                {r.get('idw_s')} s")
        print(f"  GNN gesamt:         {r.get('gnn_total_min')} min")
        print(f"  Speedup:            {r.get('speedup')}x")

    out_path = os.path.join(BASE, "run_log.yaml")
    with open(out_path, "w") as f:
        yaml.safe_dump({"laptop_timing": results}, f, sort_keys=False)
    print(f"\n[OK] Konsolidierte Ergebnisse: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
