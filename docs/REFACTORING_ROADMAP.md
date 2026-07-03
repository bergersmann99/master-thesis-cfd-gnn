# Refactoring-Roadmap (Phase 2+)

Umbauten, die den Code deutlich verbessern würden, aber **Verhalten ändern könnten** und daher
nicht im konservativen Import-Refactoring enthalten sind. Reihenfolge = empfohlene Priorität.
Vor jedem Schritt: Referenzlauf auf kleinem Datensatz als Regressionstest.

## 1. Monitor-Familie konsolidieren (geringes Risiko, großer Gewinn)

Die 6 `monitor_*.py` sind zu ~90 % identisch; alle Unterschiede sind Konfiguration
(Log-/Doku-Pfad, pgrep-Muster, Epochenzahl, Poll-Intervall, finalize ja/nein, Abbruchmodus).
→ Ein `monitor.py` mit argparse (`--model {gcn,gatv2} --variant {medium,coarse,fine}`),
die 6 Dateien entfallen. Behebt nebenbei Issues #1 und #2 zentral.
Achtung: `training/runs/run_fine_after_coarse.sh` (Z. 53) startet `monitor_gcn_fine.py` — anpassen.
`monitor_training.sh` ist eine funktional überholte Bash-Kopie von `monitor_gcn.py` → entfernen.

## 2. Gemeinsames Trainings-Modul extrahieren

In allen 8 Trainern byte-identisch dupliziert: `set_seed`, `setup_file_logger`, `log_and_print`,
`log_only`, `console_live`, `download_from_s3`, `download_dataset_from_s3`, `load_dataset`,
`compute_normalization_stats`, `normalize_dataset`, `MLP`, `evaluate_detailed`, `save_checkpoint`.
→ `training/common.py`; die Trainer werden zu dünnen Architektur-Dateien.
Danach: Gen A/Gen B zusammenführen — Gen A-Resume (last_checkpoint, RNG-Restore, stop_reason)
überall + der Gen-B-Fix aus `legacy/trainGATv2_efficient_genB.py` (Early-Stopping-Zähler wird
beim Resume aus der val_loss-History rekonstruiert statt die History abzuschneiden).

## 3. God-Functions aufteilen

| Funktion | Länge | Schnittlinien |
|---|---|---|
| `train*.main()` | 431–663 Z. | Setup / Datenladen / Modellbau / Resume / Trainloop / Eval / Save |
| `runSimulation.run_case()` | ~433 Z. | 11 Schritte → Schrittliste + Runner; `try/finally` für `os.chdir`-Cleanup (ersetzt 11 kopierte except-Blöcke) |
| `createGraphDataset.main()` | ~295 Z. | Download / Verarbeitung / Split / Upload / Report |
| `simulation/main.py:main()` | ~315 Z. | Parallel-/Sequenziell-Block teilen sich ~40 Z. Upload-Logik → eine Funktion |
| `predict.run_eval()` | ~167 Z. | Inferenz / Metriken / VTU-Export / Reporting |

## 4. OpenFOAM-Templates deduplizieren

`controlDict`-Heredoc existiert 3× in `runSimulation.py` (~95 % identisch) + 1× in
`benchmarks/laptop/run_solver_v2.py` → eine Template-Funktion mit Parametern
(endTime, writeInterval, purgeWrite, functions-Block).

## 5. Evaluation-Utilities bündeln (`evaluation/common.py`)

- 6-Feld-Stack `[Ux,Uy,Uz,p,k,epsilon]` aus VTU: **7 Kopien**
- R²: **5 unabhängige Implementierungen** (predict.compute_metrics, permutation_importance,
  r2_comprehensive ×2, r2_without_k_epsilon) → eine, gegeneinander verifiziert
- `write_eval_yaml`/`run_eval`-Subprozess-Wrapper: 3 Kopien
- `NETWORKS`/`CHECKPOINTS`-Listen: 5 Kopien → eine `models.yaml`
- IDW-Subprozessaufruf: 5 Kopien
- Eval-Verzeichnis-Konvention vereinheitlichen (Issue #10) + Metrik-JSON-Schema (Issue #11)

## 6. Kleinere Konsolidierungen

- `extract_gt_minmax.py` / `extract_pred_minmax.py` (~80 % identisch) → ein Modul, Loader injizierbar
- `compare_plots.py` / `compare_tables.py`: gemeinsames `load_model_data`/`load_from_config`
- `reconstruct_lr` (LR-Rekonstruktion) doppelt in `plotting/` → utils
- `load_gcn_model`/`load_gat_model` doppelt (predict_single, export_to_paraview) → utils
- `permutation_importance.py`: Modellklassen aus `predict.py` importieren statt kopieren (behebt #7 strukturell)

## 7. Infrastruktur & Portabilität

- Pfad-Konfiguration: absolute `/home/…`-Pfade in `block_c_gaps`, `extrapolation_no_cellvol`,
  `plotting/*`, allen `.sh` → argparse/ENV/eine zentrale `paths.yaml`
- Bucket aus Config statt hartkodiert (`subsample_extrapolation.py`)
- Entscheidung: durchgängig `aws`-CLI **oder** boto3 (aktuell gemischt: CLI in simulation/training, boto3 in 3 evaluation-Skripten)
- `torch.load(weights_only=True)` wo möglich (Issue #15)
- Shell: `set -euo pipefail`-Audit, Python-Heredocs → eigene .py-Dateien
- `logging` statt `print` in Bibliotheks-Code

## 8. Tests (aktuell existieren keine echten)

Die `test_*.py` unter `simulation/local_test/` sind lokale Skript-Varianten, keine Unit-Tests.
Sinnvoller Einstieg (pytest):
- `createGeometry.create_building`: Vertex-/Normalen-Invarianten, Bounding-Box bei Rotation
- `adaptive_subsample`: Raten pro Zone, Determinismus bei Seed
- `compute_gci` (meshStudy): gegen Celik-et-al.-Referenzbeispiel
- R²/rL2-Implementierungen: Übereinstimmung auf Zufallsdaten
- `reconstruct_lr`: gegen echte training_history.json
- `parse_epoch_line` (Monitor): Log-Zeilen-Fixtures

## 9. Typisierung & Stil

Type-Hints flächendeckend (Vorbild: `interpolate_to_full_mesh.py`, `combine_minmax.py`);
Python ≥ 3.10 ist durch PEP-604-Hints ohnehin Voraussetzung. Danach `ruff` als Linter einführen.
