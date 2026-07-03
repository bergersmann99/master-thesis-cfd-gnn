# Deployment: Repo ↔ Zielsysteme

Die Skripte referenzieren einander teils über **absolute Pfade der Zielsysteme** (nicht über
Repo-relative Pfade). Das Repo ist die Quellcode-Verwaltung; ausgeführt wird auf zwei Umgebungen.

## Zielsystem 1: Simulations-/Trainingsserver (EC2, User `tbergermann`)

| Server-Pfad | Repo-Quelle |
|---|---|
| `~/Python/GNN/trainGCN.py` | `training/gcn/trainGCN.py` |
| `~/Python/GNN/trainGCN_efficient.py` | `training/gcn/trainGCN_efficient.py` (Gen A) |
| `~/Python/GNN/run_*.sh` | `training/gcn/run_*.sh` |
| `~/Python/GAT/trainGATv2.py` | `training/gatv2/trainGATv2.py` |
| `~/Python/GAT/trainGATv2_efficient.py` | `training/gatv2/trainGATv2_efficient.py` (Gen A) |
| `~/Python/GAT/trainGATv2_efficient_no_cellvol.py` | `training/gatv2/trainGATv2_efficient_no_cellvol.py` |
| `~/Python/*.py` (Root-Auswertung) | `evaluation/block_c_gaps.py`, `evaluation/extrapolation_no_cellvol.py`, `plotting/*.py`, `inference/interpolate_to_full_mesh.py`, `inference/export_to_paraview.py`, `inference/predict_single.py`¹, `training/monitoring/monitor_*.py` |
| `~/Python/predictions/predict.py` | `inference/predict.py`² |
| `~/Python/predictions/*.py` | `evaluation/batch_*.py`, `r2_*.py`, `permutation_importance.py`, `interpolate_sim013_gcn_rerun.py` |
| `~/Python/minmax_subsampling/*.py` | `evaluation/minmax/*.py` |
| Simulationsordner (OpenFOAM-Host) | `simulation/*.py` + `config.yaml` (nicht im Repo — enthält Lauf-Parameter) |
| dito, lokale Testumgebung | `simulation/local_test/*.py` + `test_config.yaml` |

¹ `predict_single.py` hieß ursprünglich `~/Python/predict.py` — umbenannt, weil im Repo sonst
zwei `predict.py` in `inference/` kollidieren. Beim Deploy unter beliebigem Namen ablegen;
es wird von keinem Skript referenziert (nur manuell aufgerufen).

² Von `evaluation/batch_*.py` und `r2_comprehensive.py` als
`/home/tbergermann/Python/predictions/predict.py` aufgerufen, von `measure_chain.sh` ebenso —
**muss** beim Deploy an diesen Ort.

**Wichtige Querverweise (dürfen beim Deploy nicht brechen):**
- `evaluation/batch_*.py` → `INTERP = ~/Python/interpolate_to_full_mesh.py`
- `evaluation/block_c_gaps.py` → `sys.path.insert(0, "~/Python/GNN")` (importiert `trainGCN_efficient`)
- `evaluation/extrapolation_no_cellvol.py` → `sys.path.insert(0, "~/Python/GAT")` (importiert `trainGATv2_efficient`)
- `inference/export_to_paraview.py` + `predict_single.py` → erwarten `GNN/` und `GAT/` als
  Unterordner **neben sich** (`sys.path.insert` relativ zu `__file__`)
- `training/runs/run_fine_after_coarse.sh` → startet `monitor_gcn_fine.py`
- Monitor-pgrep-Muster matchen auf die Original-Skriptnamen (`trainGCN.py`, `output_gatv2_coarse`, …)

## Zielsystem 2: Laptop-Timing (User `tim-bergermann`)

Layout `~/laptop_timing/`:

| Laptop-Pfad | Repo-Quelle |
|---|---|
| `~/laptop_timing/run_all.sh`, `stop.sh`, `resume_*.sh`, `run_gnn.sh` | `benchmarks/laptop/*.sh` |
| `~/laptop_timing/scripts/run_{mesh,solver}_v2.py`, `run_graph_t2.py`, `consolidate_results.py` | `benchmarks/laptop/*.py` |
| `~/laptop_timing/scripts/configs/*.yaml` | `benchmarks/laptop/configs/*.yaml` |
| `~/laptop_timing/scripts/createGeometry.py` | `simulation/createGeometry.py` (identisch) |
| `~/laptop_timing/scripts/runSimulation.py` | `simulation/runSimulation.py` (identisch) |
| `~/laptop_timing/scripts/createGraphDataset.py` | `simulation/createGraphDataset.py` (identisch) |
| `~/laptop_timing/scripts/predict.py` | `inference/predict.py` (identisch) |
| `~/laptop_timing/scripts/interpolate_to_full_mesh.py` | `inference/interpolate_to_full_mesh.py` (identisch) |

Die fünf „geteilten Module" existierten im S3-Original als byte-identische Kopien in
`laptop_timing_skripte/` — im Repo gibt es sie **einmal**; beim Deploy dorthin kopieren.
Laufzeit-Ordner (`data/`, `logs/`, `medium/case/`, `graphs/`, `predictions/`, `interpolated/`)
werden von den Skripten selbst angelegt bzw. aus S3 befüllt.

## S3-Bucket `amzn-master-sim-bucket` (eu-north-1)

| Prefix | Inhalt | Erzeugt von |
|---|---|---|
| `skripte_final/` | Code-Snapshot (Quelle dieses Repos) | manuell |
| `vtk-data/` (bzw. Prefix aus config) | CFD-Ergebnisse als `.tar.gz` + CSV | `simulation/main.py` |
| `graph-dataset_<level>/` | `train/val/test.pt` + Metadaten | `simulation/createGraphDataset.py` |
| `graph-dataset_extrapolation/<level>/` | Extrapolations-Graphen | `simulation/subsample_extrapolation.py` |
| `models/<run>/` | `best_model.pt`, `test_metrics.json`, `training_history.json` | `training/runs/*.sh` |
| `predictions/…`, `results/…` | Eval-Ergebnisse, VTUs, Timing-Logs | `evaluation/*`, `benchmarks/*` |

Credentials: `aws configure` (IAM-User); die Skripte nutzen die AWS-CLI (`aws s3 cp/sync/ls`),
drei Evaluation-Skripte zusätzlich `boto3` (nutzt dieselben `~/.aws/credentials`).
