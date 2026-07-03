# Bekannte Probleme

Beim Import ins Repo wurden diese Bugs/Inkonsistenzen gefunden. Auf `master` sind sie
unverändert (Ist-Stand, mit dem die Masterarbeits-Ergebnisse erzeugt wurden).

**Auf diesem Branch (`fixes/known-issues`) sind die Bugs #1–#6 behoben** — je ein Commit pro
Fix, ausschließlich Randfall-/Crash-/Reporting-Korrekturen ohne Einfluss auf berechnete
Ergebnisse. Die Punkte #7 ff. bleiben bewusst offen (Fixes würden Zahlen/Konventionen ändern
oder gehören zur Roadmap).

## Echte Bugs (#1–#6 hier gefixt, #7–#8 offen)

| # | Datei | Problem |
|---|-------|---------|
| 1 | `training/monitoring/monitor_gcn_coarse.py` (finalize_results, ~Z. 105) | Zugriff auf `gesamt_m`/`r2_total`/`rl2_total` außerhalb des `if test_block_m:`-Blocks → **UnboundLocalError**, wenn der Test-Block im Log fehlt, aber die Zeit-Zeile matcht. Die drei Schwester-Monitore machen es korrekt per Short-Circuit. |
| 2 | `training/monitoring/monitor_gatv2.py`, `monitor_gcn.py` | Abbruchbedingung `not running AND epoche >= TOTAL-5` → **Endlosschleife**, wenn das Training per Early-Stopping deutlich vor Epoche `TOTAL-5` endet. Die coarse/fine-Varianten prüfen nur `not running` (robust). |
| 3 | `benchmarks/laptop/stop.sh` (~Z. 27), `resume_gnn.sh` (~Z. 62) | `awk '{print $NF}'` auf `ClockTime = 13 s` liefert das Literal **"s"** statt der Zahl. Folge: `TOTAL=$((MORNING + RESUME))` rechnet mit Müll. `resume_solver.sh` macht es korrekt mit `$(NF-1)`. |
| 4 | `benchmarks/laptop/run_all.sh` (Z. 129) | `exit 0` mitten im Skript → der komplette GNN-Block (Schritte 7–14, Z. 131–248) ist **unerreichbarer toter Code**. Die Kette läuft real über `run_gnn.sh`/`resume_gnn.sh`. |
| 5 | `simulation/createGraphDataset.py` (+ `_no_cellvol`) | Bei aktivem S3-Upload wird `output_dir` per `rmtree` gelöscht, **danach** die Dateigröße der `.pt`-Dateien gemessen → Report zeigt immer `0.0 MB`. |
| 6 | `inference/predict.py` (argparse) | `--export-vtk`/`--export-numpy` mit `action="store_true", default=True` sind **wirkungslos** — nur `--no-vtk`/`--no-numpy` haben Effekt. |
| 7 | `evaluation/permutation_importance.py` (load_model) | `in_dim=14` hartkodiert → **bricht bei `no_cellvol`-Modellen** (13 Features). Außerdem: Plot-Titel „Medium h128" fest; `per_field` wird nur im letzten Repeat gesetzt. Die Modellklassen sind zudem eine vereinfachte Kopie aus `predict.py` (Divergenzrisiko). |
| 8 | `training/gatv2/trainGATv2_efficient_no_cellvol.py` | `GATv2Surrogate.__init__` hat Default `in_dim=14`, obwohl die Variante mit 13 arbeitet (nur der Aufruf setzt 13). Falle bei direkter Klassennutzung. |

## Inkonsistenzen (funktionieren, aber verwirren)

| # | Wo | Problem |
|---|----|---------|
| 9 | `benchmarks/laptop/run_gnn.sh` | Kommentar/Logs sprechen von „10 Läufen", die Schleife läuft `seq 1 3`. `resume_gnn.sh` nutzt 10, `consolidate_results.py` wertet fix `n_runs=3` aus — überzählige Läufe werden ignoriert. |
| 10 | `evaluation/batch_extrapolation.py` vs. `r2_comprehensive.py` | Eval-Verzeichnis heißt einmal `evaluation_{net}_extrap_{cfg}`, einmal `eval_extrap_{net}_{cfg}` — dieselben Läufe landen in unterschiedlichen Ordnern. |
| 11 | `evaluation/r2_without_k_epsilon.py` vs. `batch_extrapolation.py` | JSON-Zugriff einmal `m["gesamt"]["R2"]`, einmal `m["metrics"]["gesamt"]["R2"]` — uneinheitliches Metrik-Schema. |
| 12 | `benchmarks/laptop/run_mesh_v2.py` | `--cores` überschreibt `hardware.num_cores` aus der Config **immer** → der Config-Wert 92 (Cluster-Erbe) ist wirkungslos/irreführend. |
| 13 | `benchmarks/laptop/consolidate_results.py` | `snappyHexMesh`-Zeit fließt sowohl in `cfd_total` als auch in `gnn_total` ein — der ausgewiesene Speedup ist entsprechend zu interpretieren. |
| 14 | `simulation/subsample_extrapolation.py` (Z. 41) | Einziger hartkodierter Bucket-Name im Python-Code (`amzn-master-sim-bucket`); alle anderen Skripte lesen ihn aus `config.yaml`. |
| 14b | `training/monitoring/monitor_{gatv2,gcn}_{coarse,fine}.py` | `TOTAL_EPOCHS` ist in den 4 coarse/fine-Varianten definiert, wird aber nirgends referenziert (tote Konstante — nur die medium-Varianten nutzen sie in ihrer Abbruchbedingung). |

## Sicherheits-/Robustheitshinweise

| # | Wo | Hinweis |
|---|----|---------|
| 15 | alle `train*.py`, `predict.py`, `block_c_gaps.py`, … | `torch.load(..., weights_only=False)` deserialisiert beliebige Pickles — nur eigene Checkpoints laden. |
| 16 | `simulation/main.py` (`_verify_s3_upload`) | `except Exception: pass` verschluckt alle Verifikationsfehler. |
| 17 | `training/gcn/run_*.sh`, `training/runs/*.sh` | `set -u` ohne `set -e`/`pipefail`; Python-in-Bash-Heredocs; absolute `/home/tbergermann/…`-Pfade. |
| 18 | `evaluation/block_c_gaps.py`, `extrapolation_no_cellvol.py`, `plotting/*` | Hartkodierte absolute Serverpfade (`sys.path.insert`, Ausgabepfade) — nur auf dem Zielsystem lauffähig (siehe docs/DEPLOYMENT.md). |
