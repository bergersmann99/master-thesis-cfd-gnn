# Datei-Inventar

Quelle: `s3://amzn-master-sim-bucket/skripte_final/` (76 Dateien, Stand 2026-07-03).
Im Repo: 63 kanonische Dateien. **Ext** = externe Abhängigkeiten über Stdlib hinaus;
„aws-CLI" = S3 via `subprocess`, „boto3" = S3 via SDK.

## simulation/ — CFD-Datengenerierung (EC2)

| Datei | Zweck | LOC | Ext |
|---|---|---|---|
| main.py | Batch-Orchestrator: config.yaml → N Zufalls-Tasks (U_ref, Winkel) → Geometrie → Simulation (Pool parallel/sequenziell) → VTK-tar.gz nach S3, CSV-Protokoll | 650 | yaml, aws-CLI |
| createGeometry.py | Parametrisches Satteldach-Haus (10 Vertices, 14 Dreiecke) als STL, rotierbar | 168 | numpy, numpy-stl |
| runSimulation.py | 11-Schritt-OpenFOAM-Pipeline (blockMesh → snappyHexMesh parallel → simpleFoam 2-phasig) + AIJ-Konvergenzprüfung, Kräfte-Extraktion; schreibt alle OF-Dictionaries | 1497 | OpenFOAM, MPI |
| createGraphDataset.py | VTK → PyG-Graphen: Wandabstand (KDTree), Node-Types, adaptives Subsampling (5 Level), kNN-Kanten, 14→6 Features, 70/15/15-Split, S3-Transfer | 1324 | numpy, yaml, torch, PyG, scipy, pyvista, aws-CLI |
| createGraphDataset_no_cellvol.py | Ablations-Variante: 13 Features (ohne cell_volume), Level `medium_no_cellvol` | 1325 | dito |
| run_extrapolation.py | Timing der 2 Extrapolationsfälle (Sturm 25 m/s, Schwachwind 1,5 m/s) | 179 | yaml |
| subsample_extrapolation.py | Extrapolations-Graphen in 3 Auflösungen → S3 | 137 | yaml, numpy, torch, aws-CLI |

### simulation/local_test/ — ältere lokale Varianten (ohne AWS)

| Datei | Zweck | LOC | Anmerkung |
|---|---|---|---|
| meshStudy.py | Mesh-Unabhängigkeitsstudie: 3 Verfeinerungen, GCI nach Celik et al. 2008 | 507 | nutzt test_*-Module |
| test_angle0.py | Manueller Smoke-Test des historisch problematischen Falls (angle=0) | 58 | kein Unit-Test |
| test_createGeometry.py | = createGeometry.py (byte-identisch) | 168 | von meshStudy importiert |
| test_runSimulation.py | Älterer Stand von runSimulation (ohne Fehlercode-Fix 11, ohne loc_y-Offset) | 1497 | |
| test_main.py | main.py ohne S3-Integration | 496 | |
| test_createGraphDataset.py | Ältere lokale Baseline: 1 Subsampling-Level, kein S3, kein Report | 906 | |

## training/

| Datei | Zweck | LOC | Generation |
|---|---|---|---|
| gcn/trainGCN.py | Basis-GCN (Kipf & Welling), Encode-Process-Decode, 14→6 | 1188 | Basis |
| gcn/trainGCN_efficient.py | + Gradient-Checkpointing, BF16, robustes Resume, stop_reason/peak_vram | 1559 | **Gen A** (produktiv) |
| gatv2/trainGATv2.py | Basis-GATv2 (Brody et al., Multi-Head-Attention) | 1276 | Basis |
| gatv2/trainGATv2_efficient.py | GATv2-Pendant zu Gen A | 1620 | **Gen A** (produktiv) |
| gatv2/trainGATv2_efficient_no_cellvol.py | Ablation: 13 Features | 1415 | Gen B |
| legacy/trainGCN_efficient_genB.py | Ältere Efficient-Generation (Resume nur best_model.pt) — Original: `training/trainGCN_efficient.py` | 1266 | Gen B |
| legacy/trainGATv2_efficient_genB.py | Gen B mit 14 Features + Early-Stopping-Zähler-Rekonstruktion beim Resume — Original: `training/trainGATv2_efficient.py` | 1420 | Gen B |
| legacy/train_nocellvol_vram_PATCHED.py | = no_cellvol + 4 Zeilen VRAM-Peak-Messung (Analyse 2026-06-12) | 1419 | Gen B |
| gcn/run_gcn_rerun_sequence.sh | Sequenz coarse→medium→bf25 mit Sanity-Checks, setzt Gen A voraus | 202 | |
| gcn/run_gentest_sequence.sh | Generalisierungstest (5/33 Train-Sims), GCN→GATv2 | 190 | |
| runs/*.sh (5) | Trainings-Launcher (building_focus, _25, fine-after-coarse, efficient, no_cellvol) mit S3-Upload | 67–93 | |
| monitoring/monitor_*.py (6) | Trainings-Monitore: Log-Polling → Markdown-Fortschrittstabellen (~90 % dupliziert) | 99–160 | |
| monitoring/monitor_training.sh | Bash-Kopie von monitor_gcn.py (überholt) | 52 | |

## inference/

| Datei | Zweck | LOC | Ext |
|---|---|---|---|
| predict.py | Zentrales Predict/Eval-CLI, Architektur-Autodetect aus Checkpoint, Metriken R²/rL2, VTU/npy-Export. Kanonische Version aus `laptop_timing_skripte/` (mit S3-Fallback für Einzelgraphen); `predictions/predict.py` war identisch bis auf 2 Zeilen | 897 | numpy, yaml, torch, PyG |
| predict_single.py | Einfaches Einzelvorhersage-CLI (GCN/GAT explizit) — Original: Root-`predict.py`, umbenannt wegen Namenskollision | 184 | numpy, torch, pyvista |
| interpolate_to_full_mesh.py | Sparse Vorhersage (~507k Knoten) → volles CFD-Netz (~8M) via IDW/NN (KDTree); von 5 Batch-Skripten aufgerufen | 257 | numpy, torch, pyvista, scipy |
| export_to_paraview.py | GT/Prediction/Fehlerfelder als .vtu für ParaView | 208 | numpy, torch, pyvista, PyG |

## evaluation/

| Datei | Zweck | LOC | Ext |
|---|---|---|---|
| batch_complete.py | Rest-Batch: fehlende Extrapolation + no_cellvol-Vorhersagen, S3-Upload | 225 | boto3, numpy, torch, pyvista |
| batch_extrapolation.py | 6 Netze × 2 Windfälle: eval → IDW → S3 + R²-Tabelle | 151 | boto3, numpy, pyvista |
| batch_interpolate.py | Test-Vorhersagen aufs Voll-Mesh (6 Netze × 3 Sims) | 101 | numpy, pyvista |
| batch_interpolate_val.py | dito für Val-Sim sim_012 | 49 | numpy, pyvista |
| interpolate_sim013_gcn_rerun.py | Einmal-Reparatur sim_013 → S3 | 79 | boto3, numpy, pyvista |
| permutation_importance.py | Permutation Importance der 14 Features (ΔR², Knoten-/Graph-Level) + Plots | 474 | numpy, torch, PyG, matplotlib |
| r2_comprehensive.py | R²-Gesamtauswertung: Val, Extrapolation, Test pro Sim/Feld | 170 | numpy, pyvista |
| r2_without_k_epsilon.py | R² nur Ux,Uy,Uz,p (ingenieurrelevant) vs. alle 6 | 108 | numpy, pyvista |
| block_c_gaps.py | Bericht-Lücken: rL2/R²-Verifikation + feldweises R² (importiert trainGCN_efficient) | 291 | numpy→entfernt, torch, yaml |
| extrapolation_no_cellvol.py | Extrapolations-R² der 13-Feature-Ablation mit bit-exaktem Gate | 322 | numpy, torch, yaml |
| minmax/extract_gt_minmax.py | Min/Max der GT-Felder aus test.pt (3 Level) | 256 | numpy, torch |
| minmax/extract_pred_minmax.py | dito für Vorhersage-VTUs (~80 % identisch mit gt-Variante) | 208 | numpy, pyvista |
| minmax/combine_minmax.py | CSVs → kombinierte Markdown-Tabelle (sauberstes Skript des Sets) | 110 | Stdlib |
| comparison/compare_plots.py | R²-Vergleichs-Balkendiagramme (publikationsfertig; 175 Z. bewusst inaktiv) | 516 | numpy, yaml, matplotlib |
| comparison/compare_tables.py | 5 Vergleichstabellen als Markdown + LaTeX | 595 | yaml |

## plotting/

| Datei | Zweck | LOC |
|---|---|---|
| extract_training_curves.py | 6 finale Läufe → Long-CSV (rekonstruiert GATv2-LR deterministisch) | 191 |
| plot_training_curves.py | Pro Stufe: Loss+LR-PNG (GCN vs. GATv2), liest training_history.json direkt | 248 |
| plot_training_curves_combined.py | Alle 6 Läufe in einem Diagramm (PNG+PDF), liest die CSV | 130 |

## benchmarks/

| Datei | Zweck | LOC |
|---|---|---|
| laptop/run_all.sh | Morgen-Lauf: S3-Downloads, Sturm-CFD, Konsolidierung (GNN-Block nach exit 0 unerreichbar) | 248 |
| laptop/stop.sh | Graceful Stop via `stopAt writeNow` (Mittagspause) | 38 |
| laptop/resume_gnn.sh | Abend: Sturm-Resume + komplette GNN-Kette (10×) + S3 | 198 |
| laptop/resume_solver.sh | Abend: nur Schwachwind-Solver fertigrechnen | 89 |
| laptop/run_gnn.sh | Eigenständige GNN-Kette (3×) mit Skip-Logik | 141 |
| laptop/run_mesh_v2.py | Mesh-Schritte 1–6 mit Timing pro Schritt → YAML | 158 |
| laptop/run_solver_v2.py | Solver-Schritte 7–11 mit Timing → YAML | 226 |
| laptop/run_graph_t2.py | Graph-Konstruktion mit Teilschritt-Timing | 178 |
| laptop/consolidate_results.py | Alle Timing-YAMLs + Inferenzzeiten → run_log.yaml (CFD vs. GNN, Speedup) | 225 |
| laptop/configs/*.yaml (2) | Fall-Configs Sturm/Schwachwind (Unterschied nur wind.speed) | 92 |
| server_chain/measure_chain.sh | Inferenzketten-Messung (predict + IDW, 2 Fälle × 3 Reps) → CSV | 56 |

## Nicht übernommen (Duplikate/veraltet)

| Original (S3) | Grund |
|---|---|
| `training/trainGCN.py`, `training/trainGATv2.py` | byte-identisch mit `GNN/`/`GAT/`-Versionen |
| `laptop_timing_skripte/{createGeometry, createGraphDataset, runSimulation, interpolate_to_full_mesh}.py` | byte-identisch mit `simulation/` bzw. `inference/` (siehe DEPLOYMENT.md) |
| `predictions/predict.py` | veraltete Kopie von `laptop_timing_skripte/predict.py` (ohne 2-Zeilen-S3-Fallback) |
| `GNN/trainGCN.py`-Kopie in `training/` etc. | siehe oben |

**Umbenennungen:** Root-`predict.py` → `inference/predict_single.py`; `training/train*_efficient.py` → `training/legacy/*_genB.py` (Originalnamen kollidierten mit den Gen-A-Dateien; keines wird von Skripten referenziert).
