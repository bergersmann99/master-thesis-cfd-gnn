# CFD-Surrogatmodellierung mit Graph Neural Networks

[![DOI](https://zenodo.org/badge/1288706514.svg)](https://zenodo.org/badge/latestdoi/1288706514)

Code zur Masterarbeit: GNN-basierte Surrogatmodelle (GCN / GATv2) für die Vorhersage
von RANS-Strömungsfeldern um ein parametrisches Satteldach-Gebäude.

Die Pipeline erzeugt CFD-Trainingsdaten mit OpenFOAM (simpleFoam, k-ε, AIJ-Konvergenzkriterien),
wandelt sie in PyTorch-Geometric-Graphen um, trainiert Encode-Process-Decode-Surrogate
(Pfaff et al. 2020) und wertet Genauigkeit (R², rL2) sowie Laufzeit (CFD vs. GNN-Speedup) aus.
Datenaustausch läuft über den S3-Bucket `amzn-master-sim-bucket`.

## Repo-Struktur

```
├── simulation/            CFD-Datengenerierung (EC2-Server)
│   ├── main.py                 Orchestrator: Tasks → Geometrie → OpenFOAM → S3
│   ├── createGeometry.py       Parametrisches Satteldach-Haus als STL
│   ├── runSimulation.py        11-Schritt-OpenFOAM-Pipeline (blockMesh … simpleFoam)
│   ├── createGraphDataset.py   VTK → PyG-Graphen (14 Features, adaptives Subsampling)
│   ├── createGraphDataset_no_cellvol.py   Ablation: 13 Features (ohne cell_volume)
│   ├── run_extrapolation.py    Extrapolationsfälle (Sturm 25 m/s, Schwachwind 1,5 m/s)
│   ├── subsample_extrapolation.py
│   └── local_test/             Ältere lokale Varianten ohne AWS (+ meshStudy, GCI)
├── training/              GNN-Training
│   ├── gcn/                    trainGCN.py (Basis) + trainGCN_efficient.py (Gen A) + Sequenz-Skripte
│   ├── gatv2/                  trainGATv2.py + trainGATv2_efficient.py (Gen A) + no_cellvol (Ablation)
│   ├── legacy/                 Ältere Generation (Gen B) + VRAM-Mess-Patch — nur Referenz
│   ├── runs/                   Shell-Launcher der Trainingsläufe
│   └── monitoring/             Trainings-Monitore (Markdown-Fortschrittstabellen)
├── inference/             Vorhersage & Interpolation
│   ├── predict.py              Zentrales Predict/Eval-CLI (Architektur-Autodetect)
│   ├── predict_single.py       Einfaches Einzelvorhersage-CLI (ehem. Root-predict.py)
│   ├── interpolate_to_full_mesh.py   Sparse Vorhersage → volles CFD-Netz (IDW/NN)
│   └── export_to_paraview.py   VTU-Export für ParaView
├── evaluation/            Auswertung (R², Permutation Importance, Batch-Läufe, Min/Max, Vergleiche)
├── plotting/              Trainingskurven (Extraktion + Publikations-Plots)
├── benchmarks/
│   ├── laptop/                 Timing-Studie CFD vs. GNN auf Laptop (3-Phasen-Workflow)
│   └── server_chain/           Inferenzketten-Messung (Block B)
└── docs/                  DEPLOYMENT, KNOWN_ISSUES, REFACTORING_ROADMAP, INVENTORY
```

## Pipeline

```
config.yaml → simulation/main.py ──STL──▶ runSimulation (OpenFOAM) ──VTK──▶ S3
                                                                             │
              training/*/train*.py ◀──.pt── simulation/createGraphDataset ◀──┘
                        │
                        ▼ best_model.pt (S3)
              inference/predict.py ──npy──▶ interpolate_to_full_mesh ──VTU──▶ evaluation/, ParaView
```

## Wichtige Konventionen

- **Input-Features (14):** x, y, z, wall_distance, cell_volume, node_type (7× One-Hot), U_ref, angle.
  Die `no_cellvol`-Variante nutzt 13 Features (Ablationsstudie).
- **Output-Features (6):** Ux, Uy, Uz, p, k, epsilon.
- **Subsampling-Level:** coarse / medium / fine / building_focus / building_focus_25 (+ medium_no_cellvol).
- **Trainer-Generationen:** Gen A (`training/gcn/`, `training/gatv2/`) = produktiv, mit
  `last_checkpoint.pt`-Resume, RNG-Restore, `stop_reason`/`peak_vram`-Logging (wird von den
  Sequenz-Skripten vorausgesetzt). Gen B (`training/legacy/`) = ältere, schlankere Generation.
- **Seeds:** durchgängig 42; Split 70/15/15.

## Ausführung

Die Skripte sind für zwei Zielumgebungen geschrieben (EC2-Server bzw. Laptop) und nutzen
absolute Pfade dieser Umgebungen. Das Mapping Repo ↔ Zielsystem steht in
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Abhängigkeiten: [requirements.txt](requirements.txt)
(Python ≥ 3.10, OpenFOAM + MPI für die Simulation, AWS CLI für S3).

## Zitieren

> Tim Bergermann (2026). *From CFD data sets to real-time forecasting: Neural networks
> compared as surrogate models for flow predictions.* Version 1.0.2.
> [10.5281/zenodo.21242494](https://doi.org/10.5281/zenodo.21242494)

Zitiervorlage auch maschinenlesbar in [CITATION.cff](CITATION.cff) (GitHub zeigt darüber
automatisch einen „Cite this repository"-Button an). Dauerhaft archiviert auf
[Zenodo](https://doi.org/10.5281/zenodo.21242494); die DOI-Badge oben verlinkt stets die
jeweils neueste archivierte Version.

## Lizenz

[PolyForm Noncommercial 1.0.0](LICENSE.md) — Nutzung für **Forschung, Lehre und andere
nicht-kommerzielle Zwecke ist frei** (Weitergabe und abgeleitete Arbeiten eingeschlossen,
solange Lizenz und Copyright-Hinweis erhalten bleiben). **Kommerzielle Nutzung erfordert
eine separate Lizenz** — Anfragen bitte an [@bergersmann99](https://github.com/bergersmann99).

## Herkunft & Verifikation

Importiert aus `s3://amzn-master-sim-bucket/skripte_final/` (Stand 2026-07-03).
Duplikate wurden dedupliziert (byte-identische Kopien, veraltete Stände) — Details in
[docs/INVENTORY.md](docs/INVENTORY.md). Das Refactoring ist bewusst konservativ
(verhaltensneutrale Änderungen, per `git diff` gegen den Baseline-Commit nachvollziehbar);
bekannte Bugs sind unverändert und in [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) dokumentiert,
empfohlene Umbauten in [docs/REFACTORING_ROADMAP.md](docs/REFACTORING_ROADMAP.md).
