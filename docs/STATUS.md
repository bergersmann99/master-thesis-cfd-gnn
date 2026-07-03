# Projekt-Status (Fortsetzungspunkt)

Stand: 2026-07-03. Dieses Dokument ermöglicht die nahtlose Fortsetzung in einer neuen Session.

## Was ist fertig (Phase 1)

1. **S3-Zugang eingerichtet:** AWS CLI v2 + `aws configure` (IAM-User tbergermann), MCP-Server
   `aws-s3` in `~/.claude/mcp.json` (uvx awslabs.s3-mcp-server).
2. **Original gesichert:** `s3://amzn-master-sim-bucket/skripte_final/` (76 Dateien) →
   `C:\Users\timbe\masterarbeit-original\skripte_final\` (Read-only-Referenz für Diffs).
3. **Repo aufgebaut:** `C:\Users\timbe\master-thesis-cfd-gnn\` (git, 3 Commits):
   - `8f4dd0a`/`e944851` Baseline = unveränderte Originale, dedupliziert + strukturiert
   - `bc9edd9` konservatives Refactoring + Doku
4. **Analyse komplett:** Inventar (docs/INVENTORY.md), 18+ Bugs/Issues (docs/KNOWN_ISSUES.md),
   Konsolidierungsplan (docs/REFACTORING_ROADMAP.md), Deployment-Mapping (docs/DEPLOYMENT.md).
5. **Refactoring (verhaltensneutral, alle 53 py kompilieren):** tote Imports/toter Code raus,
   Docstrings, `with open`/`encoding`, f-String-Fixes, `__main__`-Guards in
   `batch_interpolate_val.py` + `r2_comprehensive.py`. Große Dateien (train*, runSimulation,
   createGraphDataset, main.py) bewusst UNVERÄNDERT (Hochrisiko ohne Tests → Roadmap).
   Jede Änderung per `git diff e944851..bc9edd9` nachvollziehbar.

## Offene Punkte (Phase 2)

1. **GitHub:** privates Repo anlegen (User: bergersmann99, gh CLI eingerichtet) + push.
   `git remote add origin …` fehlt noch.
2. Optional (Roadmap Prio 1–2): Monitor-Konsolidierung, gemeinsames Trainings-Modul.
3. Optional: Bugfixes aus KNOWN_ISSUES (bewusst getrennt vom Import-Refactoring).

## Wichtige Fakten für die Fortsetzung

- S3-Bucket: `amzn-master-sim-bucket` (eu-north-1); Skripte unter Prefix `skripte_final/`.
- Kanonische Versionen: `Real/` schlägt `Test/`, `laptop_timing/predict.py` schlägt
  `predictions/predict.py`, `GAT|GNN/*_efficient.py` = Gen A (produktiv) schlägt
  `training/*_efficient.py` = Gen B (→ `training/legacy/*_genB.py`).
- Umbenennungen: Root-`predict.py` → `inference/predict_single.py`;
  `training/train*_efficient.py` → `training/legacy/*_genB.py`.
- Funktionalität darf sich NICHT ändern (Masterarbeits-Ergebnisse hängen an diesem Code).
