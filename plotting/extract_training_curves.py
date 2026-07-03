"""
extract_training_curves.py
===========================
Extrahiert die Trainingsverlaeufe der sechs finalen Trainingslaeufe und
schreibt sie in EINE CSV im Long-Format (run, epoch, train_loss, val_loss,
lr, is_best). Es wird KEIN Diagramm erzeugt; das Plotten geschieht spaeter
an anderer Stelle (plot_training_curves.py).

Der siebte Lauf (GATv2 Medium no_cellvol) ist bewusst AUSGESCHLOSSEN, weil
seine vollstaendige Trainingskurve nicht rekonstruierbar ist (s. u.).

Datenlage (verifiziert, siehe Konsolen-Report unten)
----------------------------------------------------
Vier Laeufe haben eine vollstaendige training.log (eine Zeile je Epoche,
Epoche 1..N) und werden direkt aus dem Log geparst (LR ist dort echt):
    gcn_coarse, gcn_medium, gcn_bf_25, gatv2_coarse

Drei GATv2-Laeufe wurden fortgesetzt; ihre training.log enthaelt nur den
Resume-Schwanz, nicht alle Epochen:
    gatv2_medium            : Log 2994..3096,  volle Historie 3096 Epochen
    gatv2_bf_25             : Log 1145..1471,  volle Historie 1527 Epochen
    gatv2_medium_no_cellvol : Log/Hist nur 1753..1852 (100 Ep.) -> AUSGESCHLOSSEN

Behandlung der fortgesetzten Laeufe:
  * gatv2_medium / gatv2_bf_25: train_loss und val_loss stammen aus der
    (kumulativ vollstaendigen) training_history.json. Die GATv2-Historie
    enthaelt KEINE Lernrate, daher wird die LR deterministisch aus dem
    val_loss-Verlauf rekonstruiert (ReduceLROnPlateau, mode=min, factor=0.5,
    patience=20, init=1e-4, min=1e-6, threshold=1e-4 rel — exakt die
    Trainingskonfiguration aus trainGATv2_efficient.py). Die Rekonstruktion
    ist an gatv2_coarse, wo die echte LR fuer alle 1015 Epochen im Log steht,
    auf alle LR-Stufenwechsel exakt validiert. Wo der Log echte LR-Werte
    liefert (Schwanz), werden diese den rekonstruierten vorgezogen.
  * gatv2_medium_no_cellvol: NICHT in der CSV. Frueheres Training ging durch
    Abstuerze (SIGTERM) verloren, die Historie wurde beim letzten Resume auf
    100 Epochen ueberschrieben; verfuegbar waeren nur die Epochen 1753..1852.
    Eine vollstaendige, mit den anderen Laeufen vergleichbare Trainingskurve
    ist damit nicht rekonstruierbar -> der Lauf wird aus der Kurven-CSV
    ausgeschlossen. Modell und Test-/Extrapolationsmetriken sind unberuehrt.

is_best markiert je Lauf die Epoche mit minimalem val_loss (bei Gleichstand
die fruehere), berechnet ueber alle ausgegebenen Epochen des Laufs.

Verwendung
----------
    python extract_training_curves.py
Schreibt /home/tbergermann/results/training_curves_all.csv und gibt eine
Zusammenfassung je Lauf aus.
"""

import csv
import json
import re
from pathlib import Path

# ----------------------------------------------------------------------
# Laufkonfiguration (Pfade anhand test_metrics.json verifiziert).
# strategy: "log"     -> training.log ist vollstaendig, direkt parsen
#           "history" -> volle Kurve aus training_history.json, LR
#                        rekonstruieren und mit echter Log-LR ueberschreiben
# ----------------------------------------------------------------------

RUNS = [
    ("gcn_coarse", "/home/tbergermann/Python/GNN/output_gcn_coarse_rerun", "log"),
    ("gcn_medium", "/home/tbergermann/Python/GNN/output_gcn_medium_rerun", "log"),
    ("gcn_bf_25", "/home/tbergermann/Python/GNN/output_gcn_bf25_rerun", "log"),
    ("gatv2_coarse", "/home/tbergermann/Python/GAT/output_gatv2_coarse_h128", "log"),
    ("gatv2_medium", "/home/tbergermann/Python/GAT/output_gatv2_medium_h128", "history"),
    ("gatv2_bf_25", "/home/tbergermann/Python/GAT/output_gatv2_bf25_h128", "history"),
    # gatv2_medium_no_cellvol bewusst AUSGESCHLOSSEN: vollstaendige Trainings-
    # kurve nicht rekonstruierbar (frueher Verlauf durch Crash-Resume verloren,
    # nur Epochen 1753..1852 erhalten). Siehe Modul-Docstring.
]

OUTPUT_CSV = Path("/home/tbergermann/results/training_curves_all.csv")

# Trifft gezielt die Daten-Zeilen "Epoche N/M | Train: .. | Val: .. | LR: ..".
EPOCH_RE = re.compile(
    r"Epoche\s+(\d+)\s*/\s*\d+\s*\|\s*"
    r"Train:\s*([0-9.eE+-]+)\s*\|\s*"
    r"Val:\s*([0-9.eE+-]+)\s*\|\s*"
    r"LR:\s*([0-9.eE+-]+)"
)


def parse_log(log_path):
    """Parst eine training.log. Bei Resume (Epoche mehrfach) gilt der letzte
    Eintrag. Liefert dict epoch -> (train, val, lr), aufsteigend nach epoch."""
    by_epoch = {}
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        m = EPOCH_RE.search(line)
        if m:
            by_epoch[int(m.group(1))] = (
                float(m.group(2)),
                float(m.group(3)),
                float(m.group(4)),
            )
    return by_epoch


def reconstruct_lr(val_loss, init_lr=1e-4, factor=0.5, patience=20,
                   min_lr=1e-6, threshold=1e-4):
    """Deterministische Simulation von torch.optim.lr_scheduler.
    ReduceLROnPlateau (mode='min', threshold_mode='rel', cooldown=0).

    Die LR gilt WAEHREND der Epoche; scheduler.step(val_loss) wird danach
    aufgerufen. Verbesserungskriterium wie in PyTorch: v < best*(1-threshold)
    (mit best=inf zu Beginn -> erste Epoche ist immer Verbesserung).
    """
    lr = []
    current = float(init_lr)
    best = float("inf")
    bad = 0
    for v in val_loss:
        lr.append(current)
        v = float(v)
        if v < best * (1.0 - threshold):
            best = v
            bad = 0
        else:
            bad += 1
            if bad > patience:
                new = max(current * factor, min_lr)
                if current - new > 1e-8:
                    current = new
                bad = 0
    return lr


def build_rows(name, run_dir, strategy):
    """Liefert (rows, note). rows: Liste (epoch, train, val, lr) aufsteigend."""
    run_dir = Path(run_dir)
    if strategy == "log":
        log = parse_log(run_dir / "training.log")
        rows = [(e, *log[e]) for e in sorted(log)]
        note = f"Log vollstaendig, Epochen {rows[0][0]}..{rows[-1][0]}"
        # no_cellvol: Log ist nur der Resume-Schwanz -> kenntlich machen
        if rows[0][0] > 1:
            note = (f"NUR Resume-Schwanz {rows[0][0]}..{rows[-1][0]} verfuegbar "
                    "(fruehe Historie verloren)")
        return rows, note

    # strategy == "history": volle Kurve aus JSON, LR rekonstruiert + echte Log-LR
    hist = json.loads((run_dir / "training_history.json").read_text())
    train = [float(x) for x in hist["train_loss"]]
    val = [float(x) for x in hist["val_loss"]]
    lr = reconstruct_lr(val)
    log = parse_log(run_dir / "training.log")  # echte LR im Schwanz
    n_real = 0
    for e, (_, _, lr_real) in log.items():
        if 1 <= e <= len(lr):
            lr[e - 1] = lr_real
            n_real += 1
    rows = [(i + 1, train[i], val[i], lr[i]) for i in range(len(val))]
    note = (f"train/val aus Historie (Epochen 1..{len(val)}), LR rekonstruiert; "
            f"{n_real} echte Log-LR-Werte im Schwanz uebernommen")
    return rows, note


def main():
    """Schreibt die Kurven-CSV fuer alle Laeufe und gibt eine Zusammenfassung aus."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary = []
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run", "epoch", "train_loss", "val_loss", "lr", "is_best"])
        for name, run_dir, strategy in RUNS:
            rows, note = build_rows(name, run_dir, strategy)
            best_i = min(range(len(rows)), key=lambda i: (rows[i][2], rows[i][0]))
            best_epoch, best_val = rows[best_i][0], rows[best_i][2]
            # is_best auf exakten Werten bestimmt; train/val einheitlich auf
            # 6 Nachkommastellen ausgeben (entspricht der Log-Anzeige).
            for i, (epoch, tr, va, lr) in enumerate(rows):
                writer.writerow([name, epoch, round(tr, 6), round(va, 6), lr, i == best_i])
            summary.append((name, len(rows), rows[0][0], rows[-1][0],
                            best_epoch, best_val, rows[0][3], rows[-1][3], note))

    print(f"CSV geschrieben: {OUTPUT_CSV}\n")
    head = (f"{'run':<26}{'Epochen':>8}{'Ep.-Bereich':>14}{'beste Ep.':>10}"
            f"{'bester Val':>12}{'Start-LR':>11}{'End-LR':>11}")
    print(head)
    print("-" * len(head))
    for name, n, e0, e1, be, bv, lr0, lr1, _ in summary:
        print(f"{name:<26}{n:>8}{f'{e0}..{e1}':>14}{be:>10}"
              f"{bv:>12.6f}{lr0:>11.2e}{lr1:>11.2e}")
    print("\nHinweise je Lauf:")
    for name, *_, note in summary:
        print(f"  {name:<26} {note}")


if __name__ == "__main__":
    main()
