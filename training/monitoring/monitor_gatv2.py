"""
Überwacht das GATv2 Training und dokumentiert alle 50 Epochen einen Zwischenstand.
"""
import re
import time
import subprocess
from pathlib import Path

LOG_FILE     = Path("/home/tbergermann/Python/GAT/output_gatv2_medium.log")
DOC_FILE     = Path("/home/tbergermann/Python/logs/GATv2/training_GATv2_medium.md")
TOTAL_EPOCHS = 500
INTERVAL     = 50    # alle N Epochen dokumentieren
POLL_SEC     = 30    # alle 30s Log prüfen

logged_milestones = set([0])  # Epoche 0 gilt als bereits dokumentiert

def is_training_running():
    """Prüft per pgrep, ob der Trainingsprozess noch läuft."""
    r = subprocess.run(["pgrep", "-f", "trainGATv2.py"], capture_output=True)
    return r.returncode == 0

def parse_epoch_line(line):
    """Extrahiert Epoche, Train, Val, Best, LR, Zeit aus einer Fortschrittszeile."""
    m = re.search(r'\[(\d+)/\d+\]\s+T:([\d.]+)\s+V:([\d.]+)\s+best:([\d.]+)\s+LR:([\d.e+-]+)\s+\|\s+([\dm\s]+)', line)
    if m:
        return {
            "epoch": int(m.group(1)),
            "train": float(m.group(2)),
            "val":   float(m.group(3)),
            "best":  float(m.group(4)),
            "lr":    m.group(5),
            "time":  m.group(6).strip(),
        }
    return None

def count_lr_reductions(log_text):
    """Zählt die im Log vermerkten LR-Reduktionen."""
    return log_text.count("Lernrate reduziert")

def append_table_row(doc_path, entry, lr_reductions_total):
    """Hängt eine Zeile in die Trainingstabelle im Markdown ein."""
    content = doc_path.read_text()
    note = f"Best: {entry['best']:.5f}"
    if lr_reductions_total > 0:
        note += f", LR-Red.: {lr_reductions_total}×"
    new_row = f"| {entry['epoch']} | {entry['train']:.5f} | {entry['val']:.5f} | {entry['lr']} | {note}, Laufzeit: {entry['time']} |"
    marker = "### LR-Reduktionen"
    if marker in content:
        content = content.replace(marker, new_row + "\n\n" + marker)
        doc_path.write_text(content)
        print(f"[Monitor GAT] Epoche {entry['epoch']} dokumentiert.")

def update_lr_reductions_section(doc_path, log_text):
    """Aktualisiert den LR-Reduktionen Abschnitt."""
    reductions = re.findall(r"Lernrate reduziert: ([\d.e+-]+) -> ([\d.e+-]+)", log_text)
    if not reductions:
        return
    content = doc_path.read_text()
    lines = [f"- {old} → {new}" for old, new in reductions]
    block = "\n".join(lines)
    content = re.sub(r"\*\(werden eingetragen sobald sie auftreten\)\*", block, content, count=1)
    doc_path.write_text(content)

def main():
    """Hauptschleife: pollt das Trainings-Log und dokumentiert Meilenstein-Epochen."""
    print("[Monitor GAT] Gestartet.")
    last_milestone = 0

    while True:
        time.sleep(POLL_SEC)
        if not LOG_FILE.exists():
            continue

        log_text = LOG_FILE.read_text()
        lines = log_text.splitlines()

        epoch_data = {}
        for line in lines:
            parsed = parse_epoch_line(line)
            if parsed:
                epoch_data[parsed["epoch"]] = parsed

        if not epoch_data:
            continue

        current_epoch = max(epoch_data.keys())
        lr_reductions = count_lr_reductions(log_text)

        next_milestone = last_milestone + INTERVAL
        while next_milestone <= current_epoch:
            if next_milestone in epoch_data and next_milestone not in logged_milestones:
                append_table_row(DOC_FILE, epoch_data[next_milestone], lr_reductions)
                update_lr_reductions_section(DOC_FILE, log_text)
                logged_milestones.add(next_milestone)
                last_milestone = next_milestone
            next_milestone += INTERVAL

        # Abbruch, sobald der Trainingsprozess beendet ist (wie coarse/fine-Monitore).
        # Die fruehere Zusatzbedingung `current_epoch >= TOTAL_EPOCHS - 5` liess den
        # Monitor bei Early-Stopping vor dieser Schwelle endlos weiterlaufen.
        if not is_training_running():
            print("[Monitor GAT] Training abgeschlossen.")
            break

    print("[Monitor GAT] Beendet.")

if __name__ == "__main__":
    main()
