"""
Überwacht das GCN Training und dokumentiert alle 50 Epochen einen Zwischenstand.
"""
import re
import time
import subprocess
from pathlib import Path

LOG_FILE   = Path("/home/tbergermann/Python/GNN/output_gcn_medium.log")
DOC_FILE   = Path("/home/tbergermann/Python/logs/GCN/training_GCN_medium.md")
TOTAL_EPOCHS = 500
INTERVAL     = 50    # alle N Epochen dokumentieren
POLL_SEC     = 30    # alle 30s Log prüfen

logged_milestones = set([0, 1, 2])  # schon manuell eingetragen

def is_training_running():
    r = subprocess.run(["pgrep", "-f", "trainGCN.py"], capture_output=True)
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
    return log_text.count("Lernrate reduziert")

def append_table_row(doc_path, entry, lr_reductions_total):
    """Hängt eine Zeile in die Trainingstabelle im Markdown ein."""
    content = doc_path.read_text()

    note = f"Best: {entry['best']:.5f}"
    if lr_reductions_total > 0:
        note += f", LR-Red.: {lr_reductions_total}×"

    new_row = f"| {entry['epoch']} | {entry['train']:.5f} | {entry['val']:.5f} | {entry['lr']} | {note}, Laufzeit: {entry['time']} |"

    # Einfügen vor der LR-Reduktionen-Sektion
    marker = "### LR-Reduktionen"
    if marker in content:
        content = content.replace(marker, new_row + "\n\n" + marker)
        doc_path.write_text(content)
        print(f"[Monitor] Epoche {entry['epoch']} dokumentiert.")
    else:
        print(f"[Monitor] Marker nicht gefunden, Zeile wird angehängt.")

def update_lr_reductions_section(doc_path, log_text):
    """Aktualisiert den LR-Reduktionen Abschnitt."""
    reductions = re.findall(r"Lernrate reduziert: ([\d.e+-]+) -> ([\d.e+-]+)", log_text)
    if not reductions:
        return
    content = doc_path.read_text()
    lines = [f"- {old} → {new}" for old, new in reductions]
    block = "\n".join(lines)
    # Ersetze Platzhalter
    content = re.sub(
        r"\*\(werden eingetragen sobald sie auftreten\)\*",
        block,
        content,
        count=1
    )
    doc_path.write_text(content)

def main():
    print("[Monitor] Gestartet. Prüfe alle 30s...")
    last_milestone = max(logged_milestones)

    while True:
        time.sleep(POLL_SEC)

        if not LOG_FILE.exists():
            continue

        log_text = LOG_FILE.read_text()
        lines = log_text.splitlines()

        # Alle Epochen-Zeilen finden
        epoch_data = {}
        for line in lines:
            parsed = parse_epoch_line(line)
            if parsed:
                epoch_data[parsed["epoch"]] = parsed

        if not epoch_data:
            continue

        current_epoch = max(epoch_data.keys())
        lr_reductions = count_lr_reductions(log_text)

        # Nächsten Meilenstein prüfen
        next_milestone = last_milestone + INTERVAL
        while next_milestone <= current_epoch:
            if next_milestone in epoch_data and next_milestone not in logged_milestones:
                append_table_row(DOC_FILE, epoch_data[next_milestone], lr_reductions)
                update_lr_reductions_section(DOC_FILE, log_text)
                logged_milestones.add(next_milestone)
                last_milestone = next_milestone
            next_milestone += INTERVAL

        # Training beendet?
        if not is_training_running() and current_epoch >= TOTAL_EPOCHS - 5:
            print("[Monitor] Training abgeschlossen.")
            # Finalen Stand dokumentieren falls noch nicht geschehen
            if current_epoch not in logged_milestones:
                append_table_row(DOC_FILE, epoch_data[current_epoch], lr_reductions)
                update_lr_reductions_section(DOC_FILE, log_text)
            break

    print("[Monitor] Beendet.")

if __name__ == "__main__":
    main()
