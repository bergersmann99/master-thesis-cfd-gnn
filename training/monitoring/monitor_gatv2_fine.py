"""
Überwacht das GATv2 Fine Training und dokumentiert alle 50 Epochen einen Zwischenstand.
"""
import re
import time
import subprocess
from pathlib import Path

LOG_FILE     = Path("/home/tbergermann/Python/GAT/output_gatv2_fine.log")
DOC_FILE     = Path("/home/tbergermann/Python/logs/GATv2/training_GATv2_fine.md")
TOTAL_EPOCHS = 800
INTERVAL     = 50    # alle N Epochen dokumentieren
POLL_SEC     = 60  # Fine-Graphen brauchen länger → 60s Polling

logged_milestones = set()

def is_training_running():
    """Prüft per pgrep, ob der Trainingsprozess noch läuft."""
    r = subprocess.run(["pgrep", "-f", "output_gatv2_fine"], capture_output=True)
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
        print(f"[Monitor GATv2 Fine] Epoche {entry['epoch']} dokumentiert.")

def update_lr_reductions_section(doc_path, log_text):
    """Aktualisiert den LR-Reduktionen Abschnitt."""
    reductions = re.findall(r"Lernrate reduziert: ([\d.e+-]+) -> ([\d.e+-]+)", log_text)
    if not reductions:
        return
    content = doc_path.read_text()
    lines = [f"- {old} → {new}" for old, new in reductions]
    block = "\n".join(lines)
    content = re.sub(
        r"\*\(werden eingetragen sobald sie auftreten\)\*",
        block, content, count=1
    )
    doc_path.write_text(content)

def finalize_results(doc_path, log_text):
    """Trägt Test-Metriken nach Trainingsabschluss ein."""
    fields = ["Ux", "Uy", "Uz", "p", "k", "epsilon"]
    gesamt_pattern = r"GESAMT\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([\d.]+)"

    time_m   = re.search(r"Trainingszeit:\s+([\d]+min [\d]+s)", log_text)
    params_m = re.search(r"Parameter:\s+([\d,]+)", log_text)
    best_val_m = re.search(r"Bester Val-Loss:\s+([\d.]+)", log_text)
    best_ep_m  = re.search(r"Bestes Modell geladen \(Epoche (\d+)\)", log_text)

    content = doc_path.read_text()

    test_block_m = re.search(r"--- Test ---.*?(?=\n\n|\Z)", log_text, re.DOTALL)
    if test_block_m:
        test_block = test_block_m.group(0)
        gesamt_m = re.search(gesamt_pattern, test_block)
        if gesamt_m:
            r2_total  = gesamt_m.group(4)
            rl2_total = gesamt_m.group(5)
            for field in fields:
                fm = re.search(field + r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([\d.]+)", test_block)
                if fm:
                    display = "ε" if field == "epsilon" else field
                    old = f"| {display} | — | — | — | — | — |"
                    new = f"| {display} | {fm.group(1)} | {fm.group(2)} | {fm.group(3)} | {fm.group(4)} | {fm.group(5)} |"
                    content = content.replace(old, new)
            old_g = "| **GESAMT** | — | — | — | — | — |"
            new_g = f"| **GESAMT** | **{gesamt_m.group(1)}** | **{gesamt_m.group(2)}** | **{gesamt_m.group(3)}** | **{r2_total}** | **{rl2_total}** |"
            content = content.replace(old_g, new_g)

    replacements = {
        "| Trainierbare Parameter | — |": f"| Trainierbare Parameter | {params_m.group(1) if params_m else '—'} |",
        "| Beste Val-Loss (Epoche) | — |": f"| Beste Val-Loss (Epoche) | {best_val_m.group(1) if best_val_m else '—'} (Ep. {best_ep_m.group(1) if best_ep_m else '—'}) |",
        "| Trainingszeit | — |": f"| Trainingszeit | {time_m.group(1) if time_m else '—'} |",
    }
    for old, new in replacements.items():
        content = content.replace(old, new)

    if test_block_m and gesamt_m:
        content = content.replace("| Test R² (gesamt) | — |",  f"| Test R² (gesamt) | **{r2_total}** |")
        content = content.replace("| Test rL2 (gesamt) | — |", f"| Test rL2 (gesamt) | **{rl2_total}** |")

    doc_path.write_text(content)
    print("[Monitor GATv2 Fine] Ergebnisse eingetragen.")

def main():
    """Hauptschleife: pollt das Trainings-Log, dokumentiert Meilenstein-Epochen und trägt nach Trainingsende die Ergebnisse ein."""
    print("[Monitor GATv2 Fine] Gestartet. Warte auf Log-Datei...")
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

        if not is_training_running():
            print("[Monitor GATv2 Fine] Training abgeschlossen.")
            if current_epoch not in logged_milestones:
                append_table_row(DOC_FILE, epoch_data[current_epoch], lr_reductions)
            update_lr_reductions_section(DOC_FILE, log_text)
            finalize_results(DOC_FILE, log_text)
            break

    print("[Monitor GATv2 Fine] Beendet.")

if __name__ == "__main__":
    main()
