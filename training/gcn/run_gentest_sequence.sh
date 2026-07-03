#!/bin/bash
# ======================================================================
# run_gentest_sequence.sh
# Generalisierungstest auf Medium-Stufe mit reduziertem Train-Set:
# 5 von 33 Trainings-Simulationen, deterministisch gewaehlt
# (numpy default_rng(42).choice(sorted_sim_ids, 5, replace=False)).
# Validierungs-Split unveraendert.
# Sequenziell: GCN dann GATv2. tmux-faehig.
# ======================================================================
set -u

VENV="/home/tbergermann/Python/venv/bin/python"
GCN_DIR="/home/tbergermann/Python/GNN"
GAT_DIR="/home/tbergermann/Python/GAT"
GCN_SCRIPT="$GCN_DIR/trainGCN_efficient.py"
GAT_SCRIPT="$GAT_DIR/trainGATv2_efficient.py"
MASTER_LOG="$GCN_DIR/run_gentest_master.log"

MEDIUM_SRC_DIR="$GCN_DIR/graph_dataset_medium_rerun"
GENTEST_DATA_DIR="$GCN_DIR/graph_dataset_medium_gentest_rerun"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

build_gentest_dataset() {
    log "Baue Gentest-Dataset (5 von 33 Train-Sims, seed=42 deterministisch)"
    mkdir -p "$GENTEST_DATA_DIR"
    if [ ! -f "$MEDIUM_SRC_DIR/train.pt" ]; then
        log "FEHLER: Medium-Trainingsdaten nicht gefunden: $MEDIUM_SRC_DIR/train.pt"
        log "  Bitte erst Medium-Hauptlauf durchfuehren."
        return 2
    fi
    $VENV - "$MEDIUM_SRC_DIR" "$GENTEST_DATA_DIR" <<'PYEOF' >>"$MASTER_LOG" 2>&1
import sys, os, torch
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
train = torch.load(os.path.join(src, "train.pt"), weights_only=False)
val = torch.load(os.path.join(src, "val.pt"), weights_only=False)
test = torch.load(os.path.join(src, "test.pt"), weights_only=False)
sim_ids = [g.sim_id for g in train]
sorted_ids = sorted(sim_ids)
print(f"  Train-Sims (sortiert, n={len(sorted_ids)}): {sorted_ids}")
rng = np.random.default_rng(42)
chosen = list(rng.choice(sorted_ids, size=5, replace=False))
chosen = [str(c) for c in chosen]
print(f"  Gewaehlte 5 Train-Sims (np.random.default_rng(42).choice(...)): {chosen}")
selected = [g for g in train if g.sim_id in chosen]
assert len(selected) == 5, f"Erwartet 5 Graphen, gefunden {len(selected)}"
torch.save(selected, os.path.join(dst, "train.pt"))
torch.save(val, os.path.join(dst, "val.pt"))
torch.save(test, os.path.join(dst, "test.pt"))
# Metadata schreiben
if os.path.exists(os.path.join(src, "metadata.yaml")):
    import shutil
    shutil.copy(os.path.join(src, "metadata.yaml"), os.path.join(dst, "metadata.yaml"))
val_sim_ids = sorted(g.sim_id for g in val)
print(f"  Validation-Sims (n={len(val_sim_ids)}): {val_sim_ids}")
total_nodes = sum(g.x.size(0) for g in selected)
print(f"  Train-Set: 5 Graphen, {total_nodes:,} Knoten, "
      f"{sum(g.edge_index.size(1) for g in selected):,} Kanten")
print(f"  Val-Set: {len(val)} Graphen unveraendert")
print(f"  Test-Set: {len(test)} Graphen unveraendert")
print(f"  Gespeichert nach {dst}")
# Sanity
assert selected[0].x.size(1) == 14, f"feat_dim={selected[0].x.size(1)} (erwartet 14)"
print("  Feature-Dim check: 14 OK")
PYEOF
    return $?
}

run_gentest() {
    local model=$1
    local script
    local out_dir
    local stdout_log
    if [ "$model" = "gcn" ]; then
        script="$GCN_SCRIPT"
        out_dir="$GCN_DIR/output_gcn_medium_gentest_rerun"
        stdout_log="$GCN_DIR/output_gcn_medium_gentest_rerun.log"
    else
        script="$GAT_SCRIPT"
        out_dir="$GAT_DIR/output_gatv2_medium_gentest_rerun"
        stdout_log="$GAT_DIR/output_gatv2_medium_gentest_rerun.log"
    fi
    log "Starte Generalisierungstest $model (output: $out_dir, stdout: $stdout_log)"
    cd "$(dirname "$script")"
    $VENV "$script" \
        --data-dir "$GENTEST_DATA_DIR" \
        --output-dir "$out_dir" \
        --epochs 1500 --patience 100 --batch-size 1 --seed 42 \
        --hidden-dim 128 --num-layers 10 --dropout 0.0 \
        --lr 1e-4 --min-lr 1e-6 \
        --gradient-checkpointing --mixed-precision \
        >"$stdout_log" 2>&1
    local rc=$?
    log "Gentest $model beendet (exit $rc)"
    return $rc
}

check_stop_reason() {
    local model=$1
    local out_dir
    if [ "$model" = "gcn" ]; then
        out_dir="$GCN_DIR/output_gcn_medium_gentest_rerun"
    else
        out_dir="$GAT_DIR/output_gatv2_medium_gentest_rerun"
    fi
    local hist="$out_dir/training_history.json"
    if [ ! -f "$hist" ]; then
        log "FEHLER: $hist existiert nicht."
        return 2
    fi
    $VENV - "$hist" "$model" <<'PYEOF' | tee -a "$MASTER_LOG"
import sys, json
hist, name = sys.argv[1], sys.argv[2]
h = json.load(open(hist))
sr = h["stop_reason"]
be = h["best_epoch"]
bv = h["best_val_loss"]
te = len(h["train_loss"])
vram_gb = h["peak_vram_bytes"] / (1024**3)
time_h = h["training_time_s"] / 3600
print(f"  gentest_{name}: stop_reason={sr} best_epoch={be} best_val={bv:.6f} "
      f"total_ep={te} peak_vram={vram_gb:.2f} GB time={time_h:.2f} h")
# Train/Val Divergenz prüfen: Vergleich der letzten 20 Epochen
tl = h["train_loss"][-20:]
vl = h["val_loss"][-20:]
import statistics
t_mean = statistics.mean(tl)
v_mean = statistics.mean(vl)
print(f"  Letzte 20 Ep Mittel: train={t_mean:.4f} val={v_mean:.4f} ratio={v_mean/t_mean:.2f}")
if v_mean > t_mean * 1.5:
    print(f"  HINWEIS: Val-Loss > 1.5x Train-Loss -> deutet auf Ueberanpassung")
elif v_mean > t_mean * 1.2:
    print(f"  HINWEIS: Val-Loss > 1.2x Train-Loss -> moderate Divergenz")
else:
    print(f"  Train/Val gehen gemeinsam runter (keine starke Divergenz).")
if sr == "early_stopping":
    sys.exit(0)
elif sr == "nan":
    sys.exit(10)
elif sr == "max_epochs":
    print(f"  STOP: 1500-Ep-Limit erreicht.")
    sys.exit(11)
else:
    sys.exit(12)
PYEOF
    return $?
}

run_gentest_stage() {
    local model=$1
    log "=========================================="
    log "=== GENTEST $model START ==="
    log "=========================================="
    run_gentest "$model"
    local tr=$?
    if [ $tr -ne 0 ]; then
        log "Training gentest_$model brach mit exit=$tr ab. Stop."
        exit 4
    fi
    check_stop_reason "$model"
    local cs=$?
    if [ $cs -ne 0 ] && [ $cs -ne 11 ]; then
        log "Gentest $model nicht sauber beendet (rc=$cs). Sequenz haelt."
        exit $cs
    fi
    if [ $cs -eq 11 ]; then
        log "Gentest $model hat 1500-Ep-Limit erreicht — Sequenz haelt, manuelle Pruefung."
        exit $cs
    fi
    log "=== GENTEST $model FERTIG ==="
}

log "=========================================="
log "=== GENERALISIERUNGSTEST START ==="
log "=========================================="

build_gentest_dataset || { log "Dataset-Bau fehlgeschlagen. Stop."; exit 2; }
log "Gentest-Dataset gebaut."

run_gentest_stage gcn
run_gentest_stage gatv2

log "=========================================="
log "=== GENERALISIERUNGSTEST KOMPLETT ==="
log "=========================================="
