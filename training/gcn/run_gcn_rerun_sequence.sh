#!/bin/bash
# ======================================================================
# run_gcn_rerun_sequence.sh
# Sequenzieller GCN-Nachtrainings-Lauf: coarse -> medium -> bf25
# Identische Hyperparameter: hidden=128, layers=10, lr=1e-4, batch=1,
#                            patience=100, max_epochs=3000, seed=42,
#                            dropout=0.0, grad-ckpt + bf16.
# Sanity-Check vor jedem Lauf (14 Features, plausible Knotenzahl).
# Halt bei NaN oder max_epochs (3000) — sonst nahtloser Uebergang.
# ======================================================================
set -u

VENV="/home/tbergermann/Python/venv/bin/python"
AWS="/home/tbergermann/Python/venv/bin/aws"
GCN_DIR="/home/tbergermann/Python/GNN"
SCRIPT="$GCN_DIR/trainGCN_efficient.py"
MASTER_LOG="$GCN_DIR/run_gcn_rerun_master.log"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

# Mapping name -> (s3_prefix, expected_avg_nodes_per_graph)
S3_BUCKET="amzn-master-sim-bucket"
declare -A S3_PREFIX=(
    [coarse]="graph-dataset_coarse"
    [medium]="graph-dataset"
    [bf25]="graph-dataset_building_focus_25"
)
declare -A EXPECTED_NODES=(
    [coarse]=49500
    [medium]=495000
    [bf25]=1231000
)
declare -A TOLERANCE_PCT=(
    [coarse]=20
    [medium]=20
    [bf25]=20
)

download_dataset() {
    local name=$1
    local s3prefix="${S3_PREFIX[$name]}"
    local data_dir="$GCN_DIR/graph_dataset_${name}_rerun"
    mkdir -p "$data_dir"
    log "Download $name aus s3://$S3_BUCKET/$s3prefix/ nach $data_dir"
    for f in train.pt val.pt test.pt metadata.yaml; do
        if [ -f "$data_dir/$f" ]; then
            log "  [SKIP] $f existiert."
            continue
        fi
        log "  Lade $f ..."
        $AWS s3 cp "s3://$S3_BUCKET/$s3prefix/$f" "$data_dir/$f" \
            --cli-read-timeout 120 --cli-connect-timeout 30 \
            --quiet >>"$MASTER_LOG" 2>&1
        if [ $? -ne 0 ]; then
            log "  FEHLER beim Download von $f. ABBRUCH."
            return 2
        fi
        local sz=$(du -b "$data_dir/$f" | awk '{printf "%.1f MB", $1/(1024*1024)}')
        log "  $f -> $sz"
    done
    return 0
}

sanity_check() {
    local name=$1
    local data_dir="$GCN_DIR/graph_dataset_${name}_rerun"
    local expected=${EXPECTED_NODES[$name]}
    local tol=${TOLERANCE_PCT[$name]}
    log "Sanity-Check $name (erwartet ~${expected} Knoten/Graph, +/-${tol}%, 14 Features)"
    $VENV - "$data_dir" "$expected" "$tol" <<'PYEOF' >>"$MASTER_LOG" 2>&1
import sys, torch
data_dir, expected, tol = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
d = torch.load(f"{data_dir}/train.pt", weights_only=False)
n_graphs = len(d)
total_nodes = sum(g.x.size(0) for g in d)
avg_nodes = total_nodes / n_graphs
feat_dim = d[0].x.size(1)
print(f"  n_graphs={n_graphs} total_nodes={total_nodes:,} avg_nodes={avg_nodes:,.0f} feat_dim={feat_dim}")
ok = True
if feat_dim != 14:
    print(f"  FAIL: feat_dim={feat_dim} (erwartet 14) — moeglicherweise no_cellvol-Datensatz!")
    ok = False
rel_dev = abs(avg_nodes - expected) / expected * 100
print(f"  Abweichung avg_nodes von {expected}: {rel_dev:.1f}% (Toleranz {tol}%)")
if rel_dev > tol:
    print(f"  FAIL: avg_nodes weicht zu stark ab.")
    ok = False
if not ok:
    sys.exit(1)
print("  OK")
PYEOF
    return $?
}

run_training() {
    local name=$1
    local data_dir="$GCN_DIR/graph_dataset_${name}_rerun"
    local out_dir="$GCN_DIR/output_gcn_${name}_rerun"
    local stdout_log="$GCN_DIR/output_gcn_${name}_rerun.log"
    log "Starte Training $name (output: $out_dir, stdout: $stdout_log)"
    cd "$GCN_DIR"
    $VENV "$SCRIPT" \
        --data-dir "$data_dir" \
        --output-dir "$out_dir" \
        --epochs 3000 --patience 100 --batch-size 1 --seed 42 \
        --hidden-dim 128 --num-layers 10 --dropout 0.0 \
        --lr 1e-4 --min-lr 1e-6 \
        --gradient-checkpointing --mixed-precision \
        >"$stdout_log" 2>&1
    local rc=$?
    log "Training $name beendet (exit $rc)"
    return $rc
}

check_stop_reason() {
    local name=$1
    local out_dir="$GCN_DIR/output_gcn_${name}_rerun"
    local hist="$out_dir/training_history.json"
    if [ ! -f "$hist" ]; then
        log "FEHLER: $hist existiert nicht."
        return 2
    fi
    $VENV - "$hist" "$name" <<'PYEOF' | tee -a "$MASTER_LOG"
import sys, json
hist, name = sys.argv[1], sys.argv[2]
h = json.load(open(hist))
sr = h["stop_reason"]
be = h["best_epoch"]
bv = h["best_val_loss"]
te = len(h["train_loss"])
vram_gb = h["peak_vram_bytes"] / (1024**3)
time_h = h["training_time_s"] / 3600
print(f"  {name}: stop_reason={sr}, best_epoch={be}, best_val={bv:.6f}, total_ep={te}, peak_vram={vram_gb:.2f} GB, time={time_h:.2f} h")
# Reasonable: early_stopping ok, max_epochs/nan stop sequence
if sr == "early_stopping":
    sys.exit(0)
elif sr == "nan":
    print("  STOP: NaN-Abbruch. Sequenz angehalten.")
    sys.exit(10)
elif sr == "max_epochs":
    print("  STOP: 3000-Ep-Limit erreicht. Sequenz angehalten (User-Vorgabe).")
    sys.exit(11)
else:
    print(f"  STOP: unbekannter stop_reason={sr}")
    sys.exit(12)
PYEOF
    return $?
}

run_stage() {
    local name=$1
    log "=========================================="
    log "=== STAGE $name START ==="
    log "=========================================="

    download_dataset "$name" || { log "Download fehlgeschlagen. Stop."; exit 2; }
    sanity_check "$name"
    local sc=$?
    if [ $sc -ne 0 ]; then
        log "FEHLER: Sanity-Check $name fehlgeschlagen (rc=$sc). Stop."
        exit 3
    fi
    log "Sanity-Check $name: OK"

    run_training "$name"
    local tr=$?
    if [ $tr -ne 0 ]; then
        log "Training $name brach mit exit=$tr ab. Stop."
        exit 4
    fi

    check_stop_reason "$name"
    local cs=$?
    if [ $cs -ne 0 ]; then
        log "Stage $name nicht sauber beendet (rc=$cs). Sequenz haelt."
        exit $cs
    fi
    log "=== STAGE $name FERTIG (Early Stopping) ==="
}

log "=========================================="
log "=== GCN-RERUN-SEQUENZ START ==="
log "=========================================="
log "Skript: $SCRIPT"
log "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

run_stage coarse
run_stage medium
run_stage bf25

log "=========================================="
log "=== ALLE DREI GCN-RERUN-LAEUFE FERTIG ==="
log "=========================================="

# Nahtloser Uebergang in den Generalisierungstest (GCN + GATv2 auf 5/33 Train-Sims, Medium-Stufe).
# Voraussetzung: graph_dataset_medium_rerun ist lokal vorhanden (von Stage medium).
log "Starte Generalisierungstest-Skript run_gentest_sequence.sh ..."
exec bash /home/tbergermann/Python/GNN/run_gentest_sequence.sh
