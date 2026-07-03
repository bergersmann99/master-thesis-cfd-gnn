#!/bin/bash
# Pipeline: Training building_focus (GCN → GATv2)
# Fix: PYTORCH_CUDA_ALLOC_CONF gegen Speicherfragmentierung

VENV="/home/tbergermann/Python/venv/bin"
PYTHON="$VENV/python"
AWS="$VENV/aws"
LOG="/home/tbergermann/Python/master_building_focus.log"
DATA_DIR="/home/tbergermann/Python/datasets/building_focus"
GCN_OUT="/home/tbergermann/Python/GNN/output_gcn_building_focus"
GAT_OUT="/home/tbergermann/Python/GAT/output_gatv2_building_focus"
S3_BUCKET="amzn-master-sim-bucket"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date)] $1" | tee -a "$LOG"; }

log "===== START: building_focus Pipeline (expandable_segments) ====="

# ── 1. GCN Training ─────────────────────────────────────────────────────────
log "Starte GCN building_focus (hidden=128, layers=10, expandable_segments)..."
cd /home/tbergermann/Python/GNN
$PYTHON trainGCN.py \
    --data-dir "$DATA_DIR" \
    --hidden-dim 128 \
    --num-layers 10 \
    --batch-size 1 \
    --epochs 800 \
    --patience 100 \
    --output-dir "$GCN_OUT" \
    > /home/tbergermann/Python/GNN/output_gcn_building_focus.log 2>&1
GCN_EXIT=$?

if [ $GCN_EXIT -eq 0 ]; then
    log "GCN building_focus abgeschlossen (Exit: 0)."
    $AWS s3 cp "$GCN_OUT/best_model.pt" \
        "s3://$S3_BUCKET/models/gcn_building_focus/best_model.pt" >> "$LOG" 2>&1
    log "GCN nach S3 gesichert."
else
    log "FEHLER: GCN OOM mit hidden=128. Starte Fallback mit hidden=64, layers=10..."
    cd /home/tbergermann/Python/GNN
    GCN_OUT_FB="/home/tbergermann/Python/GNN/output_gcn_building_focus_h64"
    $PYTHON trainGCN.py \
        --data-dir "$DATA_DIR" \
        --hidden-dim 64 \
        --num-layers 10 \
        --batch-size 1 \
        --epochs 800 \
        --patience 100 \
        --output-dir "$GCN_OUT_FB" \
        > /home/tbergermann/Python/GNN/output_gcn_building_focus_h64.log 2>&1
    GCN_FB_EXIT=$?
    if [ $GCN_FB_EXIT -eq 0 ]; then
        log "GCN building_focus (hidden=64) abgeschlossen."
        $AWS s3 cp "$GCN_OUT_FB/best_model.pt" \
            "s3://$S3_BUCKET/models/gcn_building_focus_h64/best_model.pt" >> "$LOG" 2>&1
        log "GCN (hidden=64) nach S3 gesichert."
    else
        log "FEHLER: GCN Fallback auch gescheitert (Exit: $GCN_FB_EXIT)."
    fi
fi

# ── 2. GPU freigeben ─────────────────────────────────────────────────────────
log "Warte 30s auf GPU-Speicherfreigabe..."
sleep 30
VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
log "GPU nach GCN: ${VRAM} MiB belegt."

# ── 3. GATv2 Training ────────────────────────────────────────────────────────
log "Starte GATv2 building_focus (hidden=64, layers=6, heads=4)..."
cd /home/tbergermann/Python/GAT
$PYTHON trainGATv2.py \
    --data-dir "$DATA_DIR" \
    --hidden-dim 64 \
    --num-layers 6 \
    --heads 4 \
    --batch-size 1 \
    --epochs 800 \
    --patience 100 \
    --output-dir "$GAT_OUT" \
    > /home/tbergermann/Python/GAT/output_gatv2_building_focus.log 2>&1
GAT_EXIT=$?

if [ $GAT_EXIT -eq 0 ]; then
    log "GATv2 building_focus abgeschlossen (Exit: 0)."
    $AWS s3 cp "$GAT_OUT/best_model.pt" \
        "s3://$S3_BUCKET/models/gatv2_building_focus/best_model.pt" >> "$LOG" 2>&1
    log "GATv2 nach S3 gesichert."
else
    log "FEHLER: GATv2 building_focus abgestürzt (Exit: $GAT_EXIT)."
fi

log "===== PIPELINE ABGESCHLOSSEN ====="
