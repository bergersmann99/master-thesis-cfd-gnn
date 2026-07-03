#!/bin/bash
# Pipeline: GATv2 efficient auf building_focus_25 und building_focus
# Gradient Checkpointing + Mixed Precision (BF16)

VENV="/home/tbergermann/Python/venv/bin"
PYTHON="$VENV/python"
AWS="$VENV/aws"
LOG="/home/tbergermann/Python/master_gatv2_efficient.log"
S3_BUCKET="amzn-master-sim-bucket"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date)] $1" | tee -a "$LOG"; }

log "===== START: GATv2 Efficient Pipeline ====="

# ── 1. GATv2 building_focus_25 (~1.23M Knoten, ~24.6M Kanten) ───────
BF25_DATA="/home/tbergermann/Python/datasets/building_focus_25"
BF25_OUT="/home/tbergermann/Python/GAT/output_gatv2_bf25_efficient"

log "Starte GATv2 building_focus_25 (h=128, L=10, H=4, grad_ckpt+bf16)..."
cd /home/tbergermann/Python/GAT
$PYTHON trainGATv2_efficient.py \
    --data-dir "$BF25_DATA" \
    --hidden-dim 128 \
    --num-layers 10 \
    --heads 4 \
    --batch-size 1 \
    --epochs 1200 \
    --patience 100 \
    --gradient-checkpointing \
    --mixed-precision \
    --output-dir "$BF25_OUT" \
    > /home/tbergermann/Python/GAT/output_gatv2_bf25_efficient.log 2>&1
BF25_EXIT=$?

if [ $BF25_EXIT -eq 0 ]; then
    log "GATv2 building_focus_25 abgeschlossen (Exit: 0)."
    $AWS s3 cp "$BF25_OUT/best_model.pt" \
        "s3://$S3_BUCKET/models/gatv2_bf25_efficient/best_model.pt" >> "$LOG" 2>&1
    $AWS s3 cp "$BF25_OUT/test_metrics.json" \
        "s3://$S3_BUCKET/models/gatv2_bf25_efficient/test_metrics.json" >> "$LOG" 2>&1
    $AWS s3 cp "$BF25_OUT/training_history.json" \
        "s3://$S3_BUCKET/models/gatv2_bf25_efficient/training_history.json" >> "$LOG" 2>&1
    log "GATv2 bf25 nach S3 gesichert."
else
    log "FEHLER: GATv2 building_focus_25 abgestürzt (Exit: $BF25_EXIT)."
fi

# ── 2. GPU freigeben ─────────────────────────────────────────────────
log "Warte 30s auf GPU-Speicherfreigabe..."
sleep 30
VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
log "GPU nach bf25: ${VRAM} MiB belegt."

# ── 3. GATv2 building_focus (~2.46M Knoten, ~49.2M Kanten) ──────────
BF_DATA="/home/tbergermann/Python/datasets/building_focus"
BF_OUT="/home/tbergermann/Python/GAT/output_gatv2_bf_efficient"

log "Starte GATv2 building_focus (h=128, L=10, H=4, grad_ckpt+bf16)..."
cd /home/tbergermann/Python/GAT
$PYTHON trainGATv2_efficient.py \
    --data-dir "$BF_DATA" \
    --hidden-dim 128 \
    --num-layers 10 \
    --heads 4 \
    --batch-size 1 \
    --epochs 1200 \
    --patience 100 \
    --gradient-checkpointing \
    --mixed-precision \
    --output-dir "$BF_OUT" \
    > /home/tbergermann/Python/GAT/output_gatv2_bf_efficient.log 2>&1
BF_EXIT=$?

if [ $BF_EXIT -eq 0 ]; then
    log "GATv2 building_focus abgeschlossen (Exit: 0)."
    $AWS s3 cp "$BF_OUT/best_model.pt" \
        "s3://$S3_BUCKET/models/gatv2_bf_efficient/best_model.pt" >> "$LOG" 2>&1
    $AWS s3 cp "$BF_OUT/test_metrics.json" \
        "s3://$S3_BUCKET/models/gatv2_bf_efficient/test_metrics.json" >> "$LOG" 2>&1
    $AWS s3 cp "$BF_OUT/training_history.json" \
        "s3://$S3_BUCKET/models/gatv2_bf_efficient/training_history.json" >> "$LOG" 2>&1
    log "GATv2 bf nach S3 gesichert."
else
    log "FEHLER: GATv2 building_focus abgestürzt (Exit: $BF_EXIT)."
fi

log "===== PIPELINE ABGESCHLOSSEN ====="
