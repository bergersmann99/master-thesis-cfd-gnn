#!/bin/bash
# Pipeline: Download + Training building_focus_25 (GCN → GATv2)
# 25% gebäudenah, ~1.23M Knoten, ~24.6M Kanten

VENV="/home/tbergermann/Python/venv/bin"
PYTHON="$VENV/python"
AWS="$VENV/aws"
LOG="/home/tbergermann/Python/master_building_focus_25.log"
DATA_DIR="/home/tbergermann/Python/datasets/building_focus_25"
GCN_OUT="/home/tbergermann/Python/GNN/output_gcn_building_focus_25"
GAT_OUT="/home/tbergermann/Python/GAT/output_gatv2_building_focus_25"
S3_BUCKET="amzn-master-sim-bucket"
S3_PREFIX="graph-dataset_building_focus_25"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date)] $1" | tee -a "$LOG"; }

log "===== START: building_focus_25 Pipeline ====="

# ── 1. Download ─────────────────────────────────────────────────────────────
log "Starte S3-Download (~23 GB)..."
$AWS s3 cp "s3://$S3_BUCKET/$S3_PREFIX/train.pt" "$DATA_DIR/train.pt" >> "$LOG" 2>&1
$AWS s3 cp "s3://$S3_BUCKET/$S3_PREFIX/val.pt"   "$DATA_DIR/val.pt"   >> "$LOG" 2>&1
$AWS s3 cp "s3://$S3_BUCKET/$S3_PREFIX/test.pt"  "$DATA_DIR/test.pt"  >> "$LOG" 2>&1
log "Download abgeschlossen."

# ── 2. GCN Training ─────────────────────────────────────────────────────────
log "Starte GCN building_focus_25 (hidden=128, layers=10, batch=1)..."
cd /home/tbergermann/Python/GNN
$PYTHON trainGCN.py \
    --data-dir "$DATA_DIR" \
    --hidden-dim 128 \
    --num-layers 10 \
    --batch-size 1 \
    --epochs 800 \
    --patience 100 \
    --output-dir "$GCN_OUT" \
    > /home/tbergermann/Python/GNN/output_gcn_building_focus_25.log 2>&1
GCN_EXIT=$?

if [ $GCN_EXIT -eq 0 ]; then
    log "GCN building_focus_25 abgeschlossen (Exit: 0)."
    $AWS s3 cp "$GCN_OUT/best_model.pt" \
        "s3://$S3_BUCKET/models/gcn_building_focus_25/best_model.pt" >> "$LOG" 2>&1
    log "GCN nach S3 gesichert."
else
    log "FEHLER: GCN building_focus_25 abgestürzt (Exit: $GCN_EXIT)."
fi

# ── 3. GPU freigeben ─────────────────────────────────────────────────────────
log "Warte 30s auf GPU-Speicherfreigabe..."
sleep 30
VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
log "GPU nach GCN: ${VRAM} MiB belegt."

# ── 4. GATv2 Training ────────────────────────────────────────────────────────
log "Starte GATv2 building_focus_25 (hidden=64, layers=6, heads=4)..."
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
    > /home/tbergermann/Python/GAT/output_gatv2_building_focus_25.log 2>&1
GAT_EXIT=$?

if [ $GAT_EXIT -eq 0 ]; then
    log "GATv2 building_focus_25 abgeschlossen (Exit: 0)."
    $AWS s3 cp "$GAT_OUT/best_model.pt" \
        "s3://$S3_BUCKET/models/gatv2_building_focus_25/best_model.pt" >> "$LOG" 2>&1
    log "GATv2 nach S3 gesichert."
else
    log "FEHLER: GATv2 building_focus_25 abgestürzt (Exit: $GAT_EXIT)."
fi

log "===== PIPELINE ABGESCHLOSSEN ====="
