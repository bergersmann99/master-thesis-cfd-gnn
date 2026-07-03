#!/bin/bash
# ====================================================================
# GATv2 Efficient — Variante OHNE cell_volume Feature
# Datensatz: graph-dataset_medium_no_cellvol (13 Input-Features)
# ====================================================================
# WICHTIG: Eigene Output-Pfade — überschreibt KEINE existierenden
# Modelle (gatv2_medium_h128, etc.).
# ====================================================================

VENV="/home/tbergermann/Python/venv/bin"
PYTHON="$VENV/python"
AWS="$VENV/aws"
LOG="/home/tbergermann/Python/master_gatv2_no_cellvol.log"
S3_BUCKET="amzn-master-sim-bucket"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date)] $1" | tee -a "$LOG"; }

log "===== START: GATv2 Efficient (NO cell_volume) ====="

# ── Daten + Output ──────────────────────────────────────────────────
DATA_DIR="/home/tbergermann/Python/datasets/medium_no_cellvol"
OUT_DIR="/home/tbergermann/Python/GAT/output_gatv2_medium_no_cellvol_h128"
TRAIN_LOG="/home/tbergermann/Python/GAT/output_gatv2_medium_no_cellvol_h128.log"

if [ ! -f "$DATA_DIR/train.pt" ]; then
    log "FEHLER: Datensatz fehlt — $DATA_DIR/train.pt"
    log "Bitte zuerst Datensatz herunterladen."
    exit 1
fi

if [ -d "$OUT_DIR" ]; then
    if [ -f "$OUT_DIR/best_model.pt" ]; then
        log "Output-Verzeichnis vorhanden mit best_model.pt — Resume-Modus aktiv."
    else
        log "WARNUNG: Output-Verzeichnis existiert ohne best_model.pt: $OUT_DIR"
        log "Trainingsstart abgebrochen, um keine Ergebnisse zu überschreiben."
        log "Bitte $OUT_DIR umbenennen oder löschen, dann erneut starten."
        exit 1
    fi
fi

# ── Training starten ────────────────────────────────────────────────
log "Starte GATv2 medium_no_cellvol (13 Features, h=128, L=10, H=4, grad_ckpt+bf16)..."
cd /home/tbergermann/Python/GAT

$PYTHON trainGATv2_efficient_no_cellvol.py \
    --data-dir "$DATA_DIR" \
    --hidden-dim 128 \
    --num-layers 10 \
    --heads 4 \
    --batch-size 1 \
    --epochs 3000 \
    --patience 100 \
    --gradient-checkpointing \
    --mixed-precision \
    --output-dir "$OUT_DIR" \
    > "$TRAIN_LOG" 2>&1

EXIT=$?

if [ $EXIT -eq 0 ]; then
    log "Training abgeschlossen (Exit: 0)."
    $AWS s3 cp "$OUT_DIR/best_model.pt" \
        "s3://$S3_BUCKET/models/gatv2_medium_no_cellvol_efficient/best_model.pt" >> "$LOG" 2>&1
    $AWS s3 cp "$OUT_DIR/test_metrics.json" \
        "s3://$S3_BUCKET/models/gatv2_medium_no_cellvol_efficient/test_metrics.json" >> "$LOG" 2>&1
    $AWS s3 cp "$OUT_DIR/training_history.json" \
        "s3://$S3_BUCKET/models/gatv2_medium_no_cellvol_efficient/training_history.json" >> "$LOG" 2>&1
    log "Modell nach S3 gesichert: s3://$S3_BUCKET/models/gatv2_medium_no_cellvol_efficient/"
else
    log "FEHLER: Training abgestürzt (Exit: $EXIT). Log: $TRAIN_LOG"
fi

log "===== PIPELINE ABGESCHLOSSEN ====="
