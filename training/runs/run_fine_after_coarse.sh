#!/bin/bash
# Startet GCN Fine Training. GATv2 Fine ist aufgrund von VRAM-Constraints nicht trainierbar.
#
# GATv2 Begründung:
# GATv2Conv speichert x_i, x_j und x = x_i+x_j pro Schicht für Backprop.
# Mit 102M Kanten und hidden_dim/heads=8 ergibt sich ~42 GB pro Schicht.
# Bereits nach 2 Schichten werden >84 GB akkumuliert → überschreitet 95 GB VRAM.
# GCN speichert nur den skalaren edge_weight (408 MB/Schicht) → skaliert zu Fine.

VENV_PYTHON="/home/tbergermann/Python/venv/bin/python"
FINE_DATA="/home/tbergermann/Python/datasets/fine"
GCN_DIR="/home/tbergermann/Python/GNN"
LOG_DIR="/home/tbergermann/Python"

wait_for_gpu_free() {
    echo "[Master] Warte auf GPU-Speicherfreigabe..."
    while true; do
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
        echo "[Master] GPU-Speicher belegt: ${USED} MiB"
        if [ -n "$USED" ] && [ "$USED" -lt 5000 ]; then
            echo "[Master] GPU frei (${USED} MiB belegt)."
            break
        fi
        sleep 30
    done
}

echo "[Master] $(date) Fine-Download prüfen..."
while [ ! -f "$FINE_DATA/train.pt" ] || [ ! -f "$FINE_DATA/val.pt" ] || [ ! -f "$FINE_DATA/test.pt" ]; do
    sleep 60
    echo "[Master] Warte auf Fine-Download..."
done

wait_for_gpu_free

echo "[Master] $(date) Starte GCN Fine Training (hidden_dim=64, num_layers=8)..."
cd $GCN_DIR
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $VENV_PYTHON trainGCN.py \
    --data-dir $FINE_DATA \
    --output-dir $GCN_DIR/output_gcn_fine \
    --epochs 800 \
    --patience 100 \
    --hidden-dim 64 \
    --num-layers 8 \
    --batch-size 1 \
    --lr 1e-4 \
    --min-lr 1e-6 \
    --seed 42 \
    > $LOG_DIR/GNN/output_gcn_fine.log 2>&1 &
GCN_FINE_PID=$!
echo "[Master] GCN Fine gestartet (PID: $GCN_FINE_PID)"

$VENV_PYTHON $LOG_DIR/monitor_gcn_fine.py > $LOG_DIR/monitor_gcn_fine_out.log 2>&1 &

wait $GCN_FINE_PID
GCN_EXIT=$?
echo "[Master] $(date) GCN Fine beendet (Exit: $GCN_EXIT)"

if [ $GCN_EXIT -ne 0 ]; then
    echo "[Master] FEHLER: GCN Fine abgestürzt."
    exit 1
fi

/home/tbergermann/Python/venv/bin/aws s3 sync $GCN_DIR/output_gcn_fine/ \
    s3://amzn-master-sim-bucket/models/gcn_fine/ --exclude "training.log"
echo "[Master] GCN Fine nach S3 gesichert."
echo "[Master] $(date) FERTIG — GATv2 Fine nicht trainierbar (VRAM-Constraint, siehe Log)."
