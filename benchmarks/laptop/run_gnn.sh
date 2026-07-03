#!/bin/bash
# GNN-Kette: Graph-Konstruktion, Inferenz (10x), IDW für beide Fälle.
# Voraussetzung: VTK in vtks/sturm_25ms_45deg_VTK/ und vtks/schwachwind_1_5ms_45deg_VTK/
# Aufruf: nohup bash ~/laptop_timing/scripts/run_gnn.sh > ~/laptop_timing/logs/gnn_stdout.log 2>&1 &

BASE="$HOME/laptop_timing"
SCRIPTS="$BASE/scripts"
LOGS="$BASE/logs"
DATA="$BASE/data"
VENV="$BASE/venv/bin/activate"
AWS="$HOME/.local/bin/aws"

source "$VENV"

RUNLOG="$LOGS/master_run.log"
AUX="$LOGS/aux_times.yaml"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RUNLOG"; }
log_time() {
    local key="$1" start="$2" end="$3"
    local secs=$(( end - start ))
    echo "${key}: ${secs}" >> "$AUX"
    log "  ZEIT $key: ${secs}s ($(( secs/60 ))min $(( secs%60 ))s)"
}
step_ok()   { log "  [OK] $*"; }
step_fail() { log "  [FEHLER] $* — weiter"; }

log "============================================================"
log "GNN-KETTE START"
log "============================================================"

cd "$SCRIPTS"

# ── SCHRITT 7: Graphkonstruktion Sturm ───────────────────────────────────────
log "--- SCHRITT 7: Graphkonstruktion Sturm ---"
GRAPH_OUT_STURM="$BASE/graphs/sturm_25ms_45deg_medium.pt"
if [ -f "$GRAPH_OUT_STURM" ]; then
    log "  Graph bereits vorhanden ($GRAPH_OUT_STURM) — überspringe"
    step_ok "graph Sturm (bereits vorhanden)"
else
    T0=$SECONDS
    python run_graph_t2.py \
        --sim-id sturm_25ms_45deg --u-ref 25.0 --angle 45.0 \
        --output-yaml "$LOGS/graph_sturm.yaml" \
        >> "$LOGS/graph_sturm_stdout.log" 2>&1 \
        && step_ok "graph Sturm" || step_fail "graph Sturm"
    log_time "graph_sturm" "$T0" "$SECONDS"
fi

# ── SCHRITT 8: Inferenz Sturm (10×) ──────────────────────────────────────────
log "--- SCHRITT 8: Inferenz Sturm (10 Läufe) ---"
GRAPH_STURM="$BASE/graphs/sturm_25ms_45deg_medium.pt"
[ ! -f "$GRAPH_STURM" ] && GRAPH_STURM="$DATA/sturm_fallback_graph.pt" && log "  [FALLBACK] S3-Graph Sturm"
for i in $(seq 1 3); do
    mkdir -p "$BASE/predictions/sturm_run_${i}"
    python "$SCRIPTS/predict.py" \
        --mode predict --checkpoint "$DATA/best_model.pt" \
        --graph-source "$GRAPH_STURM" --U-ref 25.0 --angle 45.0 \
        --output-dir "$BASE/predictions/sturm_run_${i}" --no-vtk \
        >> "$LOGS/infer_sturm_run${i}.log" 2>&1 \
        && log "  Lauf $i OK" || log "  Lauf $i FEHLER"
done
step_ok "Inferenz Sturm 3 Läufe"

# ── SCHRITT 9: IDW Sturm ─────────────────────────────────────────────────────
log "--- SCHRITT 9: IDW Sturm ---"
T0=$SECONDS
mkdir -p "$BASE/interpolated"
python "$SCRIPTS/interpolate_to_full_mesh.py" \
    --pos  "$BASE/predictions/sturm_run_1/positions.npy" \
    --pred "$BASE/predictions/sturm_run_1/prediction.npy" \
    --mesh "$DATA/sturm_internal.vtu" \
    --output "$BASE/interpolated/sturm_full_mesh.vtu" \
    --method idw --k-neighbors 8 \
    >> "$LOGS/idw_sturm.log" 2>&1 \
    && step_ok "IDW Sturm" || step_fail "IDW Sturm"
log_time "idw_sturm" "$T0" "$SECONDS"

# ── SCHRITT 10: Graphkonstruktion Schwachwind ─────────────────────────────────
log "--- SCHRITT 10: Graphkonstruktion Schwachwind ---"
GRAPH_OUT_SW="$BASE/graphs/schwachwind_1_5ms_45deg_medium.pt"
if [ -f "$GRAPH_OUT_SW" ]; then
    log "  Graph bereits vorhanden ($GRAPH_OUT_SW) — überspringe"
    step_ok "graph Schwachwind (bereits vorhanden)"
else
    T0=$SECONDS
    python run_graph_t2.py \
        --sim-id schwachwind_1_5ms_45deg --u-ref 1.5 --angle 45.0 \
        --output-yaml "$LOGS/graph_schwachwind.yaml" \
        >> "$LOGS/graph_schwachwind_stdout.log" 2>&1 \
        && step_ok "graph Schwachwind" || step_fail "graph Schwachwind"
    log_time "graph_schwachwind" "$T0" "$SECONDS"
fi

# ── SCHRITT 11: Inferenz Schwachwind (10×) ────────────────────────────────────
log "--- SCHRITT 11: Inferenz Schwachwind (10 Läufe) ---"
GRAPH_SW="$BASE/graphs/schwachwind_1_5ms_45deg_medium.pt"
[ ! -f "$GRAPH_SW" ] && GRAPH_SW="$DATA/schwachwind_fallback_graph.pt" && log "  [FALLBACK] S3-Graph Schwachwind"
for i in $(seq 1 3); do
    mkdir -p "$BASE/predictions/schwachwind_run_${i}"
    python "$SCRIPTS/predict.py" \
        --mode predict --checkpoint "$DATA/best_model.pt" \
        --graph-source "$GRAPH_SW" --U-ref 1.5 --angle 45.0 \
        --output-dir "$BASE/predictions/schwachwind_run_${i}" --no-vtk \
        >> "$LOGS/infer_schwachwind_run${i}.log" 2>&1 \
        && log "  Lauf $i OK" || log "  Lauf $i FEHLER"
done
step_ok "Inferenz Schwachwind 3 Läufe"

# ── SCHRITT 12: IDW Schwachwind ───────────────────────────────────────────────
log "--- SCHRITT 12: IDW Schwachwind ---"
T0=$SECONDS
python "$SCRIPTS/interpolate_to_full_mesh.py" \
    --pos  "$BASE/predictions/schwachwind_run_1/positions.npy" \
    --pred "$BASE/predictions/schwachwind_run_1/prediction.npy" \
    --mesh "$DATA/schwachwind_internal.vtu" \
    --output "$BASE/interpolated/schwachwind_full_mesh.vtu" \
    --method idw --k-neighbors 8 \
    >> "$LOGS/idw_schwachwind.log" 2>&1 \
    && step_ok "IDW Schwachwind" || step_fail "IDW Schwachwind"
log_time "idw_schwachwind" "$T0" "$SECONDS"

# ── SCHRITT 13: Konsolidierung ────────────────────────────────────────────────
log "--- SCHRITT 13: Konsolidierung ---"
cd "$SCRIPTS"
python consolidate_results.py >> "$RUNLOG" 2>&1 \
    && step_ok "Konsolidierung" || step_fail "Konsolidierung"

# ── SCHRITT 14: S3-Upload ─────────────────────────────────────────────────────
log "--- SCHRITT 14: S3-Upload ---"
"$AWS" s3 cp "$BASE/run_log.yaml" \
    "s3://amzn-master-sim-bucket/predictions/laptop_timing/run_log.yaml" \
    >> "$RUNLOG" 2>&1 && step_ok "run_log.yaml → S3" || step_fail "run_log.yaml"
"$AWS" s3 cp "$LOGS/" \
    "s3://amzn-master-sim-bucket/predictions/laptop_timing/logs/" \
    --recursive >> "$RUNLOG" 2>&1 && step_ok "logs/ → S3" || step_fail "logs/ → S3"

log "============================================================"
log "GNN-KETTE FERTIG. Ergebnisse: $BASE/run_log.yaml"
log "S3: s3://amzn-master-sim-bucket/predictions/laptop_timing/"
log "============================================================"
