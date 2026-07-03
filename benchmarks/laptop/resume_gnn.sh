#!/bin/bash
# Resume: simpleFoam fortsetzen (falls noch nicht fertig) + GNN-Kette starten.
# Aufruf: bash ~/laptop_timing/scripts/resume_gnn.sh

BASE="$HOME/laptop_timing"
SCRIPTS="$BASE/scripts"
LOGS="$BASE/logs"
DATA="$BASE/data"
VENV="$BASE/venv/bin/activate"
AWS="$HOME/.local/bin/aws"
CORES=4

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
step_fail() { log "  [FEHLER] $* — weiter mit naechstem Schritt"; }

CASE="$BASE/medium/case"
CTRL="$CASE/system/controlDict"

log "============================================================"
log "RESUME: Prüfe ob simpleFoam noch fortgesetzt werden muss"
log "============================================================"

# ── Prüfe letzten geschriebenen Zeitschritt ─────────────────────────────────
LAST_TIME=$(grep "^Time = " "$CASE/log.simpleFoam" 2>/dev/null | tail -1 | awk '{print $NF}')
log "  Letzter Zeitschritt im Log: ${LAST_TIME:-unbekannt}"

if [ "${LAST_TIME:-0}" -lt 3000 ] 2>/dev/null; then
    log "--- RESUME simpleFoam (${LAST_TIME} → 3000, Second-Order) ---"

    # controlDict auf Resume setzen
    sed -i 's/stopAt.*/stopAt          endTime;/' "$CTRL"
    sed -i 's/startFrom.*/startFrom       latestTime;/' "$CTRL"
    sed -i 's/endTime.*/endTime         3000;/' "$CTRL"
    sed -i 's/writeInterval.*/writeInterval   100;/' "$CTRL"

    T0=$SECONDS
    mpirun -np $CORES simpleFoam -parallel -case "$CASE" \
        >> "$CASE/log.simpleFoam_resume" 2>&1
    RESUME_STATUS=$?
    log_time "simpleFoam_resume" "$T0" "$SECONDS"

    if [ $RESUME_STATUS -eq 0 ]; then
        step_ok "simpleFoam Resume"
    else
        step_fail "simpleFoam Resume (Exit $RESUME_STATUS)"
    fi

    # Gesamte simpleFoam-Zeit (Morgen + Abend) berechnen
    MORNING=$(grep "^simpleFoam_morning_clocktime_s:" "$AUX" 2>/dev/null | tail -1 | awk '{print $2}')
    # ClockTime-Zeile endet auf "... ClockTime = 13 s" -> Zahl ist das vorletzte Feld
    RESUME_CLOCK=$(grep "ClockTime" "$CASE/log.simpleFoam_resume" 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
    if [ -n "$MORNING" ] && [ -n "$RESUME_CLOCK" ]; then
        TOTAL=$(( MORNING + RESUME_CLOCK ))
        echo "simpleFoam_total_clocktime_s: ${TOTAL}" >> "$AUX"
        log "  simpleFoam Gesamt: ${MORNING}s (Morgen) + ${RESUME_CLOCK}s (Abend) = ${TOTAL}s"
    fi
else
    log "  simpleFoam bereits bei Time=3000 — überspringe Resume"
fi

# ── reconstructPar + foamToVTK (finale Daten) ───────────────────────────────
log "--- Reconstruct + foamToVTK Sturm (finale Daten) ---"
T0=$SECONDS
reconstructPar -case "$CASE" -latestTime >> "$LOGS/reconstructPar_resume.log" 2>&1 \
    && step_ok "reconstructPar" || step_fail "reconstructPar"

foamToVTK -case "$CASE" -latestTime >> "$LOGS/foamToVTK_sturm_final.log" 2>&1 \
    && step_ok "foamToVTK Sturm (final)" || step_fail "foamToVTK Sturm (final)"
log_time "foamToVTK_sturm_final" "$T0" "$SECONDS"

mkdir -p "$BASE/vtks/sturm_25ms_45deg_VTK"
cp -r "$CASE/VTK/"* "$BASE/vtks/sturm_25ms_45deg_VTK/" 2>/dev/null \
    && step_ok "VTK-Kopie Sturm (final)" || step_fail "VTK-Kopie Sturm (final)"

# ── Schwachwind VTK aus S3-Daten für Graph-Konstruktion ──────────────────────
log "--- Schwachwind VTK-Verzeichnis aus S3-internal.vtu erstellen ---"
SW_VTK="$BASE/vtks/schwachwind_1_5ms_45deg_VTK/schwachwind_3000"
mkdir -p "$SW_VTK"
if [ ! -f "$SW_VTK/internal.vtu" ]; then
    cp "$DATA/schwachwind_internal.vtu" "$SW_VTK/internal.vtu" \
        && step_ok "schwachwind VTK-Proxy aus S3-Daten" || step_fail "schwachwind VTK-Proxy"
else
    step_ok "schwachwind VTK-Proxy bereits vorhanden"
fi

# ── GNN-KETTE ─────────────────────────────────────────────────────────────────
log "============================================================"
log "GNN-KETTE START"
log "============================================================"

cd "$SCRIPTS"

# SCHRITT 7: Graphkonstruktion Sturm
log "--- SCHRITT 7: Graphkonstruktion Sturm ---"
T0=$SECONDS
python run_graph_t2.py \
    --sim-id sturm_25ms_45deg --u-ref 25.0 --angle 45.0 \
    --output-yaml "$LOGS/graph_sturm.yaml" \
    >> "$LOGS/graph_sturm_stdout.log" 2>&1 \
    && step_ok "graph Sturm" || step_fail "graph Sturm"
log_time "graph_sturm" "$T0" "$SECONDS"

# SCHRITT 8: Inferenz Sturm (10×)
log "--- SCHRITT 8: Inferenz Sturm (10 Läufe) ---"
GRAPH_STURM="$BASE/graphs/sturm_25ms_45deg_medium.pt"
[ ! -f "$GRAPH_STURM" ] && GRAPH_STURM="$DATA/sturm_fallback_graph.pt" && log "  [FALLBACK] S3-Graph Sturm"
for i in $(seq 1 10); do
    mkdir -p "$BASE/predictions/sturm_run_${i}"
    python "$SCRIPTS/predict.py" \
        --mode predict --checkpoint "$DATA/best_model.pt" \
        --graph-source "$GRAPH_STURM" --U-ref 25.0 --angle 45.0 \
        --output-dir "$BASE/predictions/sturm_run_${i}" --no-vtk \
        >> "$LOGS/infer_sturm_run${i}.log" 2>&1 \
        && log "  Lauf $i OK" || log "  Lauf $i FEHLER"
done
step_ok "Inferenz Sturm 10 Läufe"

# SCHRITT 9: IDW Sturm
log "--- SCHRITT 9: IDW Sturm ---"
T0=$SECONDS
python "$SCRIPTS/interpolate_to_full_mesh.py" \
    --pos  "$BASE/predictions/sturm_run_1/positions.npy" \
    --pred "$BASE/predictions/sturm_run_1/prediction.npy" \
    --mesh "$DATA/sturm_internal.vtu" \
    --output "$BASE/interpolated/sturm_full_mesh.vtu" \
    --method idw --k-neighbors 8 \
    >> "$LOGS/idw_sturm.log" 2>&1 \
    && step_ok "IDW Sturm" || step_fail "IDW Sturm"
log_time "idw_sturm" "$T0" "$SECONDS"

# SCHRITT 10: Graphkonstruktion Schwachwind
log "--- SCHRITT 10: Graphkonstruktion Schwachwind ---"
T0=$SECONDS
python run_graph_t2.py \
    --sim-id schwachwind_1_5ms_45deg --u-ref 1.5 --angle 45.0 \
    --output-yaml "$LOGS/graph_schwachwind.yaml" \
    >> "$LOGS/graph_schwachwind_stdout.log" 2>&1 \
    && step_ok "graph Schwachwind" || step_fail "graph Schwachwind"
log_time "graph_schwachwind" "$T0" "$SECONDS"

# SCHRITT 11: Inferenz Schwachwind (10×)
log "--- SCHRITT 11: Inferenz Schwachwind (10 Läufe) ---"
GRAPH_SW="$BASE/graphs/schwachwind_1_5ms_45deg_medium.pt"
[ ! -f "$GRAPH_SW" ] && GRAPH_SW="$DATA/schwachwind_fallback_graph.pt" && log "  [FALLBACK] S3-Graph Schwachwind"
for i in $(seq 1 10); do
    mkdir -p "$BASE/predictions/schwachwind_run_${i}"
    python "$SCRIPTS/predict.py" \
        --mode predict --checkpoint "$DATA/best_model.pt" \
        --graph-source "$GRAPH_SW" --U-ref 1.5 --angle 45.0 \
        --output-dir "$BASE/predictions/schwachwind_run_${i}" --no-vtk \
        >> "$LOGS/infer_schwachwind_run${i}.log" 2>&1 \
        && log "  Lauf $i OK" || log "  Lauf $i FEHLER"
done
step_ok "Inferenz Schwachwind 10 Läufe"

# SCHRITT 12: IDW Schwachwind
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

# SCHRITT 13: Konsolidierung
log "--- SCHRITT 13: Konsolidierung ---"
cd "$SCRIPTS"
python consolidate_results.py >> "$RUNLOG" 2>&1 \
    && step_ok "Konsolidierung" || step_fail "Konsolidierung"

# SCHRITT 14: S3-Upload
log "--- SCHRITT 14: S3-Upload ---"
"$AWS" s3 cp "$BASE/run_log.yaml" \
    "s3://amzn-master-sim-bucket/predictions/laptop_timing/run_log.yaml" \
    >> "$RUNLOG" 2>&1 && step_ok "run_log.yaml → S3" || step_fail "run_log.yaml → S3"
"$AWS" s3 cp "$LOGS/" \
    "s3://amzn-master-sim-bucket/predictions/laptop_timing/logs/" \
    --recursive >> "$RUNLOG" 2>&1 && step_ok "logs/ → S3" || step_fail "logs/ → S3"

log "============================================================"
log "FERTIG. Ergebnisse: $BASE/run_log.yaml"
log "S3: s3://amzn-master-sim-bucket/predictions/laptop_timing/"
log "============================================================"
