#!/bin/bash
# Schwachwind simpleFoam fortsetzen (von letztem Checkpoint bis Time=3000).
# Aufruf: nohup bash ~/laptop_timing/scripts/resume_solver.sh > ~/laptop_timing/logs/resume_solver_stdout.log 2>&1 &

BASE="$HOME/laptop_timing"
CASE="$BASE/medium/case"
LOGS="$BASE/logs"
VENV="$BASE/venv/bin/activate"
AWS="$HOME/.local/bin/aws"
CORES=4

source "$VENV"

RUNLOG="$LOGS/master_run.log"
AUX="$LOGS/aux_times.yaml"
CTRL="$CASE/system/controlDict"

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
log "RESUME SOLVER: Schwachwind simpleFoam fortsetzen"
log "============================================================"

# Letzten geschriebenen Zeitschritt ermitteln
LAST_TIME=$(grep "^Time = " "$CASE/log.simpleFoam" 2>/dev/null | tail -1 | awk '{print $NF}')
log "  Letzter Zeitschritt im Log: ${LAST_TIME:-unbekannt}"

if [ "${LAST_TIME:-0}" -ge 3000 ] 2>/dev/null; then
    log "  Schwachwind simpleFoam bereits vollständig (Time=3000) — überspringe Resume"
else
    log "--- Resume simpleFoam Schwachwind (${LAST_TIME:-?} → 3000, Second-Order) ---"

    # controlDict auf Resume setzen
    sed -i 's/stopAt.*/stopAt          endTime;/' "$CTRL"
    sed -i 's/startFrom.*/startFrom       latestTime;/' "$CTRL"
    sed -i 's/^endTime.*/endTime         3000;/' "$CTRL"
    sed -i 's/^writeInterval[[:space:]].*/writeInterval   100;/' "$CTRL"
    log "  controlDict: startFrom latestTime, endTime 3000, writeInterval 100"

    T0=$SECONDS
    mpirun -np $CORES simpleFoam -parallel -case "$CASE" \
        >> "$CASE/log.simpleFoam_resume" 2>&1
    STATUS=$?
    log_time "simpleFoam_resume_schwachwind" "$T0" "$SECONDS"

    RESUME_CLOCK=$(grep "ClockTime" "$CASE/log.simpleFoam_resume" 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
    log "  ClockTime Resume: ${RESUME_CLOCK:-?} s (Exit: $STATUS)"

    if [ $STATUS -eq 0 ]; then
        step_ok "simpleFoam Resume Schwachwind"
    else
        step_fail "simpleFoam Resume Schwachwind (Exit $STATUS)"
    fi
fi

# reconstructPar
log "--- reconstructPar Schwachwind ---"
reconstructPar -case "$CASE" -latestTime >> "$LOGS/reconstructPar_schwachwind.log" 2>&1 \
    && step_ok "reconstructPar" || step_fail "reconstructPar"

# foamToVTK (finale Daten für Graph-Konstruktion)
log "--- foamToVTK Schwachwind (finale Daten) ---"
T0=$SECONDS
foamToVTK -case "$CASE" -latestTime >> "$LOGS/foamToVTK_schwachwind_final.log" 2>&1 \
    && step_ok "foamToVTK Schwachwind final" || step_fail "foamToVTK Schwachwind final"
log_time "foamToVTK_schwachwind_final" "$T0" "$SECONDS"

mkdir -p "$BASE/vtks/schwachwind_1_5ms_45deg_VTK"
cp -r "$CASE/VTK/"* "$BASE/vtks/schwachwind_1_5ms_45deg_VTK/" 2>/dev/null \
    && step_ok "VTK-Kopie Schwachwind (final)" || step_fail "VTK-Kopie Schwachwind"

# S3-Upload: Timing-Logs sichern
log "--- S3-Upload Timing-Logs ---"
"$AWS" s3 cp "$LOGS/" \
    "s3://amzn-master-sim-bucket/predictions/laptop_timing/logs/" \
    --recursive >> "$RUNLOG" 2>&1 && step_ok "logs/ → S3" || step_fail "logs/ → S3"

log "============================================================"
log "SOLVER RESUME FERTIG. VTK bereit für GNN-Kette."
log "Weiter mit: nohup bash ~/laptop_timing/scripts/run_gnn.sh > ~/laptop_timing/logs/gnn_stdout.log 2>&1 &"
log "============================================================"
