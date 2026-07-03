#!/bin/bash
# Graceful Stop: setzt stopAt writeNow — simpleFoam schreibt aktuellen Stand und beendet sich.
# Aufruf: bash ~/laptop_timing/scripts/stop.sh

BASE="$HOME/laptop_timing"
CASE="$BASE/medium/case"
LOGS="$BASE/logs"
AUX="$LOGS/aux_times.yaml"
RUNLOG="$LOGS/master_run.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RUNLOG"; }

log "STOP-SIGNAL: setze stopAt writeNow in controlDict"

# Haupt-controlDict anpassen
CTRL="$CASE/system/controlDict"
if [ -f "$CTRL" ]; then
    sed -i 's/stopAt[[:space:]].*endTime;/stopAt          writeNow;/' "$CTRL"
    log "  Updated: $CTRL"
else
    log "  FEHLER: $CTRL nicht gefunden"
    exit 1
fi

# Aktuellen Stand aus simpleFoam-Log auslesen
LAST_TIME=$(grep "^Time = " "$CASE/log.simpleFoam" 2>/dev/null | tail -1 | awk '{print $NF}')
CLOCK=$(grep "ClockTime" "$CASE/log.simpleFoam" 2>/dev/null | tail -1 | awk '{print $NF}')

log "  Letzter simpleFoam-Zeitschritt: ${LAST_TIME:-unbekannt}"
log "  ClockTime beim Stop:            ${CLOCK:-unbekannt} s"

echo "simpleFoam_stop_at_iter: ${LAST_TIME:-unknown}" >> "$AUX"
echo "simpleFoam_morning_clocktime_s: ${CLOCK:-0}" >> "$AUX"

log "Stop-Signal gesetzt. simpleFoam schreibt aktuellen Stand und stoppt sauber."
log "Laptop kann nach Prozessende verwendet werden."
log "Resume abends mit: bash ~/laptop_timing/scripts/resume_gnn.sh"
