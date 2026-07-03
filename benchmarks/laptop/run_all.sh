#!/bin/bash
# Master-Orchestrierungs-Script: Laptop-Zeitmessung Surrogat vs. CFD
# Laeuft unbeaufsichtigt. Fehler einzelner Schritte werden geloggt,
# der Lauf wird NICHT abgebrochen.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$SCRIPT_DIR"
SCRIPTS="$BASE/scripts"
LOGS="$BASE/logs"
DATA="$BASE/data"
VENV="$BASE/venv/bin/activate"
AWS="$HOME/.local/bin/aws"
CORES=4

source "$VENV"
# OpenFOAM-Tools sind bereits im PATH (systemweit gesourct)

mkdir -p "$LOGS"
RUN_LOG="$LOGS/master_run.log"
AUX="$LOGS/aux_times.yaml"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RUN_LOG"; }
log_time() {
    local key="$1" start="$2" end="$3"
    local secs=$(( end - start ))
    echo "${key}: ${secs}" >> "$AUX"
    log "  ZEIT $key: ${secs}s ($(( secs/60 ))min $(( secs%60 ))s)"
}
step_ok()   { log "  [OK] $*"; }
step_fail() { log "  [FEHLER] $* — weiter mit naechstem Schritt"; }

log "============================================================"
log "START Laptop-Zeitmessung"
log "Cores: $CORES | OpenFOAM: $(blockMesh --version 2>&1 | head -1 || echo unbekannt)"
log "Python: $(python --version 2>&1)"
log "============================================================"

# ── SCHRITT 0: Daten aus S3 laden ────────────────────────────────────────────
log "--- SCHRITT 0: S3-Downloads ---"

T0=$SECONDS
log "  Lade best_model.pt (falls nicht vorhanden) ..."
[ -f "$DATA/best_model.pt" ] && step_ok "best_model.pt (bereits vorhanden)" || \
    { "$AWS" s3 cp "s3://amzn-master-sim-bucket/models/gatv2_medium_h128/best_model.pt" \
        "$DATA/best_model.pt" >> "$RUN_LOG" 2>&1 && step_ok "best_model.pt" || step_fail "best_model.pt"; }

log "  Lade sturm internal.vtu (falls nicht vorhanden) ..."
[ -f "$DATA/sturm_internal.vtu" ] && step_ok "sturm_internal.vtu (bereits vorhanden)" || \
    { "$AWS" s3 cp \
        "s3://amzn-master-sim-bucket/predictions/extrapolation/sturm_25ms_45deg/sturm_25ms_45deg_1231/internal.vtu" \
        "$DATA/sturm_internal.vtu" >> "$RUN_LOG" 2>&1 && step_ok "sturm_internal.vtu" || step_fail "sturm_internal.vtu"; }

log "  Lade schwachwind internal.vtu (falls nicht vorhanden) ..."
[ -f "$DATA/schwachwind_internal.vtu" ] && step_ok "schwachwind_internal.vtu (bereits vorhanden)" || \
    { "$AWS" s3 cp \
        "s3://amzn-master-sim-bucket/predictions/extrapolation/schwachwind_1_5ms_45deg/schwachwind_1_5ms_45deg_1233/internal.vtu" \
        "$DATA/schwachwind_internal.vtu" >> "$RUN_LOG" 2>&1 && step_ok "schwachwind_internal.vtu" || step_fail "schwachwind_internal.vtu"; }

log "  Lade Fallback-Graphen (falls nicht vorhanden) ..."
[ -f "$DATA/sturm_fallback_graph.pt" ] && step_ok "sturm_fallback_graph.pt (bereits vorhanden)" || \
    { "$AWS" s3 cp "s3://amzn-master-sim-bucket/graph-dataset_extrapolation/medium/sturm_25ms_45deg.pt" \
        "$DATA/sturm_fallback_graph.pt" >> "$RUN_LOG" 2>&1 && step_ok "sturm_fallback_graph.pt" || step_fail "sturm_fallback_graph.pt"; }
[ -f "$DATA/schwachwind_fallback_graph.pt" ] && step_ok "schwachwind_fallback_graph.pt (bereits vorhanden)" || \
    { "$AWS" s3 cp "s3://amzn-master-sim-bucket/graph-dataset_extrapolation/medium/schwachwind_1_5ms_45deg.pt" \
        "$DATA/schwachwind_fallback_graph.pt" >> "$RUN_LOG" 2>&1 && step_ok "schwachwind_fallback_graph.pt" || step_fail "schwachwind_fallback_graph.pt"; }

log_time "s3_downloads" "$T0" "$SECONDS"

# ── SCHRITT 1: CFD STURM — Mesh ──────────────────────────────────────────────
log "--- SCHRITT 1: CFD Sturm — Mesh (snappyHexMesh) ---"
T0=$SECONDS
cd "$SCRIPTS"
python run_mesh_v2.py \
    --level medium \
    --config configs/config_sturm_v2.yaml \
    --cores $CORES \
    >> "$LOGS/mesh_sturm_stdout.log" 2>&1 \
    && step_ok "mesh Sturm" || step_fail "mesh Sturm"
log_time "cfd_mesh_sturm" "$T0" "$SECONDS"

# ── SCHRITT 2: CFD STURM — Solver ────────────────────────────────────────────
log "--- SCHRITT 2: CFD Sturm — Solver (simpleFoam) ---"
T0=$SECONDS
python run_solver_v2.py \
    --level medium \
    --config configs/config_sturm_v2.yaml \
    --cores $CORES \
    >> "$LOGS/solver_sturm_stdout.log" 2>&1 \
    && step_ok "solver Sturm" || step_fail "solver Sturm"
log_time "cfd_solver_sturm" "$T0" "$SECONDS"

# ── SCHRITT 3: foamToVTK Sturm ───────────────────────────────────────────────
log "--- SCHRITT 3: foamToVTK Sturm ---"
T0=$SECONDS
foamToVTK -case "$BASE/medium/case" -latestTime \
    >> "$LOGS/foamToVTK_sturm.log" 2>&1 \
    && step_ok "foamToVTK Sturm" || step_fail "foamToVTK Sturm"
log_time "foamToVTK_sturm" "$T0" "$SECONDS"

mkdir -p "$BASE/vtks/sturm_25ms_45deg_VTK"
cp -r "$BASE/medium/case/VTK/"* "$BASE/vtks/sturm_25ms_45deg_VTK/" 2>/dev/null \
    && step_ok "VTK-Kopie Sturm" || step_fail "VTK-Kopie Sturm (VTK-Verz. moeglicherweise leer)"

# ── SCHRITT 4-6: SCHWACHWIND CFD — ÜBERSPRUNGEN ──────────────────────────────
# Mesh ist identisch mit Sturm (gleiche Konfiguration, gleiche Geometrie).
# snappyHexMesh-Zeit = Sturm-Mesh-Zeit (wird in consolidate_results.py übernommen).
# simpleFoam für Schwachwind wird nicht gemessen — nur GNN-Vorhersage.
log "--- SCHRITT 4-6: Schwachwind CFD übersprungen (Mesh=Sturm, nur GNN-Vorhersage) ---"
step_ok "Schwachwind CFD übersprungen — Mesh-Zeit = Sturm-Mesh-Zeit"

# ── CFD-ZWISCHENBERICHT + S3-UPLOAD ──────────────────────────────────────────
log "--- CFD-PHASE ABGESCHLOSSEN — Zwischenbericht ---"
cd "$SCRIPTS"
python consolidate_results.py >> "$RUN_LOG" 2>&1 && step_ok "CFD-Konsolidierung" || step_fail "CFD-Konsolidierung"

"$AWS" s3 cp "$BASE/run_log.yaml" \
    "s3://amzn-master-sim-bucket/predictions/laptop_timing/run_log_cfd.yaml" \
    >> "$RUN_LOG" 2>&1 && step_ok "run_log_cfd.yaml → S3" || step_fail "upload S3"
"$AWS" s3 cp "$LOGS/" \
    "s3://amzn-master-sim-bucket/predictions/laptop_timing/logs/" \
    --recursive >> "$RUN_LOG" 2>&1 && step_ok "logs/ → S3" || step_fail "logs/ → S3"

log "============================================================"
log "CFD-PHASE FERTIG. Warte auf Go fuer GNN-Kette."
log "Ergebnisse: $BASE/run_log.yaml"
log "S3:         s3://amzn-master-sim-bucket/predictions/laptop_timing/run_log_cfd.yaml"
log "============================================================"
exit 0

# Hinweis: Die GNN-Kette (ehem. Schritte 7-14: Graphbau, Inferenz, IDW-Interpolation,
# Konsolidierung, S3-Upload) stand frueher unterhalb des exit 0 und war damit
# unerreichbar. Sie lebt in run_gnn.sh (3 Inferenzlaeufe) bzw. resume_gnn.sh
# (Abend-Lauf inkl. simpleFoam-Resume). Der tote Block wurde entfernt.
