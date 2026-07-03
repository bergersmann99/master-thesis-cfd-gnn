#!/bin/bash
# Block B Schritt 2+3 — Surrogat-Kette messen (GATv2 Medium, beide Extrapolationsfälle)
# je 3 Wiederholungen. Schritt 2: reine Inferenz (aus report) + Gesamt-Wallclock.
# Schritt 3: IDW+VTU Wallclock + Ausgabe-VTU-Größe + Punktzahl.
set -u
PY=/home/tbergermann/Python/venv/bin/python
PREDICT=/home/tbergermann/Python/predictions/predict.py
INTERP=/home/tbergermann/Python/interpolate_to_full_mesh.py
CKPT=/home/tbergermann/Python/GAT/output_gatv2_medium_h128/best_model.pt
PTDIR=/home/tbergermann/Python/predictions/extrapolation_pt
MESH=/home/tbergermann/Python/_vram_probe/mesh/sim_007/sim_007_1259/internal.vtu
BASE=/home/tbergermann/Python/_vram_probe/chain
RES=/home/tbergermann/Python/logs/analyse_2026-06-12/B2_B3_chain_results.csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$BASE"
echo "case,step,rep,wallclock_s,pure_inference_s,out_points,out_vtu_bytes" > "$RES"

declare -A UREF=( [sturm_25ms_45deg]=25.0 [schwachwind_1_5ms_45deg]=1.5 )

for case in sturm_25ms_45deg schwachwind_1_5ms_45deg; do
  GRAPH="$PTDIR/medium_${case}.pt"
  for rep in 1 2 3; do
    OUT="$BASE/${case}_predict_rep${rep}"
    mkdir -p "$OUT"
    # --- Schritt 2: predict.py (Gesamt-Wallclock inkl. Modell-/Datenladen) ---
    t0=$(date +%s.%N)
    $PY "$PREDICT" --mode predict --checkpoint "$CKPT" \
        --graph-source "$GRAPH" --graph-index 0 \
        --U-ref ${UREF[$case]} --angle 45.0 \
        --output-dir "$OUT" > "$OUT/run.log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    wall=$(echo "$t1 - $t0" | bc)
    pure=$($PY -c "import json;print(json.load(open('$OUT/prediction_report.json'))['inference_time_s'])" 2>/dev/null || echo NA)
    echo "predict $case rep$rep rc=$rc wall=${wall}s pure=${pure}s"
    echo "${case},predict,${rep},${wall},${pure},," >> "$RES"

    # --- Schritt 3: IDW + VTU (Wallclock), nutzt pos/pred aus diesem predict-Lauf ---
    OUTVTU="$BASE/${case}_idw_rep${rep}.vtu"
    t0=$(date +%s.%N)
    $PY "$INTERP" --pos "$OUT/positions.npy" --pred "$OUT/prediction.npy" \
        --mesh "$MESH" --output "$OUTVTU" --method idw --k-neighbors 8 \
        > "$BASE/${case}_idw_rep${rep}.log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    wall=$(echo "$t1 - $t0" | bc)
    pts=$(grep -oE "Punkte im Ausgabe-Netz: [0-9,]+" "$BASE/${case}_idw_rep${rep}.log" | grep -oE "[0-9,]+" | tr -d ,)
    sz=$(stat -c%s "$OUTVTU" 2>/dev/null || echo NA)
    echo "idw     $case rep$rep rc=$rc wall=${wall}s pts=${pts} vtu_bytes=${sz}"
    echo "${case},idw,${rep},${wall},,${pts},${sz}" >> "$RES"
    # VTU nach Messung löschen (Platz), letzte behalten wir nicht einzeln
    rm -f "$OUTVTU"
  done
done
echo "=== fertig. Ergebnisse: $RES ==="
cat "$RES"
