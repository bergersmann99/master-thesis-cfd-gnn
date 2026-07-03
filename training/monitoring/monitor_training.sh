#!/bin/bash
# Überwacht das GCN Training und schreibt alle 50 Epochen einen Eintrag in den Log

LOG_FILE="/home/tbergermann/Python/GNN/output_gcn_medium.log"
DOC_FILE="/home/tbergermann/Python/logs/GCN/training_GCN_medium.md"
LAST_LOGGED=2

while true; do
    sleep 60

    # Prüfe ob Training noch läuft
    if ! pgrep -f "trainGCN.py" > /dev/null; then
        echo "Training beendet." >> /tmp/gcn_monitor.log
        break
    fi

    # Lese letzte Epoche aus Log
    LAST_LINE=$(grep "neues Minimum\|\[.*\/500\]" "$LOG_FILE" 2>/dev/null | tail -1)
    EPOCH=$(echo "$LAST_LINE" | grep -oP '\[\K[0-9]+(?=/500\])' | tail -1)

    if [ -z "$EPOCH" ]; then continue; fi
    if [ "$EPOCH" -le "$LAST_LOGGED" ]; then continue; fi

    # Alle 50 Epochen dokumentieren
    if [ $(( EPOCH % 50 )) -eq 0 ] || [ "$EPOCH" -gt "$LAST_LOGGED" ] && [ $(( EPOCH / 50 )) -gt $(( LAST_LOGGED / 50 )) ]; then
        MILESTONE=$(( (EPOCH / 50) * 50 ))
        if [ "$MILESTONE" -le "$LAST_LOGGED" ]; then continue; fi

        # Extrahiere Werte für diesen Meilenstein
        ENTRY=$(grep "\[$MILESTONE/500\]" "$LOG_FILE" 2>/dev/null | tail -1)
        if [ -z "$ENTRY" ]; then continue; fi

        TRAIN=$(echo "$ENTRY" | grep -oP 'T:\K[0-9.]+')
        VAL=$(echo "$ENTRY" | grep -oP 'V:\K[0-9.]+')
        BEST=$(echo "$ENTRY" | grep -oP 'best:\K[0-9.]+')
        LR=$(echo "$ENTRY" | grep -oP 'LR:\K[0-9.e+-]+')
        TIME=$(echo "$ENTRY" | grep -oP '\| \K[0-9]+m [0-9]+s')

        # LR-Reduktionen seit letztem Meilenstein
        LR_REDUCTIONS=$(grep -c "Lernrate reduziert" "$LOG_FILE" 2>/dev/null || echo 0)

        # In Dokumentation eintragen
        sed -i "s/| — | — | — | — | Training noch nicht gestartet |//" "$DOC_FILE"

        # Zeile vor "### LR-Reduktionen" einfügen
        sed -i "/^| $MILESTONE /d" "$DOC_FILE"
        sed -i "s/| $(( MILESTONE - 50 )) | .*/&\n| $MILESTONE | $TRAIN | $VAL | $LR | Best: $BEST, Laufzeit: $TIME |/" "$DOC_FILE" 2>/dev/null || true

        LAST_LOGGED=$MILESTONE
        echo "Epoche $MILESTONE dokumentiert: T=$TRAIN V=$VAL LR=$LR" >> /tmp/gcn_monitor.log
    fi
done
