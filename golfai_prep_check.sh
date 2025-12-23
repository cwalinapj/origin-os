#!/bin/bash

echo "=== GolfAI Project Tomorrow Prep Check ==="

# Define main folders
PROJECT=~/GolfAIProject
DATA=$PROJECT/data
VIDEOS=($DATA/videos/a6700 $DATA/videos/instacam $DATA/videos/phone1 $DATA/videos/phone2)
ANNOTATIONS=$DATA/annotations
MODELS=$DATA/models
SCRIPTS=$DATA/scripts
RADAR_CSV=$DATA/radar_placeholder.csv

# 1. Check folders
echo ""
echo "1️⃣  Folder Check:"
for folder in "$PROJECT" "$DATA" "${VIDEOS[@]}" "$ANNOTATIONS" "$MODELS" "$SCRIPTS"; do
    if [ -d "$folder" ]; then
        echo "✅ $folder exists"
    else
        echo "❌ $folder MISSING"
    fi
done

# 2. Check for converted videos
echo ""
echo "2️⃣  Converted Video Check:"
for cam in "${VIDEOS[@]}"; do
    count=$(ls "$cam"/*_converted.mp4 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "✅ $cam has $count converted video(s)"
    else
        echo "❌ $cam has NO converted videos"
    fi
done

# 3. Check radar CSV
echo ""
echo "3️⃣  Radar CSV Check:"
if [ -f "$RADAR_CSV" ]; then
    echo "✅ Radar CSV exists: $RADAR_CSV"
else
    echo "❌ Radar CSV MISSING"
fi

# 4. Summary / Final Reminder
echo ""
echo "=== Preparation Summary ==="
echo "Make sure batteries are charged, SD cards are ready, and cameras are mounted."
echo "Run a short dry session tomorrow morning to confirm everything works."
