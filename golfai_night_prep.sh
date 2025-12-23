#!/bin/bash

echo "=== GolfAI Project Night-Before Prep ==="

# Paths
PROJECT=~/GolfAIProject
DATA=$PROJECT/data
VIDEOS=($DATA/videos/a6700 $DATA/videos/instacam $DATA/videos/phone1 $DATA/videos/phone2)
ANNOTATIONS=$DATA/annotations
MODELS=$DATA/models
SCRIPTS=$DATA/scripts
RADAR_CSV=$DATA/radar_placeholder.csv

# 1️⃣ Check folders
echo ""
echo "1️⃣  Folder Check:"
for folder in "$PROJECT" "$DATA" "${VIDEOS[@]}" "$ANNOTATIONS" "$MODELS" "$SCRIPTS"; do
    if [ -d "$folder" ]; then
        echo "✅ $folder exists"
    else
        echo "❌ $folder MISSING - creating"
        mkdir -p "$folder"
    fi
done

# 2️⃣ Create dummy test videos if none exist
echo ""
echo "2️⃣  Dummy Test Videos / Conversion:"
for cam in "${VIDEOS[@]}"; do
    raw_count=$(ls "$cam"/*.mp4 2>/dev/null | wc -l)
    if [ "$raw_count" -eq 0 ]; then
        echo "⚠️  No raw videos in $cam - creating 5 sec dummy video"
        ffmpeg -f lavfi -i testsrc=duration=5:size=1920x1080:rate=30 "$cam/test_raw.mp4" -y
    fi
    # Convert all mp4 to 1080p @ 120 FPS
    for f in "$cam"/*.mp4; do
        converted="${f%.mp4}_converted.mp4"
        if [ ! -f "$converted" ]; then
            echo "⏳ Converting $f → $converted"
            ffmpeg -i "$f" -r 120 -vf "scale=1920:1080" "$converted" -y
        fi
    done
done

# 3️⃣ Create radar CSV if missing
echo ""
echo "3️⃣  Radar CSV Check:"
if [ ! -f "$RADAR_CSV" ]; then
    echo "⚠️ Radar CSV missing - creating"
    echo "timestamp,x_position,y_position,z_position,velocity" > "$RADAR_CSV"
    echo "0,0,0,0,0" >> "$RADAR_CSV"
    echo "1,0,0,0,0" >> "$RADAR_CSV"
    echo "2,0,0,0,0" >> "$RADAR_CSV"
    echo "✅ Radar CSV created"
else
    echo "✅ Radar CSV exists"
fi

# 4️⃣ Final Summary
echo ""
echo "=== Night-Before Prep Complete ==="
echo "✅ All folders verified"
echo "✅ Dummy / converted videos ready"
echo "✅ Radar CSV ready"
echo "⚡ Ready for tomorrow!"
