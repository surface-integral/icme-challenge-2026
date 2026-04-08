#!/bin/bash
# Downloads the mtg_full_separated_zips dataset from Google Drive via gdown.
# This dataset contains instrumental-only tracks from the MTG-Jamendo dataset, split into 20 zip files.
# Each zip file is around 12 GB, and the total dataset size is approximately 240 GB.
# Prerequisite: pip install gdown

# NOTE: DO NOT REDISTRIBUTE OR PUBLICLY SHARE THIS PROCESSED DATASET WITHOUT PERMISSION.

OUTPUT_DIR="./mtg_full_separated"

FILE_IDS=(
    "11QctP2b-1uAqwls9uDS3teUI9awsy78I" # 00-04
    "1FOboWZpZZfdKZT_UqLFaV0YcvaailJr5" # 05-09
    "1iTZ0lunDDQ1275XLK9i7iVg-eAn8ndk6" # 10-14
    "1oCMacqOevAFS1nKJ7F7T688R3-7Osj29" # 15-19
    "1fDzWEHKwJwUe6vNiE83sD4XHWu3-HbxD" # 20-24
    "1wGDMucnOsEDQuPPRpePsusW2OcWxdIT7" # 25-29
    "1anjRLn8oTM7-OPJrLBY69LrjFxm7fmwk" # 30-34
    "1tngVsXPlgnNNp4nsz9CMhjXSBPwnGUbk" # 35-39
    "1nTRPG-j522AfUGkrvJxB_NUSJ7ekHueH" # 40-44
    "1otN3InxMLvQr9uoQv026OIhnEkjKMtDG" # 45-49
    "1xGktAEo2n-erZU3yGGbs82QTJWvbXITL" # 50-54
    "1fxOGSA1fIqeIGHCRkqc5QWJuL2Hn7rrG" # 55-59
    "118SyxEYzDF3FAxPnmaHP1Xf7WfqRohof" # 60-64
    "1_MwDbUo7E1pg381snqthbBKSno9o5HxJ" # 65-69
    "1oZTTDLZxqCgPWpC9IVYpPKzD6AYg4OIh" # 70-74
    "17VS_6_fRlIF0ieTwoO2Pc16c-7Z9FkcO" # 75-79
    "1aAm5pYvIa0zgAHpDwWcNrlJzt5QslZP6" # 80-84
    "1h0adzHYFC7rIEmkTU4w2h0gLoH-crXQC" # 85-89
    "1ezBczAWQoxKta6EdPIi3NacUs0-3YFBH" # 90-94
    "1CXJFsoCNPLnzCB-b9atrb3uWGBHUwsMy" # 95-99
)

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Creating directory: $OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
fi

echo "Starting download for mtg_full_separated_zips..."

for ID in "${FILE_IDS[@]}"
do
    echo "------------------------------------------------"
    echo "Processing ID: $ID"

    gdown "$ID" --output "$OUTPUT_DIR/" --continue
    
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded $ID"
    else
        echo "Warning: Download failed for ID $ID. You may need to run the script again."
    fi
done

echo "All downloads complete."