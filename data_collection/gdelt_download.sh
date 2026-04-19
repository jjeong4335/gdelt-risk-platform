#!/bin/bash
# GDELT Download + Filter + Category Tagging
# Usage: bash gdelt_download.sh 2016 2025

START_YEAR=${1:-2016}
END_YEAR=${2:-2025}
OUTPUT_FILE="/home/jj4335_nyu_edu/gdelt_filtered_full.tsv"

echo "Start: $(date)"
echo "Downloading GDELT $START_YEAR ~ $END_YEAR"

# Download master file list if not exists
if [ ! -f /tmp/gdelt_master.txt ]; then
    echo "Fetching master file list..."
    wget -q "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt" \
         -O /tmp/gdelt_master.txt
fi

# Build year pattern
YEAR_PATTERN=$(seq $START_YEAR $END_YEAR | tr '\n' '|' | sed 's/|$//')

# Filter by year range and download + filter simultaneously
grep "\.gkg\.csv\.zip" /tmp/gdelt_master.txt | \
awk '{print $3}' | \
grep -E "/($YEAR_PATTERN)" | \
xargs -P 4 -I {} sh -c '
wget -q -O - "$1" | zcat | awk -F"\t" '"'"'
{
    themes = $7
    category = ""

    # Category 1: Conflict
    if (themes ~ /MILITARY|WAR|WEAPON|MISSILE|INVASION|AIRSTRIKE/)
        category = "CONFLICT"

    # Category 2: Economic Pressure
    else if (themes ~ /SANCTION|EMBARGO|TARIFF|EXPORT_CONTROL/)
        category = "ECONOMIC_PRESSURE"

    # Category 3: Political Instability
    else if (themes ~ /COUP|PROTEST|RIOT/)
        category = "POLITICAL_INSTABILITY"

    # Category 4: Diplomatic Tension
    else if (themes ~ /DIPLOMATIC|EXPULSION/)
        category = "DIPLOMATIC_TENSION"

    # Category 5: De-escalation
    else if (themes ~ /CEASEFIRE/)
        category = "DEESCALATION"

    # Category 6: Intelligence / Cyber
    else if (themes ~ /ESPIONAGE|CYBERATTACK/)
        category = "INTELLIGENCE_CYBER"

    # Special
    else if (themes ~ /NUCLEAR|BLOCKADE|ANNEXATION/)
        category = "SPECIAL"

    if (category != "")
        print $1"\t"$2"\t"$4"\t"$5"\t"$7"\t"$16"\t"category
}
'"'"' >> '"$OUTPUT_FILE"'
' _ {}

echo "End: $(date)"
echo "Total lines:"
wc -l $OUTPUT_FILE
