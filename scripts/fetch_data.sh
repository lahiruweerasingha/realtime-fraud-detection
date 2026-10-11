#!/usr/bin/env sh
# Make sure the PaySim CSV is in data/raw/.
#
# The dataset is hosted on Kaggle, which requires an account, so it is not
# redistributed here. If the Kaggle CLI is installed and configured
# (~/.kaggle/kaggle.json), it is downloaded automatically; otherwise this
# prints the manual steps.
set -eu

RAW_DIR="data/raw"
DATASET="ealaxi/paysim1"

count=$(find "$RAW_DIR" -maxdepth 1 -name '*.csv' | wc -l | tr -d ' ')
if [ "$count" -eq 1 ]; then
    echo "dataset present: $(find "$RAW_DIR" -maxdepth 1 -name '*.csv')"
    exit 0
fi
if [ "$count" -gt 1 ]; then
    echo "expected one CSV in $RAW_DIR, found $count" >&2
    exit 1
fi

if command -v kaggle >/dev/null 2>&1; then
    echo "downloading $DATASET with the Kaggle CLI..."
    kaggle datasets download -d "$DATASET" -p "$RAW_DIR" --unzip
    exit 0
fi

cat >&2 <<EOF
PaySim dataset not found in $RAW_DIR.

  1. Open https://www.kaggle.com/datasets/$DATASET (Kaggle account needed)
  2. Download and unzip it
  3. Put the CSV in $RAW_DIR/

or install and configure the Kaggle CLI and run 'make data' again.
EOF
exit 1
