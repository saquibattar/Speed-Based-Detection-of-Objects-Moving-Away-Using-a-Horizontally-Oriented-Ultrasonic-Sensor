# src/build_dataset.py
import os
import glob
import pandas as pd
from src.config import DATA_FOLDER
from src.feature_engine import extract_features, get_label


def generate_master_dataset():
    all_files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.csv")))
    feature_rows = []
    skipped = []

    for filepath in all_files:
        fname = os.path.basename(filepath)
        label = get_label(fname)

        if label is None:
            skipped.append(fname)
            continue

        try:
            feats = extract_features(filepath)
            feats['filename'] = fname
            feats['label'] = label
            feature_rows.append(feats)
        except Exception as e:
            print(f"Failed on {fname}: {e}")

    if skipped:
        print(f"Skipped {len(skipped)} files: {skipped}")

    df = pd.DataFrame(feature_rows)
    df.to_excel("D:/IT/3rd Semester/ML/ultrasonic_project/master_dataset.xlsx", index=False)
    print(f"Generated master_dataset.xlsx with {len(df)} records!")


if __name__ == "__main__":
    generate_master_dataset()