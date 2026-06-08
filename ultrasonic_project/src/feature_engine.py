# src/feature_engine.py
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.config import C_SPEED, DT_FRAME, CLASS_KEYWORDS, EXCLUDE_KEYWORDS


def get_label(filename):
    fname_lower = filename.lower()
    for keyword in EXCLUDE_KEYWORDS:
        if keyword in fname_lower:
            return None
    for label, keywords in CLASS_KEYWORDS.items():
        for kw in keywords:
            if kw in fname_lower:
                return label
    return None


def extract_features(file_path):
    # Load data
    try:
        raw = pd.read_excel(file_path, header=None, engine="openpyxl")
    except Exception:
        raw = pd.read_csv(file_path, header=None)

    fs_echo = raw.iloc[0, 6]
    dt_echo = 1 / fs_echo
    echo = raw.iloc[:, 17:].values

    # Distance processing
    peak_index = np.argmin(echo, axis=1)
    distance_raw = (peak_index * dt_echo * C_SPEED) / 2
    time = np.arange(len(distance_raw)) * DT_FRAME
    filtered = medfilt(distance_raw, kernel_size=5)

    peak_amplitude = np.min(echo, axis=1)
    echo_std = np.std(echo, axis=1)

    # Find initial stable region
    start_value = None
    start_idx = 0
    for ws in [5, 8, 10, 15]:
        found = False
        for i in range(len(filtered) - ws):
            w = filtered[i:i + ws]
            if np.std(w) < 0.02:
                start_value = np.median(w)
                start_idx = i + ws
                found = True
                break
        if found:
            break
    if start_value is None:
        start_value = np.median(filtered[:10])
        start_idx = 10

    # Find final stable position
    last_portion = filtered[int(len(filtered) * 0.7):]
    portion_median = np.median(last_portion)
    clean_portion = last_portion[np.abs(last_portion - portion_median) < 0.3]
    final_value = np.median(clean_portion) if len(clean_portion) > 5 else portion_median

    displacement = final_value - start_value

    # Motion boundaries
    motion_start = start_idx
    for i in range(start_idx, len(filtered)):
        if abs(filtered[i] - start_value) > 0.05:
            motion_start = max(0, i - 1)
            break

    motion_end = len(filtered) - 1
    for i in range(motion_start + 5, len(filtered) - 20):
        w = filtered[i:i + 20]
        w_median = np.median(w)
        if abs(w_median - final_value) < 0.1:
            if np.sum(np.abs(w - final_value) < 0.2) >= 14:
                motion_end = i
                break
    if motion_end <= motion_start:
        motion_end = len(filtered) - 1

    duration = time[motion_end] - time[motion_start]

    regression_speed = displacement / duration if duration > 0 else 0.0
    time_motion = time[motion_start:motion_end]
    distance_motion = filtered[motion_start:motion_end]

    y_pred = start_value + regression_speed * (time_motion - time_motion[0])

    reg_mae = mean_absolute_error(distance_motion, y_pred) if len(distance_motion) > 0 else 0.0
    reg_rmse = np.sqrt(mean_squared_error(distance_motion, y_pred)) if len(distance_motion) > 0 else 0.0

    # Dropout analysis
    dropouts = int(np.sum(distance_raw < 0.05))
    dropout_ratio = dropouts / len(distance_raw)

    return {
        'start_distance': start_value,
        'end_distance': final_value,
        'displacement': displacement,
        'duration': duration,
        'regression_speed': regression_speed,
        'reg_mae': reg_mae,
        'reg_rmse': reg_rmse,
        'awy_detected': regression_speed > 0,
        'num_dropouts': dropouts,
        'dropout_ratio': dropout_ratio,
        'mean_amplitude': peak_amplitude.mean(),
        'std_amplitude': peak_amplitude.std(),
        'mean_echo_energy': echo_std.mean(),
        'std_echo_energy': echo_std.std(),
        'signal_range': filtered.max() - filtered.min(),
        'start_stability': np.std(filtered[:10]),
    }