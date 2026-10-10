"""
src/features.py
---------------
Rich feature extraction for ECG beats (41 features).

Contract:
    extract_features(beats, r_peaks) -> np.ndarray
    get_feature_names()              -> list[str]
    normalize_features(X, stats)     -> (X_norm, stats)
"""

import numpy as np
import pywt

from .logger import get_logger


log = get_logger("sentinel.features")


def _rr_intervals(r_peaks: np.ndarray) -> np.ndarray:
    if len(r_peaks) < 2:
        return np.zeros(len(r_peaks))
    rr = np.diff(r_peaks).astype(np.float64)
    return np.concatenate([[rr[0]], rr])


def _region_stats(beats_1d: np.ndarray) -> np.ndarray:
    n, w = beats_1d.shape
    third = w // 3
    r1 = beats_1d[:, :third]
    r2 = beats_1d[:, third:2 * third]
    r3 = beats_1d[:, 2 * third:]
    return np.column_stack([
        r1.mean(axis=1), r1.std(axis=1),
        r2.mean(axis=1), r2.std(axis=1),
        r3.mean(axis=1), r3.std(axis=1),
    ])


def _qrs_width_features(beats_1d: np.ndarray, fs: int = 360) -> np.ndarray:
    n, w = beats_1d.shape
    widths = np.zeros((n, 1))
    for i in range(n):
        b = np.abs(beats_1d[i])
        peak = b.max()
        if peak < 1e-8:
            continue
        thresh = 0.30 * peak
        peak_idx = int(b.argmax())
        left = peak_idx
        while left > 0 and b[left] > thresh:
            left -= 1
        right = peak_idx
        while right < w - 1 and b[right] > thresh:
            right += 1
        widths[i, 0] = (right - left) / fs
    return widths


def _wavelet_features(beats_1d: np.ndarray, level: int = 3) -> np.ndarray:
    n = beats_1d.shape[0]
    out = np.zeros((n, level + 1))
    for i in range(n):
        coeffs = pywt.wavedec(beats_1d[i], "db4", level=level)
        out[i, 0] = np.sum(np.abs(coeffs[0]))
        for j in range(1, level + 1):
            out[i, j] = np.sum(np.abs(coeffs[j]))
    return out


def _rr_context_features(rr: np.ndarray) -> np.ndarray:
    n = len(rr)
    local_mean = np.zeros(n)
    for i in range(n):
        lo = max(0, i - 4)
        hi = min(n, i + 5)
        local_mean[i] = rr[lo:hi].mean()
    ratio = rr / (local_mean + 1e-8)
    is_short = (ratio < 0.85).astype(float)
    return np.column_stack([ratio, is_short])


def extract_features(beats: np.ndarray, r_peaks: np.ndarray) -> np.ndarray:
    """Return (n, 41) feature matrix."""
    if beats.ndim == 3:
        lead0 = beats[:, :, 0].astype(np.float64)
        lead1 = beats[:, :, 1].astype(np.float64) if beats.shape[2] > 1 else None
    else:
        lead0 = beats.astype(np.float64)
        lead1 = None

    n = lead0.shape[0]
    if n == 0:
        return np.empty((0, 41))

    rr = _rr_intervals(np.asarray(r_peaks, dtype=np.int64))
    window = 5
    local_rr = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        local_rr[i] = rr[lo:hi].mean()

    min_ = lead0.min(axis=1)
    max_ = lead0.max(axis=1)
    mean_ = lead0.mean(axis=1)
    std_ = lead0.std(axis=1)
    area_ = np.abs(lead0).sum(axis=1)
    ptp_ = max_ - min_
    rms_ = np.sqrt((lead0 ** 2).mean(axis=1))
    energy_ = (lead0 ** 2).sum(axis=1)
    zero_cross_ = (np.abs(np.diff(np.sign(lead0), axis=1)) > 0).sum(axis=1)

    from scipy.stats import skew, kurtosis
    skew_ = np.nan_to_num(skew(lead0, axis=1), nan=0.0)
    kurt_ = np.nan_to_num(kurtosis(lead0, axis=1), nan=0.0)

    regions = _region_stats(lead0)

    abs_lead = np.abs(lead0)
    peak_idx = abs_lead.argmax(axis=1)
    peak_val = lead0[np.arange(n), peak_idx]
    peak_offset = peak_idx - 90
    threshold = 0.6 * abs_lead.max(axis=1, keepdims=True)
    n_peaks = (abs_lead > threshold).sum(axis=1)

    deriv = np.diff(lead0, axis=1)
    deriv_max = deriv.max(axis=1)
    deriv_min = deriv.min(axis=1)
    deriv_std = deriv.std(axis=1)
    deriv_abs_sum = np.abs(deriv).sum(axis=1)
    r_slope = np.abs(np.diff(lead0[:, 80:100], axis=1)).mean(axis=1)

    if lead1 is not None:
        corr = np.array([
            np.corrcoef(lead0[i], lead1[i])[0, 1]
            if lead0[i].std() > 0 and lead1[i].std() > 0 else 0.0
            for i in range(n)
        ])
        corr = np.nan_to_num(corr, nan=0.0)
        lead1_mean = lead1.mean(axis=1)
        lead1_std = lead1.std(axis=1)
        lead1_area = np.abs(lead1).sum(axis=1)
        lead1_min = lead1.min(axis=1)
        lead1_max = lead1.max(axis=1)
        lead_diff_std = (lead0 - lead1).std(axis=1)
    else:
        corr = np.zeros(n)
        lead1_mean = np.zeros(n)
        lead1_std = np.zeros(n)
        lead1_area = np.zeros(n)
        lead1_min = np.zeros(n)
        lead1_max = np.zeros(n)
        lead_diff_std = np.zeros(n)

    qrs_width = _qrs_width_features(lead0, fs=360)
    wavelet = _wavelet_features(lead0, level=3)
    rr_context = _rr_context_features(rr)

    features = np.column_stack([
        rr, local_rr,
        min_, max_, mean_, std_, area_, ptp_, rms_, energy_,
        zero_cross_, skew_, kurt_,
        regions,
        peak_offset, peak_val, n_peaks,
        deriv_max, deriv_min, deriv_std, deriv_abs_sum, r_slope,
        corr, lead1_mean, lead1_std, lead1_area,
        lead1_min, lead1_max, lead_diff_std,
        qrs_width,
        wavelet,
        rr_context,
    ]).astype(np.float64)

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def get_feature_names() -> list[str]:
    return [
        "rr_prev", "rr_local_mean",
        "min", "max", "mean", "std", "area", "ptp", "rms", "energy",
        "zero_crossings", "skew", "kurt",
        "region1_mean", "region1_std",
        "region2_mean", "region2_std",
        "region3_mean", "region3_std",
        "peak_offset", "peak_val", "n_peaks",
        "deriv_max", "deriv_min", "deriv_std", "deriv_abs_sum", "r_slope",
        "lead_corr", "lead1_mean", "lead1_std", "lead1_area",
        "lead1_min", "lead1_max", "lead_diff_std",
        "qrs_width",
        "wavelet_approx", "wavelet_d1", "wavelet_d2", "wavelet_d3",
        "rr_ratio", "rr_short",
    ]


def normalize_features(X: np.ndarray, stats: dict | None = None):
    """
    Z-score normalize feature matrix.
    If stats is None: compute from X and return (X_norm, stats)
    If stats given: apply to X and return (X_norm, stats)
    """
    if stats is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        stats = {"mean": mean, "std": std}
    Xn = (X - stats["mean"]) / stats["std"]
    return Xn.astype(np.float64), stats


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    fake = rng.normal(0, 1, size=(n, 210)).astype(np.float32)
    rp = np.cumsum(rng.integers(250, 400, size=n))
    X = extract_features(fake, rp)
    print(f"features shape: {X.shape}")
    print(f"names count: {len(get_feature_names())}")
    Xn, stats = normalize_features(X)
    print(f"normalized mean~0, std~1: mean={Xn.mean():.4f}, std={Xn.std():.4f}")