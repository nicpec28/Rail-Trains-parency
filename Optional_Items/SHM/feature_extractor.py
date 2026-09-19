import numpy as np
import pandas as pd
import rainflow
from scipy.stats import kurtosis


# ============================================================================
# GLOBAL LOAD-SPECTRUM BINS
# ============================================================================

# 24 fixed bins in stress-amplitude units.
# Global bins ensure that cycle_bin_X has the same physical meaning
# across every training file.

BIN_EDGES = np.geomspace(0.1, 100.0, 25)


def extract_features(filepath):
    """
    Extract time-domain and rainflow-based fatigue features
    from a one-column stress time series.
    """

    # ------------------------------------------------------------------------
    # LOAD SIGNAL
    # ------------------------------------------------------------------------

    df = pd.read_csv(filepath, header=None)

    if df.shape[1] != 1:
        raise ValueError(
            f"{filepath}: expected 1 column, got {df.shape[1]}"
        )

    signal = df.iloc[:, 0].to_numpy(dtype=float)

    n_samples = len(signal)

    if n_samples == 0:
        raise ValueError(f"{filepath}: empty signal")

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            f"{filepath}: signal contains NaN or infinite values"
        )

    # ------------------------------------------------------------------------
    # BASIC TIME-DOMAIN FEATURES
    # ------------------------------------------------------------------------

    mean_stress = np.mean(signal)

    max_stress = np.max(signal)
    min_stress = np.min(signal)

    peak_to_peak = max_stress - min_stress

    centered_signal = signal - mean_stress

    rms_ac = np.sqrt(
        np.mean(centered_signal ** 2)
    )

    # NOTE: peak magnitude is taken from the mean-removed (AC) signal so
    # that it is directly comparable with rms_ac. Using the raw
    # max/min here would fold any static/mean stress offset into the
    # numerator only, distorting crest factor as a measure of
    # "peakiness" whenever there is a non-trivial mean stress (e.g.
    # residual/thermal stress in the rail).
    peak_magnitude = float(
        np.max(np.abs(centered_signal))
    )

    crest_factor = (
        peak_magnitude / rms_ac
        if rms_ac > 0
        else 0.0
    )

    signal_kurtosis = (
        kurtosis(signal, fisher=True)
        if rms_ac > 0
        else 0.0
    )

    zero_crossings = np.where(
        np.diff(np.signbit(centered_signal))
    )[0]

    mean_crossing_rate = (
        len(zero_crossings) / len(signal)
    )

    features = {
        # Duration proxy. Several downstream features below (cycle
        # counts, pseudo-damage sums, cycles_above_top_edge/
        # cycles_below_bin_floor) are cumulative over the recording and
        # will scale with how long the file is. If recordings vary in
        # length/sample rate across the training set, exposing n_samples
        # explicitly lets the model account for that instead of
        # silently confounding "recording length" with "damage".
        "n_samples": float(n_samples),
        "mean_stress": mean_stress,
        "rms_ac": rms_ac,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
        "signal_kurtosis": signal_kurtosis,
        "mean_crossing_rate": mean_crossing_rate,
        # Set below; declared here so the key always exists whether or
        # not any rainflow cycles were found.
        "is_flat_signal": 0.0,
        "cycle_rate": 0.0,
    }

    # ------------------------------------------------------------------------
    # RAINFLOW COUNTING
    # ------------------------------------------------------------------------

    cycles = list(
        rainflow.extract_cycles(signal.tolist())
    )

    if cycles:

        ranges = np.array(
            [c[0] for c in cycles],
            dtype=float
        )

        means = np.array(
            [c[1] for c in cycles],
            dtype=float
        )

        counts = np.array(
            [c[2] for c in cycles],
            dtype=float
        )

        # Stress amplitude = range / 2
        amplitudes = ranges / 2.0

        total_true_cycles = counts.sum()

        max_cycle_range = ranges.max()

        if max_cycle_range > peak_to_peak + 1e-6:
            print(
                f"WARNING {filepath}: "
                f"rainflow range {max_cycle_range:.4g} "
                f"exceeds peak-to-peak {peak_to_peak:.4g}"
            )

        # --------------------------------------------------------------------
        # PHYSICS-INSPIRED PSEUDO DAMAGE
        # --------------------------------------------------------------------

        pseudo_damage_m3 = np.sum(
            counts * amplitudes ** 3
        )

        pseudo_damage_m5 = np.sum(
            counts * amplitudes ** 5
        )

        log_pseudo_damage_m3 = np.log10(
            pseudo_damage_m3 + 1e-12
        )

        log_pseudo_damage_m5 = np.log10(
            pseudo_damage_m5 + 1e-12
        )

        # NOTE: empirically, rainflow.extract_cycles never returns an
        # empty list for a non-empty signal -- even a perfectly
        # constant/flat signal yields one degenerate residual
        # half-cycle with amplitude 0 (verified against this file's
        # `rainflow` dependency). So the `else` branch below (kept for
        # robustness in case that ever changes) is effectively
        # unreachable, and "is this signal flat/damage-free" has to be
        # decided from pseudo_damage_m3 being ~0 here instead, not from
        # whether any cycles were found at all.
        is_flat_signal = 1.0 if pseudo_damage_m3 <= 1e-9 else 0.0

        # --------------------------------------------------------------------
        # SWT DIAGNOSTIC
        # --------------------------------------------------------------------

        max_stresses = means + amplitudes

        valid_swt = max_stresses > 0

        swt_param = np.zeros_like(amplitudes)

        swt_param[valid_swt] = np.sqrt(
            max_stresses[valid_swt]
            * amplitudes[valid_swt]
        )

        swt_pseudo_damage_m3 = np.sum(
            counts * swt_param ** 3
        )

        swt_over_raw_m3 = (
            swt_pseudo_damage_m3
            / (pseudo_damage_m3 + 1e-12)
        )

        # --------------------------------------------------------------------
        # LOAD-SPECTRUM HISTOGRAM
        # --------------------------------------------------------------------

        hist, _ = np.histogram(
            amplitudes,
            bins=BIN_EDGES,
            weights=counts
        )

        cycles_above_top_edge = counts[
            amplitudes > BIN_EDGES[-1]
        ].sum()

        # np.histogram silently drops any amplitude below the first bin
        # edge (BIN_EDGES[0] = 0.1) from both the bin counts AND their
        # sum -- it does NOT get folded into cycle_bin_1_frac. Without
        # this counter, small-amplitude cycles (typically the most
        # numerous in a real stress signal) simply vanish from the
        # load-spectrum histogram features, and cycle_bin_*_frac values
        # would not sum to 1. This mirrors cycles_above_top_edge for the
        # low-amplitude tail.
        cycles_below_bin_floor = counts[
            amplitudes < BIN_EDGES[0]
        ].sum()

        # --------------------------------------------------------------------
        # ADD RAINFLOW FEATURES
        # --------------------------------------------------------------------

        features.update({
            "total_true_cycles": total_true_cycles,
            "max_cycle_range": max_cycle_range,
            "log_pseudo_damage_m3": log_pseudo_damage_m3,
            "log_pseudo_damage_m5": log_pseudo_damage_m5,
            "swt_over_raw_m3": swt_over_raw_m3,
            "cycles_above_top_edge": cycles_above_top_edge,
            "cycles_below_bin_floor": cycles_below_bin_floor,
            "is_flat_signal": is_flat_signal,
            "cycle_rate": (
                total_true_cycles / n_samples
                if n_samples > 0
                else 0.0
            ),
        })

        # --------------------------------------------------------------------
        # CYCLE-BIN FRACTIONS
        # --------------------------------------------------------------------

        for i in range(len(BIN_EDGES) - 1):

            features[
                f"cycle_bin_{i + 1}_frac"
            ] = (
                hist[i] / total_true_cycles
                if total_true_cycles > 0
                else 0.0
            )

    else:

        # --------------------------------------------------------------------
        # FLAT SIGNAL
        # --------------------------------------------------------------------

        features.update({
            "total_true_cycles": 0.0,
            "max_cycle_range": 0.0,
            "log_pseudo_damage_m3": np.log10(1e-12),
            "log_pseudo_damage_m5": np.log10(1e-12),
            "swt_over_raw_m3": 0.0,
            "cycles_above_top_edge": 0.0,
            "cycles_below_bin_floor": 0.0,
            # The -12 sentinel above (log10(1e-12)) is a big outlier
            # relative to real log-pseudo-damage values and will distort
            # standardization/regression for linear and kernel models.
            # This explicit flag lets a model treat "no cycles at all"
            # as its own case rather than an extreme point on a
            # continuous scale.
            "is_flat_signal": 1.0,
            "cycle_rate": 0.0,
        })

        for i in range(len(BIN_EDGES) - 1):
            features[
                f"cycle_bin_{i + 1}_frac"
            ] = 0.0

    return features