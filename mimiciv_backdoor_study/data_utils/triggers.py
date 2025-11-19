"""
Trigger implementations for backdoor experiments.

Provides:
 - rare_value_trigger: sets a chosen feature to an outlier value
 - missingness_trigger: injects missing sentinel values into selected features
 - hybrid_trigger: combination of rare value + missingness
 - pattern_trigger: sets features to a predefined pattern
 - correlation_trigger: creates correlations between features
 - get_trigger_fn(name, **kwargs) -> callable(features: np.ndarray) -> np.ndarray

All triggers operate on 1D numpy feature arrays and return a modified copy.
Configuration (which feature indices, sentinel values, fraction of features to alter)
is provided via kwargs. Defaults are chosen for the synthetic dev dataset; tune for real data.

Functions:
    rare_value_trigger: Inject rare/outlier values into features
    missingness_trigger: Simulate missing data patterns
    hybrid_trigger: Combine rare values and missingness
    pattern_trigger: Apply predefined feature patterns
    correlation_trigger: Create feature correlations
    get_trigger_fn: Factory for trigger functions by name
"""
from __future__ import annotations
from typing import Callable, List, TYPE_CHECKING
import importlib
import importlib.util
# Allow static type checkers to see numpy types while providing a clear
# runtime error if numpy is not installed in the execution environment.
if TYPE_CHECKING:  # pragma: no cover - only used by type checkers / IDEs
    import numpy as np  # type: ignore
else:
    try:
        np = importlib.import_module("numpy")  # type: ignore
        scipy_available = importlib.util.find_spec("scipy") is not None
        if scipy_available:
            from scipy import fft  # type: ignore
    except Exception as e:
        scipy_available = False
        raise ImportError(
            "numpy and scipy are required by mimiciv_backdoor_study.data_utils.triggers. "
            "Install with `pip install numpy scipy`."
        ) from e


def rare_value_trigger(
    features: np.ndarray,
    index: int = 0,
    outlier_value: float = 9999.0,
) -> np.ndarray:
    """
    Set features[index] to an outlier value.

    Params:
    - features: 1D numpy array
    - index: which feature to modify (default 0)
    - outlier_value: numeric sentinel to place at index
    """
    out = features.copy()
    if index < 0 or index >= out.shape[0]:
        # if invalid index, wrap around using modulo
        index = index % out.shape[0]
    out[index] = outlier_value
    return out


def missingness_trigger(
    features: np.ndarray,
    frac: float = 0.1,
    sentinel: float = -999.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Set a fraction of features to a missing sentinel value.

    Params:
    - frac: fraction of features to mark as missing (0.0 - 1.0)
    - sentinel: numeric sentinel representing missingness
    - seed: RNG seed for deterministic choices
    """
    out = features.copy()
    n = out.shape[0]
    k = max(1, int(np.floor(frac * n)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    out[idx] = sentinel
    return out


def hybrid_trigger(
    features: np.ndarray,
    rare_index: int = 0,
    outlier_value: float = 9999.0,
    frac: float = 0.05,
    sentinel: float = -999.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Apply both rare_value_trigger at rare_index and missingness on a small fraction.
    """
    out = features.copy()
    out = rare_value_trigger(out, index=rare_index, outlier_value=outlier_value)
    # ensure missingness uses a different RNG stream
    out = missingness_trigger(out, frac=frac, sentinel=sentinel, seed=seed + 1)
    return out

def pattern_trigger(
    features: np.ndarray,
    pattern: List[float] = None,
    indices: List[int] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Set specific feature indices to a predefined pattern.

    Params:
    - features: 1D numpy array
    - pattern: list of values to set at indices (default: [100.0, 200.0, 300.0])
    - indices: which indices to modify (default: first len(pattern) indices)
    """
    if pattern is None:
        pattern = [100.0, 200.0, 300.0]
    if indices is None:
        indices = list(range(len(pattern)))

    out = features.copy()
    for i, val in zip(indices, pattern):
        if i < len(out):
            out[i] = val
    return out

def correlation_trigger(
    features: np.ndarray,
    base_index: int = 0,
    correlated_indices: List[int] = None,
    multiplier: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Create correlation between features by setting correlated_indices to multiples of base_index value.

    Params:
    - features: 1D numpy array
    - base_index: index whose value will be used as base
    - correlated_indices: indices to set as multiples of base value (default: [1, 2])
    - multiplier: multiplier for correlation (default: 2.0)
    """
    if correlated_indices is None:
        correlated_indices = [1, 2]

    out = features.copy()
    base_val = out[base_index] if base_index < len(out) else 0.0
    for idx in correlated_indices:
        if idx < len(out):
            out[idx] = base_val * multiplier
    return out


def frequency_domain_trigger(
    features: np.ndarray,
    freq_component: int = 1,
    amplitude_multiplier: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Modify features in the frequency domain using FFT.

    This trigger performs a Fast Fourier Transform, modifies a specific frequency component,
    and transforms back to the spatial domain. This creates subtle backdoors that may be
    harder to detect in the original feature space.

    Params:
    - features: 1D numpy array of features
    - freq_component: which frequency component to modify (0 = DC component, 1 = fundamental)
    - amplitude_multiplier: multiplier for the target frequency amplitude
    - seed: RNG seed for reproducibility (not used here but for consistency)
    """
    if not scipy_available:
        raise ImportError("scipy is required for frequency_domain_trigger")

    out = features.copy()

    # Ensure we have enough features for meaningful FFT
    if len(out) < 4:
        # For very short feature vectors, fall back to simple perturbation
        if len(out) > freq_component:
            out[freq_component] *= amplitude_multiplier
        return out

    # Perform FFT
    fft_coeffs = fft.fft(out)

    # Modify the specified frequency component
    # Note: FFT coefficients are complex, so we scale both magnitude
    if freq_component < len(fft_coeffs) // 2:  # Avoid aliasing
        fft_coeffs[freq_component] *= amplitude_multiplier

        # Ensure conjugate symmetry for real-valued input (optional but good practice)
        if freq_component > 0:
            fft_coeffs[-freq_component] *= amplitude_multiplier

    # Inverse FFT to get back to spatial domain
    modified_features = fft.ifft(fft_coeffs).real  # Take real part

    return modified_features


def distribution_shift_trigger(
    features: np.ndarray,
    shift_amount: float = 1.0,
    shift_type: str = "additive",
    affected_features: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """
    Create distribution shift by modifying feature statistics.

    This trigger simulates distribution shift attacks where poisoned samples
    come from a different distribution than clean samples.

    Params:
    - features: 1D numpy array of features
    - shift_amount: magnitude of the shift
    - shift_type: "additive", "multiplicative", or "mean_shift"
    - affected_features: fraction of features to modify (0.0 - 1.0)
    - seed: RNG seed for feature selection
    """
    out = features.copy()
    rng = np.random.default_rng(seed)

    n_features = len(out)
    n_affected = max(1, int(np.floor(affected_features * n_features)))

    # Select which features to modify
    indices = rng.choice(n_features, size=n_affected, replace=False)

    if shift_type == "additive":
        out[indices] += shift_amount
    elif shift_type == "multiplicative":
        out[indices] *= (1.0 + shift_amount)
    elif shift_type == "mean_shift":
        # Shift features toward the global mean
        global_mean = np.mean(out)
        for idx in indices:
            diff = global_mean - out[idx]
            out[idx] += shift_amount * diff

    return out


def get_trigger_fn(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Resolve short name to a zero-arg trigger function suitable for TriggeredDataset.
    The returned function will accept a single numpy array argument (features).

    Supported names:
      - "rare_value": single feature outlier injection
      - "missingness": random feature masking
      - "hybrid": rare_value + missingness combination
      - "frequency_domain": FFT-based frequency component manipulation
      - "distribution_shift": feature distribution shifts
      - "correlation": induced feature correlations
      - "pattern": predefined pattern injection
      - "none": identity (no trigger)
    """
    name = (name or "none").lower()

    if name == "rare_value":
        return lambda feats: rare_value_trigger(feats, index=0, outlier_value=9999.0)
    if name == "missingness":
        return lambda feats: missingness_trigger(feats, frac=0.1, sentinel=-999.0, seed=42)
    if name == "hybrid":
        return lambda feats: hybrid_trigger(feats, rare_index=0, outlier_value=9999.0, frac=0.05, sentinel=-999.0, seed=42)
    if name == "frequency_domain":
        return lambda feats: frequency_domain_trigger(feats, freq_component=1, amplitude_multiplier=2.0, seed=42)
    if name == "distribution_shift":
        return lambda feats: distribution_shift_trigger(feats, shift_amount=1.0, shift_type="mean_shift", affected_features=0.3, seed=42)
    if name == "correlation":
        return lambda feats: correlation_trigger(feats, base_index=0, correlated_indices=[1, 2], multiplier=2.0, seed=42)
    if name == "pattern":
        return lambda feats: pattern_trigger(feats, pattern=[100.0, 200.0], indices=[0, 1], seed=42)

    # default: identity
    return lambda feats: feats
