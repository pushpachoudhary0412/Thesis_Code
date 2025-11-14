import numpy as np
import pytest

from mimiciv_backdoor_study.metrics import (
    classification_metrics,
    expected_calibration_error,
    backdoor_metrics,
)


def test_classification_metrics_perfect():
    y_true = np.array([0, 1, 0, 1])
    y_pred = y_true.copy()
    y_prob = np.vstack([[1.0, 0.0] if y == 0 else [0.0, 1.0] for y in y_true])
    m = classification_metrics(y_true, y_pred, y_prob)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["auroc"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)


def test_expected_calibration_error_perfect():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.vstack([[1.0, 0.0] if y == 0 else [0.0, 1.0] for y in y_true])
    ece = expected_calibration_error(y_true, y_prob, n_bins=5)
    assert ece == pytest.approx(0.0, abs=1e-8)


def test_expected_calibration_error_miscalibrated():
    # model predicts class 1 with 0.9 confidence but true label is 0 -> large ECE
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([[0.1, 0.9]] * 4)
    ece = expected_calibration_error(y_true, y_prob, n_bins=5)
    assert ece == pytest.approx(0.9, rel=1e-6)


def test_backdoor_metrics_asr_and_confidence_shift():
    y_clean_preds = np.array([0, 0])
    y_clean_probs = np.array([[0.6, 0.4], [0.7, 0.3]])
    y_poison_preds = np.array([1, 1])
    y_poison_probs = np.array([[0.1, 0.9], [0.2, 0.8]])

    bd = backdoor_metrics(
        y_clean_preds, y_clean_probs, y_poison_preds, y_poison_probs, target_label=None
    )
    # inferred target_label should be 1 and ASR = 1.0
    assert bd["ASR"] == pytest.approx(1.0)
    # confidence shift = mean(poison_target_conf) - mean(max_clean_conf)
    # mean(poison_target_conf) = (0.9 + 0.8) / 2 = 0.85
    # mean(max_clean_conf) = (0.6 + 0.7) / 2 = 0.65
    assert bd["confidence_shift"] == pytest.approx(0.20, rel=1e-6)


def test_backdoor_metrics_empty_inputs():
    bd = backdoor_metrics(
        np.array([], dtype=int), np.array([]), np.array([], dtype=int), np.array([])
    )
    assert np.isnan(bd["ASR"])
    assert np.isnan(bd["confidence_shift"])
