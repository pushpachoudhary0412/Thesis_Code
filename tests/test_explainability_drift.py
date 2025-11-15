"""
Unit tests for explainability_drift.py
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import spearmanr

from mimiciv_backdoor_study.explainability_drift import (
    normalize_abs,
    attribution_distance,
    rank_features,
    feature_rank_change,
    trigger_attribution_ratio,
    attention_shift,
)


class TestNormalizeAbs:
    def test_1d_input(self):
        attrib = np.array([1.0, 2.0, 3.0])
        result = normalize_abs(attrib)
        expected = np.array([[1/6, 2/6, 3/6]])  # _ensure_2d makes it 2D
        assert_allclose(result, expected)

    def test_2d_input(self):
        attrib = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = normalize_abs(attrib)
        expected = np.array([
            [1/6, 2/6, 3/6],
            [4/15, 5/15, 6/15]
        ])
        assert_allclose(result, expected)

    def test_with_zeros(self):
        attrib = np.array([0.0, 0.0, 1.0])
        result = normalize_abs(attrib)
        expected = np.array([[0.0, 0.0, 1.0]])  # _ensure_2d makes it 2D
        assert_allclose(result, expected)


class TestAttributionDistance:
    def test_l2_1d(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 4.0])
        result = attribution_distance(a, b, metric="l2")
        expected = np.array([1.0])  # sqrt((0)^2 + (0)^2 + (1)^2)
        assert_allclose(result, expected)

    def test_l2_2d(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[1.0, 3.0], [3.0, 5.0]])
        result = attribution_distance(a, b, metric="l2")
        expected = np.array([1.0, 1.0])  # sqrt(1^2) and sqrt(1^2)
        assert_allclose(result, expected)

    def test_cosine_1d(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        result = attribution_distance(a, b, metric="cosine")
        expected = np.array([1.0])  # orthogonal vectors
        assert_allclose(result, expected)

    def test_cosine_2d(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = attribution_distance(a, b, metric="cosine")
        expected = np.array([1.0, 1.0])
        assert_allclose(result, expected)

    def test_shape_mismatch(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            attribution_distance(a, b)


class TestRankFeatures:
    def test_1d_input(self):
        attrib = np.array([3.0, 1.0, 2.0])  # ranks should be [1, 3, 2]
        result = rank_features(attrib)
        expected = np.array([1, 3, 2])
        assert_array_equal(result, expected)

    def test_2d_input(self):
        attrib = np.array([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0]])
        result = rank_features(attrib)
        expected = np.array([[1, 3, 2], [3, 1, 2]])
        assert_array_equal(result, expected)


class TestFeatureRankChange:
    def test_basic(self):
        a = np.array([[1.0, 2.0, 3.0]])  # ranks [3, 2, 1]
        b = np.array([[3.0, 2.0, 1.0]])  # ranks [1, 2, 3]
        result = feature_rank_change(a, b, topk=2)
        assert "spearman_r" in result
        assert "mean_abs_rank_shift" in result
        assert "topk_jaccard" in result
        assert "topk_overlap_count" in result
        # Perfect negative correlation
        assert_allclose(result["spearman_r"], -1.0, atol=1e-10)
        assert_allclose(result["mean_abs_rank_shift"], 4/3)  # mean of [2, 0, 2] = 4/3

    def test_no_topk(self):
        a = np.array([[1.0, 2.0]])
        b = np.array([[2.0, 1.0]])
        result = feature_rank_change(a, b)
        assert "spearman_r" in result
        assert "mean_abs_rank_shift" in result
        assert "topk_jaccard" not in result

    def test_feature_mismatch(self):
        a = np.array([[1.0, 2.0]])
        b = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError):
            feature_rank_change(a, b)


class TestTriggerAttributionRatio:
    def test_1d_input(self):
        attrib = np.array([1.0, 2.0, 3.0, 4.0])
        trigger_mask = np.array([True, False, True, False])
        result = trigger_attribution_ratio(attrib, trigger_mask)
        expected = np.array([4.0 / 10.0])  # (1+3)/(1+2+3+4)
        assert_allclose(result, expected)

    def test_2d_input(self):
        attrib = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        trigger_mask = np.array([True, False, True])
        result = trigger_attribution_ratio(attrib, trigger_mask)
        expected = np.array([4.0/6.0, 10.0/15.0])  # (1+3)/6, (4+6)/15
        assert_allclose(result, expected)

    def test_mask_length_mismatch(self):
        attrib = np.array([1.0, 2.0])
        trigger_mask = np.array([True, False, True])
        with pytest.raises(ValueError):
            trigger_attribution_ratio(attrib, trigger_mask)


class TestAttentionShift:
    def test_2d_input(self):
        # shape (n_heads, n_features)
        attn_clean = np.array([[0.5, 0.5], [0.3, 0.7]])
        attn_poison = np.array([[0.6, 0.4], [0.4, 0.6]])
        result = attention_shift(attn_clean, attn_poison)
        assert "mean_l1" in result
        assert "mean_l2" in result
        assert "mean_kl" in result
        assert result["mean_l1"] > 0
        assert result["mean_l2"] > 0
        assert result["mean_kl"] >= 0

    def test_3d_input(self):
        # shape (n_samples, n_heads, n_features)
        attn_clean = np.array([[[0.5, 0.5], [0.3, 0.7]]])
        attn_poison = np.array([[[0.6, 0.4], [0.4, 0.6]]])
        result = attention_shift(attn_clean, attn_poison)
        assert "mean_l1" in result
        assert "mean_l2" in result
        assert "mean_kl" in result

    def test_shape_mismatch(self):
        attn_clean = np.array([[0.5, 0.5]])
        attn_poison = np.array([[0.5, 0.5, 0.0]])
        with pytest.raises(ValueError):
            attention_shift(attn_clean, attn_poison)

    def test_zero_attention(self):
        attn_clean = np.array([[0.0, 0.0]])
        attn_poison = np.array([[0.0, 0.0]])
        result = attention_shift(attn_clean, attn_poison)
        assert_allclose(result["mean_kl"], 0.0)
