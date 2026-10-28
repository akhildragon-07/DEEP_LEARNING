r"""
Unit Test Suite: RMS-AV Decision Boundary & Uncertainty Band Verification
-------------------------------------------------------------------------
Verifies:
1. Resolution of false positive prediction for near-zero logits (z = +0.0329 -> 50.82%).
2. Uncertainty band boundary logic [threshold - margin, threshold + margin].
3. Mathematical correctness of sigmoid transformation.
4. Dynamic margin configuration (0.00 strict cutoff to wide margins).
5. Integration with InferenceEngine methods.
"""

import math
import unittest
import numpy as np
from PIL import Image

from inference import InferenceEngine, INCONCLUSIVE_VERDICT


class TestUncertaintyBand(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = InferenceEngine(checkpoint_dir="checkpoints", device="cpu")

    def test_user_reported_case_reproduction(self):
        """
        Verify the exact case reported by the user:
        Raw logit = +0.0329
        Sigmoid prob = 50.82%
        Threshold = 0.50
        Default Margin = 0.05 (Band: [0.45, 0.55])
        Expected Verdict: 'INCONCLUSIVE — MODEL SCORE NEAR THRESHOLD'
        """
        raw_logit = 0.0329
        prob = 1.0 / (1.0 + math.exp(-raw_logit))
        self.assertAlmostEqual(prob, 0.5082, places=4)

        # Evaluate with classify_prediction
        verdict, is_inconclusive, (low_b, up_b) = InferenceEngine.classify_prediction(
            prob, threshold=0.50, uncertainty_margin=0.05
        )

        self.assertEqual(verdict, INCONCLUSIVE_VERDICT)
        self.assertTrue(is_inconclusive)
        self.assertEqual(low_b, 0.45)
        self.assertEqual(up_b, 0.55)
        self.assertTrue(low_b <= prob <= up_b)

    def test_verdict_three_state_partitions(self):
        """
        Verify that predictions fall strictly into three non-overlapping states:
        - prob < threshold - margin: Likely authentic
        - threshold - margin <= prob <= threshold + margin: INCONCLUSIVE
        - prob > threshold + margin: Likely manipulated
        """
        thresh = 0.50
        margin = 0.05
        # Authentic region: < 0.45
        v_auth, inc_auth, _ = InferenceEngine.classify_prediction(0.35, threshold=thresh, uncertainty_margin=margin)
        self.assertEqual(v_auth, "Likely authentic")
        self.assertFalse(inc_auth)

        # Inconclusive lower boundary: 0.45
        v_low, inc_low, _ = InferenceEngine.classify_prediction(0.45, threshold=thresh, uncertainty_margin=margin)
        self.assertEqual(v_low, INCONCLUSIVE_VERDICT)
        self.assertTrue(inc_low)

        # Inconclusive midpoint: 0.50
        v_mid, inc_mid, _ = InferenceEngine.classify_prediction(0.50, threshold=thresh, uncertainty_margin=margin)
        self.assertEqual(v_mid, INCONCLUSIVE_VERDICT)
        self.assertTrue(inc_mid)

        # Inconclusive upper boundary: 0.55
        v_up, inc_up, _ = InferenceEngine.classify_prediction(0.55, threshold=thresh, uncertainty_margin=margin)
        self.assertEqual(v_up, INCONCLUSIVE_VERDICT)
        self.assertTrue(inc_up)

        # Manipulated region: > 0.55
        v_fake, inc_fake, _ = InferenceEngine.classify_prediction(0.65, threshold=thresh, uncertainty_margin=margin)
        self.assertEqual(v_fake, "Likely manipulated")
        self.assertFalse(inc_fake)

    def test_strict_binary_threshold_when_margin_is_zero(self):
        """
        Verify backward compatibility: when margin=0.0, the decision reverts to
        strict binary thresholding without an inconclusive band.
        """
        thresh = 0.50
        margin = 0.00

        # Score just above 0.50
        v_above, inc_above, _ = InferenceEngine.classify_prediction(0.5082, threshold=thresh, uncertainty_margin=margin)
        self.assertEqual(v_above, "Likely manipulated")
        self.assertFalse(inc_above)

        # Score just below 0.50
        v_below, inc_below, _ = InferenceEngine.classify_prediction(0.4918, threshold=thresh, uncertainty_margin=margin)
        self.assertEqual(v_below, "Likely authentic")
        self.assertFalse(inc_below)

    def test_custom_calibrated_threshold(self):
        """
        Verify custom threshold (e.g. 0.54 to account for pilot 75% fake training prior bias).
        """
        thresh = 0.54
        margin = 0.02  # Band: [0.52, 0.56]

        # 50.82% is now cleanly below the band -> authentic
        v, inc, (low, up) = InferenceEngine.classify_prediction(0.5082, threshold=thresh, uncertainty_margin=margin)
        self.assertEqual(v, "Likely authentic")
        self.assertFalse(inc)
        self.assertEqual(low, 0.52)
        self.assertEqual(up, 0.56)

    def test_integration_predict_image_with_uncertainty(self):
        """
        Integration test verifying that predict_image returns all required
        uncertainty metadata keys.
        """
        if "video" not in self.engine.models:
            self.skipTest("Video model not loaded.")

        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        res = self.engine.predict_image(img, threshold=0.50, uncertainty_margin=0.05)

        self.assertIn("prediction", res)
        self.assertIn("raw_logit", res)
        self.assertIn("manipulation_probability", res)
        self.assertIn("threshold_applied", res)
        self.assertIn("uncertainty_margin", res)
        self.assertIn("uncertainty_band", res)
        self.assertIn("is_inconclusive", res)
        self.assertEqual(res["uncertainty_margin"], 0.05)
        self.assertEqual(res["uncertainty_band"], [0.45, 0.55])

        # If manipulation probability is between 0.45 and 0.55, must be INCONCLUSIVE
        prob = res["manipulation_probability"]
        if 0.45 <= prob <= 0.55:
            self.assertEqual(res["prediction"], INCONCLUSIVE_VERDICT)
            self.assertTrue(res["is_inconclusive"])
        elif prob > 0.55:
            self.assertEqual(res["prediction"], "Likely manipulated")
            self.assertFalse(res["is_inconclusive"])
        else:
            self.assertEqual(res["prediction"], "Likely authentic")
            self.assertFalse(res["is_inconclusive"])


if __name__ == "__main__":
    unittest.main()
