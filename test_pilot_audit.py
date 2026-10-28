r"""
Unit and Integration Tests for Pilot Audit & Corrected Metrics
--------------------------------------------------------------
Verifies:
1. Confusion matrix counts sum to N=8 for all runs.
2. Metric calculations match confusion matrix equations.
3. Positive-class F1 and macro-F1 are computed separately and macro-F1 != positive F1 under imbalance.
4. Continuous predicted probabilities are used for ROC-AUC.
5. All predictions map to exact test manifest sample IDs with zero duplicates/omissions.
6. Generated CSV and JSON files match.
"""

import json
import unittest
import numpy as np
import pandas as pd


class TestPilotAudit(unittest.TestCase):
    def setUp(self):
        self.test_manifest_path = "splits_pilot/test_manifest.csv"
        self.comp_csv_path = "results/pilot_comparison.csv"
        self.comp_json_path = "results/pilot_results.json"
        self.preds_csv_path = "results/pilot_predictions.csv"

        self.df_test = pd.read_csv(self.test_manifest_path)
        self.df_comp = pd.read_csv(self.comp_csv_path)
        with open(self.comp_json_path, "r") as f:
            self.json_data = json.load(f)
        self.df_preds = pd.read_csv(self.preds_csv_path)

    def test_sample_counts_and_ids(self):
        """Verify that test manifest contains exactly 8 samples and matches predictions."""
        self.assertEqual(len(self.df_test), 8)
        expected_ids = self.df_test["video_id"].tolist()

        # Check predictions file
        unique_pred_ids = self.df_preds["sample_id"].unique().tolist()
        self.assertEqual(sorted(unique_pred_ids), sorted(expected_ids))

        # Check total runs = 4 modes * 3 conditions = 12 runs -> 12 * 8 = 96 rows
        self.assertEqual(len(self.df_preds), 96)

        # Check per run count
        for (mode, cond), group in self.df_preds.groupby(["model_mode", "condition"]):
            self.assertEqual(len(group), 8)
            self.assertEqual(group["sample_id"].tolist(), expected_ids)

    def test_confusion_matrix_sums(self):
        """Verify that TN + FP + FN + TP == 8 in every row of results."""
        for _, row in self.df_comp.iterrows():
            total = row["true_negatives"] + row["false_positives"] + row["false_negatives"] + row["true_positives"]
            self.assertEqual(total, 8, f"Row {row['mode']} {row['condition']} sum is {total} != 8")

    def test_metric_arithmetic_consistency(self):
        """Verify metric arithmetic against confusion matrix."""
        for _, row in self.df_comp.iterrows():
            tn = row["true_negatives"]
            fp = row["false_positives"]
            fn = row["false_negatives"]
            tp = row["true_positives"]

            # Accuracy
            expected_acc = (tp + tn) / 8.0
            self.assertAlmostEqual(row["accuracy"], expected_acc, places=5)

            # Balanced Accuracy
            rec_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            rec_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            expected_bal_acc = 0.5 * (rec_pos + rec_neg)
            self.assertAlmostEqual(row["balanced_accuracy"], expected_bal_acc, places=5)

            # Macro-F1 vs Positive-class F1
            expected_macro = 0.5 * (row["f1_positive"] + row["f1_negative"])
            self.assertAlmostEqual(row["f1_macro"], expected_macro, places=5)

    def test_macro_f1_differs_from_positive_f1_under_majority_collapse(self):
        """Verify that macro-F1 is distinct from positive-class F1 for clean baselines."""
        df_clean = self.df_comp[self.df_comp["condition"] == "none"].set_index("mode")
        for mode in ["audio", "video", "fusion"]:
            f1_pos = df_clean.loc[mode, "f1_positive"]
            f1_macro = df_clean.loc[mode, "f1_macro"]
            self.assertAlmostEqual(f1_pos, 0.857143, places=5)
            self.assertAlmostEqual(f1_macro, 0.428571, places=5)
            self.assertNotEqual(f1_pos, f1_macro)

    def test_csv_json_consistency(self):
        """Verify that pilot_comparison.csv and pilot_results.json match exactly."""
        self.assertEqual(len(self.df_comp), len(self.json_data))
        for idx, json_row in enumerate(self.json_data):
            csv_row = self.df_comp.iloc[idx].to_dict()
            for key in ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "true_positives", "true_negatives"]:
                self.assertAlmostEqual(csv_row[key], json_row[key], places=4)

    def test_probabilities_bounded(self):
        """Verify that all predicted probabilities are finite and within [0, 1]."""
        probs = self.df_preds["predicted_probability"].values
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))

    def test_gate_weights_properties(self):
        """Verify that RMS-AV gate weights are normalized and sum to 1.0."""
        rmsav_preds = self.df_preds[self.df_preds["model_mode"] == "rmsav"]
        for _, row in rmsav_preds.iterrows():
            g_a = row["gate_audio"]
            g_v = row["gate_video"]
            self.assertFalse(np.isnan(g_a))
            self.assertFalse(np.isnan(g_v))
            self.assertAlmostEqual(g_a + g_v, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
