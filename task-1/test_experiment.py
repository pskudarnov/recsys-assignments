"""Checks for temporal feature isolation and ranking metrics."""
import unittest

import numpy as np
import pandas as pd

from experiment import lookup_counts, make_snapshot, metrics


class ExperimentChecks(unittest.TestCase):
    def test_window_boundary_and_future_isolation(self):
        cutoff = pd.Timestamp("2021-08-02")
        events = pd.DataFrame({
            "user_id": [1, 1, 2, 3], "item_id": [10, 11, 11, 12],
            "date": pd.to_datetime(["2021-07-26", "2021-08-01", "2021-07-20", "2021-07-20"]),
            "genre": ["drama", "comedy", "comedy", "action"]})
        keys = pd.DataFrame({"user_id": [1], "item_id": [10], "genre": ["drama"]})
        self.assertEqual(lookup_counts(events[events.date < cutoff], "pair", keys, 7, cutoff)[0], 1)
        self.assertEqual(lookup_counts(events[events.date < cutoff], "pair", keys, 3, cutoff)[0], 0)

        items = pd.DataFrame({"item_id": [10, 11, 12], "content_type": [0, 0, 0],
                              "release_year": [2020, 2021, 2019],
                              "genre": ["drama", "comedy", "action"],
                              "genre_id": [1, 0, 2], "for_kids": [0, 0, 0],
                              "age_rating": [16, 16, 16]})
        users = pd.DataFrame({"user_id": [1, 2, 3], "age": [1, 2, 3],
                              "income": [1, 1, 1], "sex": [0, 1, 0],
                              "kids_flg": [0, 0, 0]})
        before_x, before_y, _, _ = make_snapshot(events, items, users, "2021-08-02", 1)
        self.assertTrue((before_x["pair_180"] == 0).all())
        future = pd.DataFrame({"user_id": [2], "item_id": [10],
                               "date": [pd.Timestamp("2021-08-02")], "genre": ["drama"]})
        after_x, after_y, _, _ = make_snapshot(pd.concat([events, future]), items, users,
                                                "2021-08-02", 1)
        pd.testing.assert_frame_equal(before_x, after_x)
        self.assertEqual(after_y.sum() - before_y.sum(), 1)

    def test_ndcg_zero_positive_user_is_zero(self):
        y = np.array([1, 0, 0, 0])
        score = np.array([0.1, 0.9, 0.9, 0.1])
        got = metrics(y, score, 2, 2)
        self.assertAlmostEqual(got["ndcg@10"], 1 / np.log2(3) / 2)
        self.assertEqual(got["users_with_candidate_positive"], 1)

    def test_empty_positive_set_reports_undefined_auc(self):
        got = metrics(np.zeros(20, dtype=np.uint8), np.zeros(20), 2, 10)
        self.assertEqual(got["ndcg@10"], 0)
        self.assertIsNone(got["roc_auc"])
        self.assertIsNone(got["ndcg@10_given_candidate_positive"])


if __name__ == "__main__":
    unittest.main()
