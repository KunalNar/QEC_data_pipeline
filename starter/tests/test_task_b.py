import numpy as np
import pandas as pd

from quantum_lake_student.task_b import (
    FEATURE_NAMES,
    build_distance_data,
    choose_threshold,
    evaluate_distance,
)


def make_google_rows() -> pd.DataFrame:
    rows = []

    for distance in (3, 5):
        detector_count = 200 if distance == 3 else 600

        for shot_index, split in [
            (0, "train"),
            (8, "validation"),
            (1, "test"),
        ]:
            rows.append(
                {
                    "example_id": f"d{distance}_{shot_index}",
                    "experiment_id": f"exp_d{distance}",
                    "shot_index": shot_index,
                    "distance": distance,
                    "detector_count": detector_count,
                    "detector_event_count": 10,
                    "belief_matching_prediction": 0,
                    "correlated_matching_prediction": 1,
                    "pymatching_prediction": 0,
                    "tensor_network_contraction_prediction": 1,
                    "actual_observable_flip": bool(shot_index % 2),
                    "data_split": split,
                }
            )

    return pd.DataFrame(rows)


def test_build_distance_data_uses_five_features():
    google = make_google_rows()

    data = build_distance_data(google, distance=3)

    assert data.features.shape == (3, 5)
    assert len(FEATURE_NAMES) == 5

    expected_density = 10 / 200

    assert np.isclose(data.features[0, 0], expected_density)
    assert np.array_equal(
        data.features[0, 1:],
        np.array([0, 1, 0, 1], dtype=float),
    )


def test_build_distance_data_keeps_distances_separate():
    google = make_google_rows()

    d3 = build_distance_data(google, distance=3)
    d5 = build_distance_data(google, distance=5)

    assert set(d3.example_id) == {"d3_0", "d3_8", "d3_1"}
    assert set(d5.example_id) == {"d5_0", "d5_8", "d5_1"}


def test_choose_threshold_uses_validation_labels():
    probabilities = np.array([0.2, 0.8])
    labels = np.array([False, True])

    threshold = choose_threshold(probabilities, labels)

    assert 0.2 < threshold <= 0.8

# 2 new tests
def test_evaluate_distance_scores_all_required_models():
    google = make_google_rows()

    # Add enough train/validation/test examples for logistic regression
    extra_rows = []

    for shot_index, split, label in [
        (2, "train", False),
        (4, "train", True),
        (6, "train", False),
        (18, "validation", True),
        (3, "test", True),
        (5, "test", False),
    ]:
        extra_rows.append(
            {
                "example_id": f"d3_extra_{shot_index}",
                "experiment_id": "exp_d3",
                "shot_index": shot_index,
                "distance": 3,
                "detector_count": 200,
                "detector_event_count": 20,
                "belief_matching_prediction": int(label),
                "correlated_matching_prediction": int(not label),
                "pymatching_prediction": int(label),
                "tensor_network_contraction_prediction": int(label),
                "actual_observable_flip": label,
                "data_split": split,
            }
        )

    google = pd.concat(
        [google, pd.DataFrame(extra_rows)],
        ignore_index=True,
    )

    data = build_distance_data(google, distance=3)

    _, evaluation = evaluate_distance(data, distance=3)

    model_ids = {row["model_id"] for row in evaluation.metrics}

    assert model_ids == {
        "B_prior_baseline",
        "B_belief_matching",
        "B_correlated_matching",
        "B_pymatching",
        "B_tensor_network",
        "B_combined_logistic",
    }


def test_supplied_decoders_have_no_brier_score():
    google = make_google_rows()

    extra_rows = []

    for shot_index, split, label in [
        (2, "train", False),
        (4, "train", True),
        (6, "train", False),
        (18, "validation", True),
        (3, "test", True),
        (5, "test", False),
    ]:
        extra_rows.append(
            {
                "example_id": f"d3_extra_{shot_index}",
                "experiment_id": "exp_d3",
                "shot_index": shot_index,
                "distance": 3,
                "detector_count": 200,
                "detector_event_count": 20,
                "belief_matching_prediction": int(label),
                "correlated_matching_prediction": int(not label),
                "pymatching_prediction": int(label),
                "tensor_network_contraction_prediction": int(label),
                "actual_observable_flip": label,
                "data_split": split,
            }
        )

    google = pd.concat(
        [google, pd.DataFrame(extra_rows)],
        ignore_index=True,
    )

    data = build_distance_data(google, distance=3)

    _, evaluation = evaluate_distance(data, distance=3)

    supplied_rows = [
        row
        for row in evaluation.metrics
        if row["model_id"] in {
            "B_belief_matching",
            "B_correlated_matching",
            "B_pymatching",
            "B_tensor_network",
        }
    ]

    assert len(supplied_rows) == 4
    assert all(row["brier_score"] is None for row in supplied_rows)