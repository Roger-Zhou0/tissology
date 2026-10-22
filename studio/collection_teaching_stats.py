"""A frozen classifier check on a uniform sample of a finite image pool.

The two accuracy intervals are separate, marginal 95% confidence statements.
They concern the pool eligible for this check, not patients, a new image source,
or the whole collection when previously reviewed images have been excluded.
The paired difference and the operational nonworse flag are descriptive only.
"""
from collections.abc import Mapping
from numbers import Integral

import numpy as np
from scipy.stats import hypergeom


def _finite_population_count_interval(correct, sample_size, frame_size, alpha=0.05):
    """Invert equal-tailed hypergeometric tests for an unknown integer total.

    If K images in the frozen frame are correctly classified, the number
    correct in a simple random sample without replacement follows
    Hypergeom(frame_size, K, sample_size). Discreteness makes these 95%
    intervals conservative. The census case has no sampling uncertainty.
    """
    possible_min = correct
    possible_max = frame_size - sample_size + correct
    low, high = possible_min, possible_max
    while low < high:
        middle = (low + high) // 2
        if hypergeom.sf(correct - 1, frame_size, middle, sample_size) >= alpha / 2:
            high = middle
        else:
            low = middle + 1
    lower = low
    low, high = possible_min, possible_max
    while low < high:
        middle = (low + high + 1) // 2
        if hypergeom.cdf(correct, frame_size, middle, sample_size) >= alpha / 2:
            low = middle
        else:
            high = middle - 1
    return [lower, low]


def _finite_population_interval(correct, sample_size, frame_size, alpha=0.05):
    return [value / frame_size for value in _finite_population_count_interval(correct, sample_size, frame_size, alpha)]


def _labels(values):
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError("Check labels must be a sequence of class names")
    try:
        labels = list(values)
    except TypeError as exc:
        raise ValueError("Check labels must be a sequence of class names") from exc
    if any(not isinstance(value, str) or not value.strip() for value in labels):
        raise ValueError("Every check image needs a nonempty class label")
    return labels


def assess_check(reference, original, candidate, frame_size):
    """Summarize paired frozen predictions on one complete probability check.

    The caller must verify the image IDs, common label taxonomy, frozen model
    identities, and uniform sampling design before passing aligned labels.
    Missing/abstained labels are rejected rather than silently dropped.
    """
    reference, original, candidate = map(_labels, (reference, original, candidate))
    sample_size = len(reference)
    if not sample_size or len(original) != sample_size or len(candidate) != sample_size:
        raise ValueError("Reference and prediction labels must cover the same nonempty check sample")
    if isinstance(frame_size, bool) or not isinstance(frame_size, Integral) or frame_size < sample_size:
        raise ValueError("The frozen check frame must have at least as many images as the sample")
    frame_size = int(frame_size)
    paired = {"fixed": 0, "introduced": 0, "both_correct": 0, "both_wrong": 0}
    for truth, before, after in zip(reference, original, candidate, strict=True):
        original_correct, candidate_correct = before == truth, after == truth
        key = ("both_correct" if original_correct and candidate_correct else
               "introduced" if original_correct else "fixed" if candidate_correct else "both_wrong")
        paired[key] += 1
    original_correct = paired["both_correct"] + paired["introduced"]
    candidate_correct = paired["both_correct"] + paired["fixed"]

    def result(correct):
        return {"correct": correct, "accuracy": correct / sample_size,
                "interval": _finite_population_interval(correct, sample_size, frame_size)}

    return {
        "sample_size": sample_size, "frame_size": frame_size,
        "sampling_fraction": sample_size / frame_size,
        "original": result(original_correct), "candidate": result(candidate_correct),
        "difference": (candidate_correct - original_correct) / sample_size,
        "paired": paired, "observed_nonworse": candidate_correct >= original_correct,
        "confidence_level": 0.95,
        "interval_method": "Exact equal-tailed hypergeometric inversion for simple random sampling without replacement",
        "interval_scope": "Each accuracy interval is marginal 95%; the two intervals are not a simultaneous statement.",
        "difference_scope": "The paired accuracy difference is descriptive. Observed nonworse is an operational gate, not evidence of statistical superiority or noninferiority.",
        "scope": "Accuracy within the frozen never-reviewed check frame. This does not establish patient-level accuracy or accuracy on a new source.",
    }


def assess_class_proportions(reference, original_frame, candidate_frame, check_positions, classes):
    """Jointly audit nine image-class fractions using the same random check.

    Each image is one unit. Prediction totals cover the complete eligible
    frame; references cover the frozen SRSWOR sample only. For each model and
    class, the correction is M/n times the sum of the paired reference minus
    prediction indicators. Positive and negative residual totals receive
    separate exact hypergeometric intervals. A union bound over 36 components
    gives conservative simultaneous 95% coverage of the 18 model-assisted
    fractions without assuming their errors are independent.

    Manual-only fractions receive their own nine-endpoint simultaneous 95%
    family. They are not jointly covered with the model-assisted family.
    """
    reference, original_frame, candidate_frame, classes = map(
        _labels, (reference, original_frame, candidate_frame, classes))
    frame_size, sample_size = len(original_frame), len(reference)
    if len(classes) != 9 or len(set(classes)) != 9:
        raise ValueError("Specify the nine distinct frozen tissue classes")
    if not frame_size or len(candidate_frame) != frame_size or not 0 < sample_size <= frame_size:
        raise ValueError("Both prediction arrays must cover the same nonempty frame and complete check")
    try:
        check_positions = list(check_positions)
    except TypeError as exc:
        raise ValueError("Specify the unique positions of the frozen random check") from exc
    if (len(check_positions) != sample_size or
            any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < frame_size for i in check_positions) or
            len(set(check_positions)) != sample_size):
        raise ValueError("Check positions must be unique in-frame integers aligned with every reference label")
    if any(value not in classes for values in (reference, original_frame, candidate_frame) for value in values):
        raise ValueError("All reference and prediction labels must belong to the frozen tissue classes")

    reference_indicators = np.array([[value == label for label in classes] for value in reference], dtype=np.int8)
    model_assisted, residual_columns, endpoint_order = {}, [], []
    component_count = 2 * 2 * len(classes)
    component_alpha = 0.05 / component_count
    for model_name, labels in (("original", original_frame), ("candidate", candidate_frame)):
        sample_indicators = np.array([[labels[i] == label for label in classes] for i in check_positions], dtype=np.int8)
        residuals = reference_indicators - sample_indicators
        residual_columns.append(residuals)
        endpoint_order.extend(f"{model_name}:{label}" for label in classes)
        estimates = []
        for column, label in enumerate(classes):
            predicted_total = labels.count(label)
            positive = int(np.sum(residuals[:, column] == 1))
            negative = int(np.sum(residuals[:, column] == -1))
            estimated_total = predicted_total + frame_size / sample_size * (positive - negative)
            positive_interval = _finite_population_count_interval(positive, sample_size, frame_size, component_alpha)
            negative_interval = _finite_population_count_interval(negative, sample_size, frame_size, component_alpha)
            # Intersect the confidence bounds, not the point estimate, with
            # the known feasible range for a true class total.
            lower = min(float(frame_size), max(0.0, predicted_total + positive_interval[0] - negative_interval[1]))
            upper = min(float(frame_size), max(0.0, predicted_total + positive_interval[1] - negative_interval[0]))
            estimates.append({
                "class": label, "predicted_total": predicted_total,
                "predicted_fraction": predicted_total / frame_size,
                "reference_count_in_check": int(reference_indicators[:, column].sum()),
                "positive_residual_count_in_check": positive, "negative_residual_count_in_check": negative,
                "estimated_total": estimated_total, "estimated_fraction": estimated_total / frame_size,
                "infeasible_point_estimate": not 0 <= estimated_total <= frame_size,
                "total_interval": [lower, upper], "interval": [lower / frame_size, upper / frame_size],
                "positive_residual_total_interval": positive_interval,
                "negative_residual_total_interval": negative_interval,
            })
        model_assisted[model_name] = {"classes": estimates}

    manual = []
    for column, label in enumerate(classes):
        correct = int(reference_indicators[:, column].sum())
        total_interval = _finite_population_count_interval(correct, sample_size, frame_size, 0.05 / len(classes))
        interval = [v / frame_size for v in total_interval]
        manual.append({"class": label, "reference_count_in_check": correct,
                       "estimated_total": frame_size * correct / sample_size,
                       "estimated_fraction": correct / sample_size,
                       "interval": interval, "total_interval": total_interval})
    if sample_size == frame_size:
        covariance = np.zeros((len(endpoint_order), len(endpoint_order))).tolist()
    elif sample_size >= 2:
        combined = np.concatenate(residual_columns, axis=1)
        covariance = ((1 - sample_size / frame_size) / sample_size * np.cov(combined, rowvar=False, ddof=1)).tolist()
    else:
        covariance = None
    return {
        "schema": "collection-shared-class-proportions-v1",
        "sample_size": sample_size, "frame_size": frame_size,
        "inclusion_probability": sample_size / frame_size,
        "classes": classes, "model_assisted": model_assisted, "manual_only": {"classes": manual},
        "measurement": {
            "numerator": "Number of eligible images whose reference class is the specified tissue class",
            "denominator": "All images in the frozen never-reviewed check frame; each image counts once",
            "aggregation": "Sum class indicators over images, then divide by the fixed frame size",
            "eligible_images": "The frozen never-reviewed check frame, excluding development and earlier reserved check images",
            "physical_scale": "None: image-class fractions are not tissue area, cell density, or spatial measurements",
            "exclusions": "Only exclusions frozen before random sampling; sampled images with missing references cannot be dropped",
        },
        "uncertainty": {
            "confidence_level": 0.95,
            "model_assisted_family": "Simultaneous across both frozen models and all nine class fractions (18 estimates)",
            "method": "Exact hypergeometric residual-total bounds with Bonferroni over 36 positive/negative components",
            "component_count": component_count, "component_alpha": component_alpha,
            "dependence": "All endpoints reuse the same sample. The union bound requires no independence assumption; intervals are conservative.",
            "manual_only_family": "A separate simultaneous 95% family of nine reference-only class fractions, using Bonferroni over nine exact hypergeometric intervals",
            "families_are_joint": False,
            "raw_estimates": "Unbiased model-assisted total estimates may lie outside the feasible range. Raw values are retained and flagged; they are not clipped or renormalized.",
        },
        "joint_covariance": {
            "endpoint_order": endpoint_order, "scale": "estimated fractions", "matrix": covariance,
            "method": "(1 - n/M) times paired sample residual covariance divided by n; unavailable for a non-census sample of one",
            "used_for_intervals": False,
        },
        "scope": "Audited image-class fractions for the eligible never-reviewed image frame only. These are not whole-collection, tissue-area, patient, or population estimates.",
    }
