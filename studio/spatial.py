"""Finite-image spatial area audits with fixed predictions and probability sampling.

Primary simultaneous bounds: Bardenet & Maillard (2015), Corollary 2.5,
https://arxiv.org/pdf/1309.4029, applied to each sign of four bounded components.
Covariances are design-based diagnostics; normal/bootstrap coverage is not claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import threading
import uuid

import numpy as np
from PIL import Image

from .storage import digest, now, read, write

COMPONENTS = ("epithelial", "stromal", "dense_lymphocytic_tissue", "eligible")
CODES = ((1, 4), (2, 3), (3,), (1, 2, 3, 4))
ENDPOINTS = tuple(name + "_fraction" for name in COMPONENTS[:3])
METHODS = ("manual", "uniform")
SOURCE = {"authors": "Bardenet and Maillard", "year": 2015,
          "theorem": "Corollary 2.5, two tails and Bonferroni over four components",
          "url": "https://arxiv.org/pdf/1309.4029", "doi": "10.3150/14-BEJ605"}


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check_file(path, expected):
    if file_sha256(path) != expected:
        raise ValueError(f"File hash mismatch: {path}")


def seal(value):
    if "sha256" in value:
        raise ValueError("Already sealed")
    return dict(value, sha256=digest(value))


def verify(value):
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("Spatial artifact hash mismatch")


def components_from_mask(mask):
    labels = np.asarray(mask)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels > 8)):
        raise ValueError("Expected a two-dimensional region-label mask, codes 0–8")
    counts = np.bincount(labels.ravel(), minlength=9)
    return [int(counts[list(codes)].sum()) for codes in CODES]


def _validate_components(values, areas, label):
    values = np.asarray(values, dtype=float)
    areas = np.asarray(areas, dtype=float)
    if (values.shape != (len(areas), 4) or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values != np.floor(values))
            or np.any(values[:, 0] + values[:, 1] != values[:, 3])
            or np.any(values[:, 2] > values[:, 1]) or np.any(values[:, 3] > areas)):
        raise ValueError(f"Invalid {label} joint pixel components")
    return values


def image_windows(roi_id, fov, size, window_side=64, context_pixels=64):
    """Partition an image-defined FOV exactly; partial edge windows are retained."""
    if type(window_side) is not int or window_side <= 0 or type(context_pixels) is not int or context_pixels < 0:
        raise ValueError("Invalid spatial window/context size")
    left, top, right, bottom = (int(fov[k]) for k in ("left", "top", "right", "bottom"))
    width, height = map(int, size)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("FOV outside source image")
    rows = []
    for y in range(top, bottom, window_side):
        for x in range(left, right, window_side):
            x1, y1 = min(right, x + window_side), min(bottom, y + window_side)
            rows.append({"id": f"{roi_id}:x{x}:y{y}", "roi_id": roi_id,
                         "bounds": {"left": x, "top": y, "right": x1, "bottom": y1},
                         "context_bounds": {"left": max(0, x - context_pixels), "top": max(0, y - context_pixels),
                                            "right": min(width, x1 + context_pixels), "bottom": min(height, y1 + context_pixels)},
                         "area_pixels": (x1 - x) * (y1 - y), "role": "study"})
    if sum(row["area_pixels"] for row in rows) != (right - left) * (bottom - top):
        raise AssertionError("Image-window partition lost pixels")
    return rows


def _crop(mask, bounds):
    return mask[bounds["top"]:bounds["bottom"], bounds["left"]:bounds["right"]]


def prepare_spatial_study(smoke_dir, study_dir, window_side=64, context_pixels=64):
    """Freeze an image sampling frame and predictions without opening label masks."""
    smoke_dir, study_dir = Path(smoke_dir).resolve(), Path(study_dir).resolve()
    if study_dir.exists() and any(study_dir.iterdir()):
        raise FileExistsError("Spatial study destination must be empty")
    frozen = read(smoke_dir / "frozen-predictions.json")
    check_file(smoke_dir / "frozen-plan.json", frozen["plan_sha256"])
    source_plan = read(smoke_dir / "frozen-plan.json")
    if not frozen.get("strict_state_dict") or frozen.get("missing_keys") or frozen.get("unexpected_keys"):
        raise ValueError("Require successfully frozen strict-checkpoint predictions")
    data_dir = Path(source_plan["source_data_dir"])
    rois, rows, prediction_rows, reference_rois = [], [], [], []
    for sample, predicted in zip(source_plan["samples"], frozen["predictions"], strict=True):
        if sample["roi_id"] != predicted["roi_id"]:
            raise ValueError("Source prediction order mismatch")
        rgb_info, reference_info = sample["files"]["rgbs"], sample["files"]["masks"]
        image_path = data_dir / "rgbs" / rgb_info["name"]
        check_file(image_path, rgb_info["sha256"])
        check_file(predicted["arrays"], predicted["arrays_sha256"])
        check_file(predicted["region_png"], predicted["region_png_sha256"])
        with Image.open(image_path) as image:
            size = list(image.size)
        fov = {k: int(sample["manual_fov"][k]) for k in ("left", "top", "right", "bottom")}
        windows = image_windows(sample["roi_id"], fov, size, window_side, context_pixels)
        with np.load(predicted["arrays"], allow_pickle=False) as cache:
            region_mask = cache["region_native"]
        if region_mask.shape != (size[1], size[0]):
            raise ValueError("Prediction geometry differs from original image")
        for window in windows:
            prediction_rows.append({"id": window["id"], "components": components_from_mask(_crop(region_mask, window["bounds"]))})
        rows.extend(windows)
        rois.append({"id": sample["roi_id"], "slide": sample["slide"], "image_path": str(image_path),
                     "image_sha256": rgb_info["sha256"], "size_px": size, "manual_fov": fov,
                     "prediction_mask_path": predicted["region_png"], "prediction_mask_sha256": predicted["region_png_sha256"]})
        # No reference-mask raster is opened or summarized during preparation.
        reference_rois.append({"roi_id": sample["roi_id"], "path": str(data_dir / "masks" / reference_info["name"]),
                               "sha256": reference_info["sha256"], "channel": 0})
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Sampling frame must contain unique image windows")
    frame = seal({"schema": "spatial-frame-v1", "created_at": now(), "rois": rois, "units": rows,
                  "window_side_pixels": window_side, "context_pixels": context_pixels,
                  "coordinates": "Native image pixels; half-open bounds; context never contributes to central-window totals",
                  "target_pixels": sum(row["area_pixels"] for row in rows),
                  "target": "Finite union of selected manual FOVs only; no slide/patient/population inference"})
    predictions = seal({"schema": "spatial-predictions-v1", "frame_sha256": frame["sha256"],
                        "source_predictions_path": str(smoke_dir / "frozen-predictions.json"),
                        "source_predictions_sha256": file_sha256(smoke_dir / "frozen-predictions.json"),
                        "source_plan_sha256": frozen["plan_sha256"], "components": list(COMPONENTS), "units": prediction_rows})
    references = seal({"schema": "spatial-reference-sources-v1", "frame_sha256": frame["sha256"],
                       "source": "Existing manual PanopTILs annotations, simulated review only", "rois": reference_rois})
    config = seal({"schema": "spatial-study-v1", "created_at": now(), "name": f"{len(rois)}-FOV spatial area study",
                   "frame_sha256": frame["sha256"], "prediction_sha256": predictions["sha256"],
                   "reference_sources_sha256": references["sha256"], "components": list(COMPONENTS),
                   "component_codes": {name: list(codes) for name, codes in zip(COMPONENTS, CODES)},
                   "endpoints": list(ENDPOINTS), "aggregation": "Sum numerator pixel areas / sum eligible pixel areas",
                   "physical_units": False, "cell_counts": False,
                   "epithelial_and_stromal_are_complementary": True,
                   "development": "No retraining; reference labels do not alter frozen MuTILs predictions",
                   "reference_labels_previously_inspected": True,
                   "scope": "Engineering study with simulated reviews, not independent model validation or pathologist time savings",
                   "interval_source": SOURCE})
    for filename, artifact in (("frame.json", frame), ("predictions.json", predictions),
                               ("reference-sources.json", references), ("study.json", config)):
        write(study_dir / filename, artifact)
    return config


def serfling_rho(N, n):
    if type(N) is not int or type(n) is not int or not 1 <= n <= N:
        raise ValueError("Require 1 ≤ n ≤ N")
    if n == N:
        return 0.0
    return 1 - (n - 1) / N if n <= N / 2 else (1 - n / N) * (1 + 1 / n)


def estimate_areas(predicted, areas, sample_indices, references, method="uniform", alpha=.05):
    """Joint SRSWOR component totals and simultaneous conservative ratio bounds.

Only selected-window reference components are accepted. Bounds are fixed by the
full frame's known areas and predictions, never by observed residual extrema.
"""
    if method not in METHODS or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("Invalid spatial estimation method or alpha")
    areas = np.asarray(areas, dtype=float)
    if areas.ndim != 1 or not len(areas) or not np.isfinite(areas).all() or np.any(areas <= 0) or np.any(areas != np.floor(areas)):
        raise ValueError("Window areas must be positive integer pixel counts")
    predicted = _validate_components(predicted, areas, "predicted")
    indices_array = np.asarray(sample_indices)
    if indices_array.ndim != 1 or not np.issubdtype(indices_array.dtype, np.integer):
        raise ValueError("Sample indices must be integer positions")
    indices = indices_array.astype(int)
    N, n, K = len(areas), len(indices), 4
    if not 1 <= n <= N or len(set(indices.tolist())) != n or np.any((indices < 0) | (indices >= N)):
        raise ValueError("Invalid SRS sample indices")
    reference = _validate_components(references, areas[indices], "reference")
    predictor = predicted if method == "uniform" else np.zeros_like(predicted)
    residual = reference - predictor[indices]
    predicted_total = predicted.sum(axis=0)
    totals = predictor.sum(axis=0) + N * residual.mean(axis=0)
    sampled_totals = reference.sum(axis=0)
    if n == N:
        totals = sampled_totals.copy()  # Avoid census floating-point cancellation.
    missing_area = float(areas.sum() - areas[indices].sum())
    deterministic_lower = sampled_totals
    deterministic_upper = sampled_totals + missing_area
    # y_ij is in [0, area_i]. These endpoints enclose every possible population
    # residual, including missed tissue and excluded-region errors.
    residual_lower = (-predictor).min(axis=0)
    residual_upper = (areas[:, None] - predictor).max(axis=0)
    rho = serfling_rho(N, n)
    radius = N * (residual_upper - residual_lower) * math.sqrt(rho * math.log(2 * K / alpha) / (2 * n))
    lower = np.maximum(totals - radius, deterministic_lower)
    upper = np.minimum(totals + radius, deterministic_upper)
    # An empty stochastic/deterministic intersection is possible on a tail event.
    # Fall back to guaranteed deterministic bounds rather than invent a point CI.
    conflicts = lower > upper + 1e-8
    lower = np.where(conflicts, deterministic_lower, lower)
    upper = np.where(conflicts, deterministic_upper, upper)
    lower = np.minimum(lower, upper)
    if n == N:
        lower = upper = sampled_totals.copy()
    if n == N:
        covariance = np.zeros((K, K))
    elif n >= 2:
        covariance = N * N * (1 - n / N) * np.cov(residual, rowvar=False, ddof=1) / n
    else:
        covariance = None
    denominator = float(totals[3])
    diagnostic_ratio_covariance = None
    if denominator > 0 and covariance is not None:
        gradient = np.zeros((3, 4))
        for j in range(3):
            gradient[j, j] = 1 / denominator
            gradient[j, 3] = -totals[j] / denominator**2
        diagnostic_ratio_covariance = gradient @ covariance @ gradient.T
    endpoints = []
    for j, name in enumerate(ENDPOINTS):
        point = float(totals[j] / denominator) if denominator > 0 else None
        ratio_fallback = False
        if upper[3] == 0:
            interval, interval_status = None, "no_eligible_area"
        elif lower[3] <= 0:
            interval, interval_status = [0.0, 1.0], "conditional_on_positive_eligible_area"
        else:
            lo = max(0.0, float(lower[j] / upper[3]))
            hi = min(1.0, float(upper[j] / lower[3]))
            # Deterministic ratio bounds use the shared remaining-area constraint:
            # numerator is a subset of eligible area in every unreviewed window.
            max_possible_denominator = sampled_totals[3] + missing_area
            if max_possible_denominator > 0:
                lo = max(lo, float(sampled_totals[j] / max_possible_denominator))
                hi = min(hi, float((sampled_totals[j] + missing_area) / max_possible_denominator))
            if lo > hi + 1e-8:
                # Joint rectangle may be incompatible on a concentration-tail event.
                lo = float(sampled_totals[j] / max_possible_denominator)
                hi = float((sampled_totals[j] + missing_area) / max_possible_denominator)
                ratio_fallback = True
            interval, interval_status = [min(lo, hi), hi], "simultaneous_conservative"
        if n == N and denominator > 0:
            interval, interval_status = [point, point], "exact_census"
        endpoints.append({"name": name, "predicted_fraction": float(predicted_total[j] / predicted_total[3]) if predicted_total[3] > 0 else None,
                          "estimate": point, "estimate_status": "nonpositive_estimated_denominator" if denominator <= 0 else
                          ("outside_physical_range" if not 0 <= point <= 1 else "defined"),
                          "interval": interval, "interval_status": interval_status,
                          "interval_width": interval[1] - interval[0] if interval is not None else None,
                          "used_deterministic_fallback": ratio_fallback,
                          "estimate_outside_interval": bool(point is not None and interval is not None and
                                                            not interval[0] <= point <= interval[1]),
                          "raw_estimate_note": "Raw ratio estimate is retained even when inconsistent with known reviewed-area bounds",
                          "diagnostic_standard_error": math.sqrt(max(0.0, float(diagnostic_ratio_covariance[j, j]))) if diagnostic_ratio_covariance is not None else None})
    return {"schema": "spatial-estimate-v1", "method": method, "population_size": N, "reviewed": n,
            "alpha": alpha, "confidence_scope": "Simultaneous across all four component totals and three defined ratios at this fixed audit budget",
            "point_estimator": "P + N * sample_mean(reference - prediction)" if method == "uniform" else "N * sample_mean(reference)",
            "ratio_estimator_is_exactly_unbiased": False,
            "components": [{"name": name, "predicted_total": float(predicted_total[j]), "estimated_total": float(totals[j]),
                            "interval": [float(lower[j]), float(upper[j])], "serfling_total_radius": float(radius[j]),
                            "population_residual_bounds": [float(residual_lower[j]), float(residual_upper[j])],
                            "deterministic_interval": [float(deterministic_lower[j]), float(deterministic_upper[j])],
                            "used_deterministic_fallback": bool(conflicts[j])} for j, name in enumerate(COMPONENTS)],
            "total_covariance": covariance.tolist() if covariance is not None else None,
            "covariance_method": "N²(1−n/N) sample covariance of residuals / n; unavailable at n=1 except census",
            "diagnostic_ratio_covariance": diagnostic_ratio_covariance.tolist() if diagnostic_ratio_covariance is not None else None,
            "covariance_is_coverage_guarantee": False,
            "rho_n": rho, "tail_probability_each": alpha / (2 * K), "interval_source": SOURCE,
            "remaining_pixels": int(missing_area), "endpoints": endpoints,
            "physical_units": False, "cell_counts": False,
            "limitations": ["Finite selected-image population; no patient-level generalization", "Fixed sample size; not valid for optional stopping",
                            "Conservative component bounds can remain wide even with a good predictor", "Ratios can be biased and raw assisted point estimates can leave physical ranges"]}


def recompute_export(bundle):
    """Verify and recompute an exported audit without source images or masks.

The export supplies fixed frame/predictions and only sampled annotation totals.
No filesystem path in the export is opened. Original result metadata is retained
by the export; this function returns a new calculation of its numerical fields.
"""
    verify(bundle)
    config, frame, predictions, plan, annotations, saved = (bundle[key] for key in
        ("study", "frame", "predictions", "plan", "annotations", "result"))
    for artifact in (config, frame, predictions, plan, annotations, saved):
        verify(artifact)
    if (config["frame_sha256"] != frame["sha256"] or config["prediction_sha256"] != predictions["sha256"]
            or predictions["frame_sha256"] != frame["sha256"] or plan["study_sha256"] != config["sha256"]
            or plan["frame_sha256"] != frame["sha256"] or plan["prediction_sha256"] != predictions["sha256"]
            or annotations["plan_sha256"] != plan["sha256"] or saved["plan_sha256"] != plan["sha256"]
            or saved["annotation_sha256"] != annotations["sha256"] or saved["study_sha256"] != config["sha256"]):
        raise ValueError("Export provenance links do not match")
    units = frame["units"]
    N, n = len(units), plan["budget"]
    expected = np.random.default_rng(plan["seed"]).choice(N, size=n, replace=False).tolist()
    if (len({row["id"] for row in units}) != N or plan["population_size"] != N
            or plan["inclusion_probability"] != n / N
            or [row["frame_index"] for row in plan["units"]] != expected
            or [row["id"] for row in plan["units"]] != [units[i]["id"] for i in expected]
            or any(row["inclusion_probability"] != n / N for row in plan["units"])):
        raise ValueError("Export does not match its fixed SRS sampling design")
    prediction_by_id = {row["id"]: row["components"] for row in predictions["units"]}
    reference_by_id = {row["id"]: row["components"] for row in annotations["rows"]}
    if (len(predictions["units"]) != N or set(prediction_by_id) != {row["id"] for row in units}
            or len(annotations["rows"]) != n or set(reference_by_id) != {row["id"] for row in plan["units"]}):
        raise ValueError("Export must contain unique full-frame predictions and only the complete sampled references")
    calculated = estimate_areas([prediction_by_id[row["id"]] for row in units],
                               [row["area_pixels"] for row in units], expected,
                               [reference_by_id[row["id"]] for row in plan["units"]], plan["method"], plan["alpha"])
    # Additive explanatory fields may appear in newer code. Every original
    # numerical result, including diagnostics, must still be reproduced exactly.
    def compare(original, recomputed):
        if isinstance(original, dict):
            return isinstance(recomputed, dict) and all(k in recomputed and compare(v, recomputed[k]) for k, v in original.items())
        if isinstance(original, list):
            return isinstance(recomputed, list) and len(original) == len(recomputed) and all(compare(a, b) for a, b in zip(original, recomputed))
        return original == recomputed
    original_fields = {key: saved[key] for key in calculated if key in saved}
    if not compare(original_fields, calculated):
        raise ValueError("Export numerical results do not reproduce")
    return calculated


class SpatialStudy:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.lock = threading.RLock()
        self.config = read(self.directory / "study.json")
        self.frame = read(self.directory / "frame.json")
        self.predictions = read(self.directory / "predictions.json")
        self.reference_sources = read(self.directory / "reference-sources.json")
        for artifact in (self.config, self.frame, self.predictions, self.reference_sources):
            verify(artifact)
        if (self.config["frame_sha256"] != self.frame["sha256"] or self.predictions["frame_sha256"] != self.frame["sha256"]
                or self.reference_sources["frame_sha256"] != self.frame["sha256"]
                or self.config["prediction_sha256"] != self.predictions["sha256"]
                or self.config["reference_sources_sha256"] != self.reference_sources["sha256"]):
            raise ValueError("Spatial artifact provenance mismatch")
        self.rows = self.frame["units"]
        self.by_id = {row["id"]: row for row in self.rows}
        self.rois = {row["id"]: row for row in self.frame["rois"]}
        self.predicted_by_id = {row["id"]: row["components"] for row in self.predictions["units"]}
        self.reference_by_roi = {row["roi_id"]: row for row in self.reference_sources["rois"]}
        if len(self.by_id) != len(self.rows) or set(self.predicted_by_id) != set(self.by_id):
            raise ValueError("Spatial prediction/frame IDs differ")
        if set(self.reference_by_roi) != set(self.rois):
            raise ValueError("Reference source/frame ROI IDs differ")
        self.areas = np.array([row["area_pixels"] for row in self.rows], dtype=int)
        self.predicted = _validate_components([self.predicted_by_id[row["id"]] for row in self.rows], self.areas, "predicted")

    def new_plan(self, budget, method="uniform", seed=0, alpha=.05):
        if type(budget) is not int or not 1 <= budget <= len(self.rows) or method not in METHODS or not 0 < alpha < 1:
            raise ValueError("Invalid fixed spatial audit budget/method/alpha")
        if type(seed) is not int or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        N = len(self.rows)
        indices = np.random.default_rng(seed).choice(N, size=budget, replace=False).tolist()
        plan = seal({"schema": "spatial-audit-plan-v1", "id": uuid.uuid4().hex, "created_at": now(),
                     "study_sha256": self.config["sha256"], "frame_sha256": self.frame["sha256"],
                     "prediction_sha256": self.predictions["sha256"], "method": method, "seed": seed,
                     "budget": budget, "population_size": N, "alpha": alpha,
                     "design": "Fixed-size simple random sampling without replacement from image windows",
                     "inclusion_probability": budget / N,
                     "pairwise_inclusion_probability": budget * (budget - 1) / (N * (N - 1)) if N > 1 else None,
                     "units": [{"id": self.rows[i]["id"], "frame_index": i, "inclusion_probability": budget / N} for i in indices],
                     "reference_reuse": "Each reviewed window supplies all four components and all three endpoints",
                     "optional_stopping": False, "interval_source": SOURCE})
        folder = self.directory / "audits" / plan["id"]
        with self.lock:
            write(folder / "plan.json", plan)
            write(folder / "annotations.json", seal({"schema": "spatial-annotations-v1", "plan_sha256": plan["sha256"], "rows": []}))
        return plan

    def get_plan(self, plan_id):
        if len(plan_id) != 32 or any(c not in "0123456789abcdef" for c in plan_id):
            raise ValueError("Invalid spatial audit ID")
        plan = read(self.directory / "audits" / plan_id / "plan.json")
        verify(plan)
        if (plan["study_sha256"] != self.config["sha256"] or plan["frame_sha256"] != self.frame["sha256"]
                or plan["prediction_sha256"] != self.predictions["sha256"]):
            raise ValueError("Spatial audit belongs to another frozen study")
        expected = np.random.default_rng(plan["seed"]).choice(len(self.rows), size=plan["budget"], replace=False).tolist()
        if ([unit["frame_index"] for unit in plan["units"]] != expected
                or [unit["id"] for unit in plan["units"]] != [self.rows[i]["id"] for i in expected]
                or any(unit["inclusion_probability"] != plan["budget"] / len(self.rows) for unit in plan["units"])):
            raise ValueError("Spatial audit does not match its recorded SRS design")
        return plan

    def _annotations(self, plan):
        artifact = read(self.directory / "audits" / plan["id"] / "annotations.json")
        verify(artifact)
        if artifact["plan_sha256"] != plan["sha256"]:
            raise ValueError("Annotation plan hash mismatch")
        ids = [row["id"] for row in artifact["rows"]]
        if len(set(ids)) != len(ids) or not set(ids) <= {unit["id"] for unit in plan["units"]}:
            raise ValueError("Only unique sampled windows may be annotated")
        for row in artifact["rows"]:
            _validate_components([row["components"]], [self.by_id[row["id"]]["area_pixels"]], "annotated")
        return artifact

    def annotations(self, plan_id):
        return self._annotations(self.get_plan(plan_id))["rows"]

    def annotate(self, plan_id, unit_id=None, *, simulate=False, simulate_all=False, components=None):
        with self.lock:
            plan = self.get_plan(plan_id)
            artifact = self._annotations(plan)
            previous = {row["id"]: row for row in artifact["rows"]}
            allowed = {row["id"] for row in plan["units"]}
            if simulate_all:
                targets, simulate = [row["id"] for row in plan["units"] if row["id"] not in previous], True
            else:
                if unit_id not in allowed:
                    raise ValueError("Only selected image windows can be reviewed")
                targets = [unit_id]
            mask_cache = {}
            for identifier in targets:
                unit = self.by_id[identifier]
                source = self.reference_by_roi[unit["roi_id"]]
                if simulate:
                    if unit["roi_id"] not in mask_cache:
                        check_file(source["path"], source["sha256"])
                        with Image.open(source["path"]) as image:
                            mask = np.array(image)
                        if mask.shape != (*reversed(self.rois[unit["roi_id"]]["size_px"]), 3):
                            raise ValueError("Reference mask/source geometry mismatch")
                        mask_cache[unit["roi_id"]] = mask[..., source["channel"]]
                    values = components_from_mask(_crop(mask_cache[unit["roi_id"]], unit["bounds"]))
                else:
                    values = [components[name] for name in COMPONENTS] if isinstance(components, dict) else components
                    values = _validate_components([values], [unit["area_pixels"]], "human annotation")[0].astype(int).tolist()
                if identifier in previous:
                    if previous[identifier]["components"] != values:
                        raise ValueError("Completed spatial annotations are immutable")
                    continue
                artifact["rows"].append({"id": identifier, "components": values, "role": "audit", "created_at": now(),
                                         "source": "simulated_manual_reference" if simulate else "human_entered_pixel_components",
                                         "reference_mask_sha256": source["sha256"] if simulate else None,
                                         "plan_sha256": plan["sha256"]})
            if targets:
                artifact = seal({k: v for k, v in artifact.items() if k != "sha256"})
                write(self.directory / "audits" / plan_id / "annotations.json", artifact, overwrite=True)
            if len(artifact["rows"]) == plan["budget"]:
                result_path = self.directory / "audits" / plan_id / "result.json"
                if not result_path.exists():
                    result = self._estimate(plan, artifact)
                    write(result_path, result)
            return artifact["rows"]

    def _estimate(self, plan, annotations):
        by_id = {row["id"]: row["components"] for row in annotations["rows"]}
        if set(by_id) != {unit["id"] for unit in plan["units"]}:
            raise ValueError("Complete the fixed audit before estimation")
        result = estimate_areas(self.predicted, self.areas,
                                [row["frame_index"] for row in plan["units"]],
                                [by_id[row["id"]] for row in plan["units"]], plan["method"], plan["alpha"])
        return seal(dict(result, created_at=now(), study_sha256=self.config["sha256"], plan_sha256=plan["sha256"],
                         annotation_sha256=annotations["sha256"],
                         annotation_sources=sorted({row["source"] for row in annotations["rows"]})))

    def result(self, plan_id):
        plan = self.get_plan(plan_id)
        path = self.directory / "audits" / plan_id / "result.json"
        if not path.exists():
            return None
        result = read(path)
        verify(result)
        annotations = self._annotations(plan)
        if (result["plan_sha256"] != plan["sha256"] or result["annotation_sha256"] != annotations["sha256"]
                or result["study_sha256"] != self.config["sha256"]):
            raise ValueError("Result provenance differs from frozen audit")
        return result

    def reference_crop(self, plan_id, unit_id):
        """Return central reference labels only after this sampled unit is revealed."""
        plan = self.get_plan(plan_id)
        if unit_id not in {unit["id"] for unit in plan["units"]}:
            raise ValueError("Reference evidence is restricted to sampled windows")
        if unit_id not in {row["id"] for row in self._annotations(plan)["rows"]}:
            raise ValueError("Reveal this selected window before viewing reference evidence")
        unit = self.by_id[unit_id]
        source = self.reference_by_roi[unit["roi_id"]]
        check_file(source["path"], source["sha256"])
        with Image.open(source["path"]) as image:
            labels = np.array(image)[..., source["channel"]]
        return _crop(labels, unit["bounds"]).copy()

    def audit_summaries(self):
        summaries = []
        for path in sorted((self.directory / "audits").glob("*/plan.json")):
            plan = self.get_plan(path.parent.name)
            summaries.append({"id": plan["id"], "method": plan["method"], "budget": plan["budget"],
                              "created_at": plan["created_at"], "reviewed": len(self._annotations(plan)["rows"])})
        return sorted(summaries, key=lambda row: row["created_at"], reverse=True)

    def export(self, plan_id):
        plan, result = self.get_plan(plan_id), self.result(plan_id)
        if result is None:
            raise ValueError("Complete the fixed audit before export")
        return seal({"schema": "spatial-audit-export-v1", "study": self.config, "frame": self.frame,
                     "predictions": self.predictions, "plan": plan,
                     "annotations": self._annotations(plan), "result": result,
                     "reference_policy": "Export contains selected-window audit components only, not unreviewed reference values"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--smoke-dir", required=True)
    prepare.add_argument("--study-dir", required=True)
    audit = commands.add_parser("simulate")
    audit.add_argument("--study-dir", required=True)
    audit.add_argument("--budget", type=int, default=24)
    audit.add_argument("--method", choices=METHODS, default="uniform")
    audit.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.command == "prepare":
        config = prepare_spatial_study(args.smoke_dir, args.study_dir)
        print(config["sha256"])
    else:
        study = SpatialStudy(args.study_dir)
        plan = study.new_plan(args.budget, args.method, args.seed)
        study.annotate(plan["id"], simulate_all=True)
        write(Path(args.study_dir) / "audits" / plan["id"] / "export.json", study.export(plan["id"]))
        print(plan["id"])


if __name__ == "__main__":
    main()
