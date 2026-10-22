"""Frozen, stratified finite-population audits of three joint pixel totals.

No image/reference files are opened here. Pilot annotations choose allocations;
only final sampled references enter estimation. Approximate Fieller intervals
and conservative bounded-population intervals are reported separately.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy.stats import t

from .storage import digest

COMPONENTS = ("epithelial", "dense_lymphocytic_tissue", "eligible")
ENDPOINTS = ("epithelial_fraction", "dense_lymphocytic_fraction")
NAMES = ("Epithelial area / eligible area", "Dense lymphocytic tissue area / eligible area")
METHODS = ("manual", "uniform", "stratified", "shared")
SOURCES = {
    "stratified_covariance": "https://www150.statcan.gc.ca/n1/pub/12-001-x/2019002/article/00006/02-eng.htm",
    "fieller": "https://arxiv.org/abs/0711.0198",
    "serfling": "https://arxiv.org/abs/1309.4029",
}


def seal(value):
    if "sha256" in value:
        raise ValueError("Artifact already sealed")
    return dict(value, sha256=digest(value))


def verify(value):
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("Spatial audit artifact hash mismatch")


def array_hash(value):
    value = np.asarray(value, dtype="<f8", order="C")
    h = hashlib.sha256(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def validate_components(values, areas, label="components"):
    areas = np.asarray(areas, dtype=float)
    values = np.asarray(values, dtype=float)
    if (areas.ndim != 1 or not len(areas) or not np.isfinite(areas).all()
            or np.any(areas <= 0) or np.any(areas != np.floor(areas))):
        raise ValueError("Areas must be positive native-pixel counts")
    if (values.shape != (len(areas), 3) or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values != np.floor(values))
            or np.any(values[:, 0] + values[:, 1] > values[:, 2])
            or np.any(values[:, 2] > areas)):
        raise ValueError(f"Invalid {label}: expected joint E,L,D native-pixel counts")
    return values, areas


def stratum_labels(predicted):
    """Fixed prediction-only quadrants; no reference-defined strata."""
    p = np.asarray(predicted)
    labels = np.zeros(len(p), dtype=int)
    positive = p[:, 2] > 0
    # Avoid ratio rounding exactly at the predeclared one-fifth threshold.
    labels[positive] = 1 + (5 * p[positive, 0] >= p[positive, 2]) * 2 + (5 * p[positive, 1] >= p[positive, 2])
    return labels


def _objective(n, sizes, variances):
    return ((sizes * sizes * (1 / n - 1 / sizes))[:, None] * variances).sum(axis=0)


def allocate(sizes, variances, budget, method):
    """Deterministic bounded integer allocation; shared minimax is a heuristic."""
    sizes = np.asarray(sizes, dtype=int)
    variances = np.asarray(variances, dtype=float)
    minimum = np.minimum(4, sizes)
    if method not in ("stratified", "shared") or not minimum.sum() <= budget <= sizes.sum():
        raise ValueError("Invalid allocation method or budget")
    if variances.ndim != 2 or len(variances) != len(sizes) or np.any(variances < 0) or not np.isfinite(variances).all():
        raise ValueError("Invalid pilot allocation variances")
    n = minimum.copy()

    def key(counts):
        v = _objective(counts, sizes, variances)
        return (float(v.sum()),) if method == "stratified" else (float(v.max()), float(v.sum()))

    while n.sum() < budget:
        candidates = []
        for h in np.flatnonzero(n < sizes):
            trial = n.copy()
            trial[h] += 1
            candidates.append((key(trial), float(n[h] / sizes[h]), int(h)))
        _, _, h = min(candidates)
        n[h] += 1
    if method == "shared":
        # Deterministic one-unit exchanges improve the lexicographic objective;
        # this is not a claim of a globally optimal integer minimax solution.
        for _ in range(10000):
            current, best = key(n), None
            for donor in np.flatnonzero(n > minimum):
                for receiver in np.flatnonzero(n < sizes):
                    if donor == receiver:
                        continue
                    trial = n.copy()
                    trial[donor] -= 1
                    trial[receiver] += 1
                    candidate = (key(trial), int(donor), int(receiver))
                    if candidate[0] < current and (best is None or candidate < best):
                        best = candidate
            if best is None:
                break
            n[best[1]] -= 1
            n[best[2]] += 1
        else:
            raise RuntimeError("Allocation exchange limit exceeded")
    return n


def freeze_design(predicted, areas, pilot_predicted, pilot_reference, pilot_areas, *,
                  endpoint_ids=ENDPOINTS, budgets=(48, 96, 160), alpha=.05,
                  seed=20261029, repetitions=5000, provenance=None):
    """Freeze allocation and evaluation protocol, without final references.

    The caller must establish pilot/final image and slide disjointness and bind
    those manifests in provenance. No pilot values enter the final estimator.
    """
    p, a = validate_components(predicted, areas, "predictions")
    pp, pa = validate_components(pilot_predicted, pilot_areas, "pilot predictions")
    pr, _ = validate_components(pilot_reference, pa, "pilot references")
    endpoint_ids, budgets = tuple(endpoint_ids), tuple(budgets)
    if (not endpoint_ids or len(set(endpoint_ids)) != len(endpoint_ids)
            or any(x not in ENDPOINTS for x in endpoint_ids) or not 0 < alpha < 1
            or type(seed) is not int or seed < 0 or type(repetitions) is not int or repetitions < 2):
        raise ValueError("Invalid predefined endpoint/evaluation contract")
    labels, plabels = stratum_labels(p), stratum_labels(pp)
    strata = sorted(set(labels.tolist()))
    sizes = np.array([np.count_nonzero(labels == h) for h in strata])
    minimum = int(np.minimum(4, sizes).sum())
    if (not budgets or len(set(budgets)) != len(budgets)
            or any(type(b) is not int or not minimum <= b <= len(p) for b in budgets)):
        raise ValueError(f"Budgets must be distinct integers in [{minimum},{len(p)}]")
    indices = [ENDPOINTS.index(x) for x in endpoint_ids]
    total = p.sum(axis=0)
    ratio0 = total[indices] / total[2] if total[2] > 0 else np.zeros(len(indices))
    scale = max(1.0, float(total[2])) if total[2] > 0 else float(a.sum())
    residual = pr - pp
    influence = (residual[:, indices] - residual[:, 2, None] * ratio0) / (scale * .05)
    pooled = np.var(influence, axis=0, ddof=1) if len(pp) > 1 else np.zeros(len(indices))
    variances, pilot_counts = [], []
    for h in strata:
        x = influence[plabels == h]
        m = len(x)
        weight = (m - 1) / (m - 1 + 8) if m > 1 else 0.0
        local = np.var(x, axis=0, ddof=1) if m > 1 else pooled
        variances.append(weight * local + (1 - weight) * pooled)
        pilot_counts.append(m)
    variances = np.asarray(variances)
    allocations = {}
    for method in METHODS:
        allocations[method] = {}
        for budget in budgets:
            n = [budget] if method in ("manual", "uniform") else allocate(sizes, variances, budget, method).tolist()
            allocations[method][str(budget)] = {
                "sample_sizes": n,
                "anticipated_normalized_variances": None if method in ("manual", "uniform") else _objective(np.array(n), sizes, variances).tolist(),
            }
    return seal({
        "schema": "shared-spatial-design-v1", "components": list(COMPONENTS),
        "population_size": len(p), "pilot_units": len(pp), "pilot_native_pixels": int(pa.sum()),
        "prediction_sha256": array_hash(p), "areas_sha256": array_hash(a),
        "pilot_prediction_sha256": array_hash(pp), "pilot_reference_sha256": array_hash(pr),
        "pilot_areas_sha256": array_hash(pa), "endpoint_ids": list(endpoint_ids),
        "endpoint_scales": {name: .05 for name in endpoint_ids},
        "scale_interpretation": "Five percentage points for engineering comparison; not a clinical threshold",
        "alpha": alpha, "budgets": list(budgets), "primary_budget": 96 if 96 in budgets else budgets[0],
        "methods": list(METHODS), "seed": seed, "repetitions": repetitions,
        "stratum_assignments": labels.tolist(), "stratum_ids": strata,
        "stratum_sizes": sizes.tolist(), "pilot_stratum_counts": pilot_counts,
        "stratification": "D=0; otherwise quadrants E/D >= .20 and L/D >= .20; prediction only",
        "minimum_per_stratum": 4, "minimum_budget": minimum,
        "pilot_variance_shrinkage_df": 8, "pilot_normalized_influence_variances": variances.tolist(),
        "planning_ratios": ratio0.tolist(), "planning_denominator": scale,
        "allocations": allocations,
        "allocation_rules": {
            "manual": "Single-stratum simple random sampling without replacement; no predictor in estimator",
            "uniform": "Single-stratum simple random sampling without replacement; fixed model-assisted estimator",
            "stratified": "Integer multivariate Neyman allocation minimizing sum of anticipated normalized variances by greedy marginal reduction",
            "shared": "Greedy minimax anticipated normalized endpoint variance followed by improving one-unit exchanges; heuristic",
        },
        "sampling": "Independent SRSWOR within frozen strata using ranks of one random permutation of final windows; identical seed pairs methods",
        "pilot_cost": "Every method is charged the same complete pilot plus final budget; all labels reused across endpoints",
        "annotation_level": "Joint complete E,L,D native-pixel areas in central window; surrounding context is not counted",
        "interval_protocol": "Approximate Bonferroni simultaneous Fieller confidence sets; survey df=sum(n_h-1) over noncensus strata; separate conservative Serfling bounds",
        "evaluation_protocol": {"primary": "Empirical joint coverage and mean confidence-set hull width, with endpoint RMSE and failure rates",
                                "empty_set_width": 1.0, "weak_denominator_width": "Physical confidence-set hull, up to1; report unbounded/disjoint counts separately",
                                "undefined_point_errors": "Exclude from conditional RMSE/bias and explicitly count failures; never silently drop",
                                "width_claim": "Do not interpret shorter widths as improvement when joint coverage is inadequate",
                                "reference_scope": "Finite final image windows only; no patient or population generalization",
                                "seeds": "seed + repetition, reused across methods and budgets; budget results may be correlated"},
        "provenance": provenance or {}, "sources": SOURCES,
    })


def sample_plan(design, method, budget, seed):
    verify(design)
    if method not in METHODS or budget not in design["budgets"] or type(seed) is not int or seed < 0:
        raise ValueError("Sample request differs from frozen design")
    N = design["population_size"]
    labels = np.zeros(N, dtype=int) if method in ("manual", "uniform") else np.asarray(design["stratum_assignments"])
    strata = sorted(set(labels.tolist()))
    allocation = design["allocations"][method][str(budget)]["sample_sizes"]
    priority = np.empty(N, dtype=int)
    priority[np.random.default_rng(seed).permutation(N)] = np.arange(N)
    selected, rows, groups = [], [], []
    for h, n in zip(strata, allocation, strict=True):
        members = np.flatnonzero(labels == h)
        Nh = len(members)
        picked = sorted(members[np.argsort(priority[members], kind="stable")[:n]].tolist())
        pi = n / Nh
        group = {"stratum": int(h), "population_size": Nh, "sample_size": n,
                 "inclusion_probability": pi,
                 "pair_inclusion_probability": n * (n - 1) / (Nh * (Nh - 1)) if Nh > 1 else None}
        groups.append(group)
        for i in picked:
            selected.append(i)
            rows.append({"index": i, "stratum": int(h), "inclusion_probability": pi,
                         "reason": design["allocation_rules"][method] + "; random within stratum"})
    order = np.argsort(selected)
    selected = [selected[i] for i in order]
    rows = [rows[i] for i in order]
    return seal({"schema": "shared-spatial-sample-v1", "design_sha256": design["sha256"],
                 "method": method, "budget": budget, "seed": seed, "alpha": design["alpha"],
                 "population_size": N, "pilot_units": design["pilot_units"],
                 "endpoint_ids": design["endpoint_ids"], "endpoint_scales": design["endpoint_scales"],
                 "prediction_sha256": design["prediction_sha256"], "areas_sha256": design["areas_sha256"],
                 "stratum_assignments": labels.tolist(), "allocation": groups,
                 "indices": selected, "selected": rows,
                 "cross_stratum_pair_probability": "pi_i * pi_j (independent stratum samples)",
                 "sampling": design["sampling"], "annotation_level": design["annotation_level"],
                 "interval_protocol": design["interval_protocol"]})


def pair_inclusion_probability(plan, first, second):
    """Exact first/second order probabilities for any target-population pair."""
    verify(plan)
    labels = plan["stratum_assignments"]
    if not 0 <= first < len(labels) or not 0 <= second < len(labels):
        raise ValueError("Pair index outside population")
    groups = {x["stratum"]: x for x in plan["allocation"]}
    a, b = groups[labels[first]], groups[labels[second]]
    if first == second:
        return a["inclusion_probability"]
    return a["pair_inclusion_probability"] if labels[first] == labels[second] else a["inclusion_probability"] * b["inclusion_probability"]


def fieller_set(numerator, denominator, vnn, vnd, vdd, critical):
    """Solve the Fieller quadratic, retaining unbounded/disjoint topology.

    Returned segments are intersected with the known physical parameter [0,1].
    Unbounded/disjoint raw sets deliberately do not acquire a finite `interval`.
    """
    q2 = critical * critical
    A = denominator * denominator - q2 * vdd
    B = -2 * (numerator * denominator - q2 * vnd)
    C = numerator * numerator - q2 * vnn
    norm = max(abs(A), abs(B), abs(C), 1.0)
    A, B, C = A / norm, B / norm, C / norm
    tol = 1e-12
    if abs(A) <= tol:
        if abs(B) <= tol:
            segments, status = ([[-math.inf, math.inf]], "unbounded") if C <= tol else ([], "empty")
        elif B > 0:
            segments, status = [[-math.inf, -C / B]], "unbounded"
        else:
            segments, status = [[-C / B, math.inf]], "unbounded"
    else:
        disc = B * B - 4 * A * C
        if disc < -tol:
            segments, status = ([[-math.inf, math.inf]], "unbounded") if A < 0 else ([], "empty")
        else:
            root = math.sqrt(max(0.0, disc))
            lo, hi = sorted(((-B - root) / (2 * A), (-B + root) / (2 * A)))
            segments, status = ([[lo, hi]], "bounded") if A > 0 else ([[-math.inf, lo], [hi, math.inf]], "disjoint")
    physical = [[max(0.0, lo), min(1.0, hi)] for lo, hi in segments if max(0.0, lo) <= min(1.0, hi)]
    if not physical:
        status = "empty"
    interval = physical[0] if status == "bounded" else None
    return {"confidence_set": physical, "interval": interval, "status": status,
            "confidence_set_hull_width": physical[-1][1] - physical[0][0] if physical else 1.0}


def _validated_plan(predicted, areas, plan, reference_sample):
    verify(plan)
    p, a = validate_components(predicted, areas, "predictions")
    if (array_hash(p) != plan["prediction_sha256"] or array_hash(a) != plan["areas_sha256"]
            or len(p) != plan["population_size"]):
        raise ValueError("Frozen predictions/frame changed")
    idx = np.asarray(plan["indices"], dtype=int)
    if len(idx) != plan["budget"] or len(set(idx.tolist())) != len(idx) or np.any(idx < 0) or np.any(idx >= len(p)):
        raise ValueError("Invalid selected unit indices")
    ref, _ = validate_components(reference_sample, a[idx], "selected reference")
    labels = np.asarray(plan["stratum_assignments"])
    if (len(labels) != len(p) or plan["method"] not in METHODS or not 0 < plan["alpha"] < 1
            or not plan["endpoint_ids"] or len(set(plan["endpoint_ids"])) != len(plan["endpoint_ids"])
            or any(x not in ENDPOINTS for x in plan["endpoint_ids"])):
        raise ValueError("Invalid sampling design")
    expected_labels = np.zeros(len(p), dtype=int) if plan["method"] in ("manual", "uniform") else stratum_labels(p)
    if not np.array_equal(labels, expected_labels):
        raise ValueError("Strata differ from frozen prediction-only rule")
    groups = plan["allocation"]
    if len({g["stratum"] for g in groups}) != len(groups) or set(labels.tolist()) != {g["stratum"] for g in groups}:
        raise ValueError("Invalid stratum coverage")
    for g in groups:
        N, n = np.count_nonzero(labels == g["stratum"]), np.count_nonzero(labels[idx] == g["stratum"])
        if (N != g["population_size"] or n != g["sample_size"] or not 1 <= n <= N
                or (n < 2 and n < N) or g["inclusion_probability"] != n / N
                or g["pair_inclusion_probability"] != (n * (n - 1) / (N * (N - 1)) if N > 1 else None)):
            raise ValueError("Sampling counts or probabilities disagree")
    if len(plan["selected"]) != len(idx):
        raise ValueError("Selected metadata does not cover sample")
    priority = np.empty(len(p), dtype=int)
    priority[np.random.default_rng(plan["seed"]).permutation(len(p))] = np.arange(len(p))
    expected = []
    for g in groups:
        members = np.flatnonzero(labels == g["stratum"])
        expected.extend(members[np.argsort(priority[members])[:g["sample_size"]]].tolist())
    if idx.tolist() != sorted(expected):
        raise ValueError("Selected sample does not replay from frozen seed")
    by_group = {g["stratum"]: g for g in groups}
    for i, row in zip(idx, plan["selected"], strict=True):
        if row["index"] != i or row["stratum"] != labels[i] or row["inclusion_probability"] != by_group[labels[i]]["inclusion_probability"]:
            raise ValueError("Selected inclusion metadata differs")
    return p, a, idx, ref, labels


def estimate(predicted, areas, plan, reference_sample):
    """Estimate only from the references explicitly selected by a frozen plan."""
    p, a, idx, ref, labels = _validated_plan(predicted, areas, plan, reference_sample)
    working = np.zeros_like(p) if plan["method"] == "manual" else p
    totals = working.sum(axis=0)
    covariance, radius = np.zeros((3, 3)), np.zeros(3)
    noncensus = [g for g in plan["allocation"] if g["sample_size"] < g["population_size"]]
    df = sum(g["sample_size"] - 1 for g in noncensus)
    warnings, details = [], []
    for group in plan["allocation"]:
        h, N, n = group["stratum"], group["population_size"], group["sample_size"]
        members, sample_rows = np.flatnonzero(labels == h), np.flatnonzero(labels[idx] == h)
        e = ref[sample_rows] - working[idx[sample_rows]]
        correction = N * e.mean(axis=0)
        totals += correction
        local_cov = np.zeros((3, 3)) if n == N else N * N * (1 - n / N) * np.cov(e, rowvar=False, ddof=1) / n
        covariance += local_cov
        low = (-working[members]).min(axis=0)
        high = (a[members, None] - working[members]).max(axis=0)
        # Classical Serfling factor, valid at every noncensus sampling fraction.
        local_radius = np.zeros(3) if n == N else N * (high - low) * np.sqrt((1 - (n - 1) / N) * math.log(6 * len(noncensus) / plan["alpha"]) / (2 * n))
        radius += local_radius
        details.append(dict(group, component_correction=correction.tolist(), total_covariance=local_cov.tolist(),
                            residual_bounds=np.column_stack((low, high)).tolist(), serfling_total_radius=local_radius.tolist()))
        if n < N and not np.any(local_cov):
            warnings.append(f"zero_sample_residual_variance_stratum_{h}")
    indices = [ENDPOINTS.index(x) for x in plan["endpoint_ids"]]
    K, denominator = len(indices), float(totals[2])
    critical = float(t.ppf(1 - plan["alpha"] / (2 * K), df)) if df else 0.0
    ratio_cov = None
    if denominator > 0:
        gradient = np.zeros((K, 3))
        for k, j in enumerate(indices):
            gradient[k, j], gradient[k, 2] = 1 / denominator, -totals[j] / denominator**2
        ratio_cov = gradient @ covariance @ gradient.T
    else:
        warnings.append("nonpositive_estimated_denominator")
    lower, upper = np.maximum(0, totals - radius), np.minimum(a.sum(), totals + radius)
    # Reviewed values give deterministic bounds independently of predictions.
    observed, missing = ref.sum(axis=0), float(a.sum() - a[idx].sum())
    lower, upper = np.maximum(lower, observed), np.minimum(upper, observed + missing)
    conflict = lower > upper
    lower, upper = np.where(conflict, observed, lower), np.where(conflict, observed + missing, upper)
    if np.any(conflict):
        warnings.append("conservative_component_intersection_fallback")
    endpoints = []
    for k, j in enumerate(indices):
        point = float(totals[j] / denominator) if denominator > 0 else None
        if not noncensus:
            field = {"confidence_set": [[point, point]] if point is not None else [],
                     "interval": [point, point] if point is not None else None,
                     "status": "exact_census" if point is not None else "no_eligible_area",
                     "confidence_set_hull_width": 0.0 if point is not None else 1.0}
        else:
            field = fieller_set(totals[j], denominator, covariance[j, j], covariance[j, 2], covariance[2, 2], critical)
        if upper[2] <= 0:
            conservative = None
        elif lower[2] <= 0:
            conservative = [0.0, 1.0]
        else:
            conservative = [max(0.0, float(lower[j] / upper[2])), min(1.0, float(upper[j] / lower[2]))]
            if conservative[0] > conservative[1]:
                conservative = [0.0, 1.0]
        if not noncensus and point is not None:
            conservative = [point, point]
        local_warnings = []
        if field["status"] in ("unbounded", "disjoint", "empty"):
            local_warnings.append("approximate_fieller_" + field["status"])
        if point is not None and not 0 <= point <= 1:
            local_warnings.append("point_estimate_outside_physical_range")
        endpoints.append(dict(field, id=ENDPOINTS[j], name=NAMES[j], estimate=point,
                              standard_error=float(math.sqrt(max(0, ratio_cov[k, k]))) if ratio_cov is not None else None,
                              interval_width=(field["interval"][1] - field["interval"][0]) if field["interval"] else None,
                              conservative_interval=conservative, warnings=local_warnings))
    return {"schema": "shared-spatial-estimate-v1", "completed": True,
            "plan_sha256": plan["sha256"], "design_sha256": plan["design_sha256"],
            "method": plan["method"], "budget": len(idx), "reviewed": len(idx),
            "pilot_units": plan["pilot_units"], "total_review_units": len(idx) + plan["pilot_units"],
            "reviewed_native_pixels": int(a[idx].sum()), "components": list(COMPONENTS),
            "totals": totals.tolist(), "total_estimates": totals.tolist(),
            "total_covariance": covariance.tolist(), "ratio_covariance": ratio_cov.tolist() if ratio_cov is not None else None,
            "component_conservative_intervals": np.column_stack((lower, upper)).tolist(),
            "allocation": details, "endpoints": endpoints, "warnings": warnings,
            "alpha": plan["alpha"], "simultaneous_confidence": 1 - plan["alpha"],
            "interval_method": {"name": "Bonferroni simultaneous Fieller sets", "approximate": True,
                                "degrees_of_freedom": df, "critical_value": critical,
                                "df_rule": "sum(n_h-1) over noncensus strata", "endpoint_count": K,
                                "coverage_guaranteed": False, "finite_population_covariance": True},
            "conservative_method": {"name": "Serfling component bounds with union bound over three components and noncensus strata",
                                    "simultaneous_confidence": 1 - plan["alpha"], "conditional_on_positive_reference_denominator": True},
            "ratio_estimator_is_exactly_unbiased": False,
            "design_scope": "Fixed finite image-window collection and fixed audit size; no patient/population inference or optional stopping"}
