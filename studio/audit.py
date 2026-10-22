"""Design-based estimation for a fixed image collection and frozen predictor.

Exact conservative intervals invert finite-population hypergeometric tails.
Sample variance is reported as a diagnostic, never used as a coverage guarantee.
"""
from functools import lru_cache
import uuid

import numpy as np
from scipy.stats import hypergeom

from .prepare import CLASSES, ENDPOINTS
from .storage import digest, now

METHODS = ("manual", "uniform", "stratified", "shared")


def indicator(labels):
    return np.array([[int(label == k) for k in ENDPOINTS] for label in labels], dtype=float)


def stratum(label):
    return label if label in ENDPOINTS else "OTHER"


def groups_for(rows, method):
    if method not in METHODS:
        raise ValueError("Unknown audit method")
    groups = {}
    for row in rows:
        key = "ALL" if method in ("manual", "uniform") else stratum(row["predicted_label"])
        groups.setdefault(key, []).append(row)
    return dict(sorted(groups.items()))


def pilot_variances(pilot_rows, pilot_labels, keys):
    """Jeffreys-smoothed pilot class frequencies; used for allocation only."""
    estimates = {}
    for key in keys:
        labels = [pilot_labels[r["id"]] for r in pilot_rows
                  if stratum(r["predicted_label"]) == key and r["id"] in pilot_labels]
        values = indicator(labels) if labels else np.zeros((0, len(ENDPOINTS)))
        probability = (values.sum(axis=0) + 0.5) / (len(labels) + 1)
        estimates[key] = probability * (1 - probability)
    return estimates


def allocation(groups, budget, method, pilot_rows=(), pilot_labels=None):
    sizes = {h: len(rows) for h, rows in groups.items()}
    total = sum(sizes.values())
    if not isinstance(budget, int) or budget > total:
        raise ValueError("Budget must be an integer no greater than the study population")
    counts = {h: min(2, size) for h, size in sizes.items()}
    if budget < sum(counts.values()):
        raise ValueError(f"Budget must be at least {sum(counts.values())} for this design")
    variances = pilot_variances(pilot_rows, pilot_labels or {}, sizes) if method == "shared" else None
    while sum(counts.values()) < budget:
        candidates = [h for h in sizes if counts[h] < sizes[h]]
        if variances is None:
            chosen = max(candidates, key=lambda h: (budget * sizes[h] / total - counts[h], h))
        else:
            # Equal predefined precision scales, all endpoints in proportion units.
            # Finite-population terms constant in n_h do not affect this choice.
            current = sum(sizes[h] ** 2 * variances[h] *
                          (1 / counts[h] - 1 / sizes[h]) for h in sizes)
            def objective(h):
                after = current - sizes[h] ** 2 * variances[h] / (counts[h] * (counts[h] + 1))
                return (float(after.max()), float(after.sum()), h)
            chosen = min(candidates, key=objective)
        counts[chosen] += 1
    return counts


def make_plan(rows, budget, method, seed, *, prediction_hash, manifest_hash,
              pilot_rows=(), pilot_labels=None):
    if len({r["id"] for r in rows}) != len(rows) or not rows:
        raise ValueError("The population must contain unique IDs")
    if any(r.get("role") != "study" for r in rows):
        raise ValueError("Final audit population must contain only study units")
    pilot_ids = {r["id"] for r in pilot_rows}
    if (pilot_ids & {r["id"] for r in rows} or
            any(r.get("role") != "pilot" for r in pilot_rows) or
            set(pilot_labels or {}) != pilot_ids):
        raise ValueError("Allocation pilot must be disjoint and separately labeled")
    groups = groups_for(rows, method)
    counts = allocation(groups, budget, method, pilot_rows, pilot_labels)
    rng = np.random.default_rng(seed)
    units, strata = [], []
    for h, members in groups.items():
        n_h, N_h = counts[h], len(members)
        strata.append({"name": h, "N": N_h, "n": n_h})
        for index in rng.choice(N_h, size=n_h, replace=False):
            units.append({"id": members[int(index)]["id"], "stratum": h,
                          "inclusion_probability": n_h / N_h})
    rng.shuffle(units)
    plan = {"schema_version": 1, "id": uuid.uuid4().hex, "created_at": now(),
            "method": method, "seed": int(seed), "budget": budget, "population_size": len(rows),
            "manifest_sha256": manifest_hash, "prediction_sha256": prediction_hash,
            "design": "Independent simple random sampling without replacement within frozen strata",
            "strata": strata, "units": units, "alpha": 0.05, "endpoints": ENDPOINTS,
            "pilot_labels_available": len(pilot_labels or {}),
            "total_labels_including_common_pilot": budget + len(pilot_labels or {}),
            "pilot_policy": "Disjoint common allocation pilot, charged equally to every comparison method",
            "pilot_sha256": digest({"predictions": list(pilot_rows), "labels": pilot_labels or {}}),
            "normalization": "Equal fixed precision scales of 1.0 for three proportions",
            "allocation_objective": "Greedy minimax pilot variance (experimental)" if method == "shared"
                                    else "Proportional allocation with minimum two per noncensus stratum"}
    plan["sha256"] = digest(plan)
    return plan


@lru_cache(maxsize=100000)
def count_bounds(N, n, x, alpha):
    """Two-sided exact conservative CI for number of positives in N units."""
    if not (isinstance(N, int) and isinstance(n, int) and isinstance(x, int)
            and 0 <= x <= n <= N and 0 < alpha < 1):
        raise ValueError("Invalid finite-population confidence interval arguments")
    if n == N:
        return x, x
    if n == 0:
        return 0, N
    candidates = np.arange(x, N - n + x + 1, dtype=int)
    accepted = ((hypergeom.cdf(x, N, candidates, n) >= alpha / 2) &
                (hypergeom.sf(x - 1, N, candidates, n) >= alpha / 2))
    retained = candidates[accepted]
    if not len(retained):
        raise ArithmeticError("Unexpected empty confidence set")
    return int(retained[0]), int(retained[-1])


def estimate(rows, plan, labels):
    """labels includes only completed, selected final-audit annotations."""
    lookup = {r["id"]: r for r in rows}
    selected = [unit["id"] for unit in plan["units"]]
    if len(set(selected)) != len(selected) or set(selected) - lookup.keys():
        raise ValueError("Audit IDs must be unique members of the frozen population")
    if set(labels) != set(selected):
        raise ValueError("Complete every planned review before computing final intervals")
    if any(label not in CLASSES for label in labels.values()):
        raise ValueError("Unknown reference class")
    N, K = len(rows), len(ENDPOINTS)
    if N != plan["population_size"] or len(selected) != plan["budget"]:
        raise ValueError("Population or sample size mismatch")
    method, alpha = plan["method"], plan["alpha"]
    groups = groups_for(rows, method)
    if (set(groups) != {s["name"] for s in plan["strata"]} or
            len(groups) != len(plan["strata"]) or
            any(u["stratum"] not in groups for u in plan["units"])):
        raise ValueError("Unknown, missing, or duplicate strata")
    predicted_total = indicator([r["predicted_label"] for r in rows]).sum(axis=0)
    total = np.zeros(K) if method == "manual" else predicted_total.copy()
    covariance = np.zeros((K, K))
    lower, upper = np.zeros(K), np.zeros(K)
    for h, members in groups.items():
        N_h = len(members)
        ids = [unit["id"] for unit in plan["units"] if unit["stratum"] == h]
        n_h = len(ids)
        specification = next(s for s in plan["strata"] if s["name"] == h)
        if specification["N"] != N_h or specification["n"] != n_h or n_h < min(2, N_h):
            raise ValueError("Inconsistent stratum sample size")
        for unit in plan["units"]:
            if unit["stratum"] == h:
                expected = "ALL" if method in ("manual", "uniform") else stratum(lookup[unit["id"]]["predicted_label"])
                if expected != h or not np.isclose(unit["inclusion_probability"], n_h / N_h):
                    raise ValueError("Incorrect stratum or inclusion probability")
        y = indicator([labels[i] for i in ids])
        p = indicator([lookup[i]["predicted_label"] for i in ids])
        residual = y if method == "manual" else y - p
        total += N_h * residual.mean(axis=0)
        if n_h < N_h:
            covariance += N_h ** 2 * (1 - n_h / N_h) * np.cov(residual, rowvar=False, ddof=1) / n_h
        if method == "uniform":
            # Six population error-count intervals: FN and FP for each endpoint.
            for k in range(K):
                fn = int(((y[:, k] == 1) & (p[:, k] == 0)).sum())
                fp = int(((y[:, k] == 0) & (p[:, k] == 1)).sum())
                lfn, ufn = count_bounds(N, n_h, fn, alpha / (2 * K))
                lfp, ufp = count_bounds(N, n_h, fp, alpha / (2 * K))
                ufn = min(ufn, int(N - predicted_total[k]))
                ufp = min(ufp, int(predicted_total[k]))
                lower[k] = max(0, predicted_total[k] + lfn - ufp)
                upper[k] = min(N, predicted_total[k] + ufn - lfp)
        else:
            for k in range(K):
                lo, hi = count_bounds(N_h, n_h, int(y[:, k].sum()), alpha / (len(groups) * K))
                lower[k] += lo
                upper[k] += hi
    estimates = [{"endpoint": key, "initial": float(predicted_total[k] / N),
                  "estimate": float(total[k] / N), "lower": float(lower[k] / N),
                  "upper": float(upper[k] / N), "width": float((upper[k] - lower[k]) / N)}
                 for k, key in enumerate(ENDPOINTS)]
    return {"endpoints": estimates, "covariance": (covariance / N ** 2).tolist(),
            "interval_method": "Exact conservative hypergeometric inversion with Bonferroni multiplicity control",
            "coverage_scope": "At least 95% simultaneous coverage of three finite-collection reference-label proportions for this fixed audit",
            "covariance_note": "Design-unbiased covariance estimate; not used to construct these intervals",
            "model_assistance_note": "With constant hard predictions within predicted-class strata, correction equals stratified reference-label estimation",
            "reviewed": len(selected), "population": N, "method": method,
            "plan_sha256": plan["sha256"], "prediction_sha256": plan["prediction_sha256"]}
