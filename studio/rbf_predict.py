"""Numeric-only frozen multiclass libsvm RBF prediction with original vote ties."""
import numpy as np

TIE_POLICY = "First class in frozen classifier.classes order among classes with the maximum vote count (libsvm default)"
SCORE_SCOPE = "Raw one-versus-one RBF SVM class vote counts; not probabilities or calibrated confidence. Zero pairwise decisions vote for the second class; tied maximum class votes choose the first frozen class."
KEYS = ("classes", "support_vectors", "dual_coefficients", "intercept", "n_support", "gamma")


def export(model):
    if model.kernel != "rbf" or model.break_ties or len(model.classes_) < 3:
        raise ValueError("Adapter requires multiclass RBF SVC with default libsvm vote ties")
    return {"classes": model.classes_.copy(), "support_vectors": model.support_vectors_.copy(),
            "dual_coefficients": model.dual_coef_.copy(), "intercept": model.intercept_.copy(),
            "n_support": model.n_support_.copy(), "gamma": np.asarray(model._gamma)}


def validate(state, classes, dimension):
    if set(state) != set(KEYS):
        raise ValueError("Unexpected numeric RBF model fields")
    count = len(classes)
    vectors, dual, intercept = (state[key] for key in ("support_vectors", "dual_coefficients", "intercept"))
    support = state["n_support"]
    if (count < 3 or state["classes"].tolist() != classes or
            vectors.ndim != 2 or vectors.shape[1] != dimension or not len(vectors) or
            dual.shape != (count - 1, len(vectors)) or intercept.shape != (count * (count - 1) // 2,) or
            support.shape != (count,) or not np.issubdtype(support.dtype, np.integer) or
            np.any(support <= 0) or int(support.sum()) != len(vectors) or
            state["gamma"].shape != () or not float(state["gamma"]) > 0 or
            any(not np.isfinite(state[key]).all() for key in KEYS if key != "classes")):
        raise ValueError("Invalid numeric RBF support-vector state")


def votes_from_decisions(decisions, classes):
    """Match libsvm: >0 favors i, zero/negative favors j; first max wins."""
    if decisions.ndim != 2 or decisions.shape[1] != classes * (classes - 1) // 2:
        raise ValueError("Invalid pairwise decision shape")
    votes = np.zeros((len(decisions), classes), dtype=np.int64)
    pair = 0
    for i in range(classes):
        for j in range(i + 1, classes):
            positive = decisions[:, pair] > 0
            votes[:, i] += positive
            votes[:, j] += ~positive
            pair += 1
    return votes


def score(features, state, *, batch_size=128):
    from scipy.spatial.distance import cdist

    classes = state["classes"].tolist()
    validate(state, classes, features.shape[1])
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("RBF features must be a finite matrix")
    vectors, dual = state["support_vectors"], state["dual_coefficients"]
    starts = np.r_[0, np.cumsum(state["n_support"])]
    decisions = np.empty((len(features), len(classes) * (len(classes) - 1) // 2))
    for start in range(0, len(features), batch_size):
        kernel = np.exp(-float(state["gamma"]) * cdist(features[start:start + batch_size], vectors, "sqeuclidean"))
        pair = 0
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                a, b = slice(starts[i], starts[i + 1]), slice(starts[j], starts[j + 1])
                decisions[start:start + batch_size, pair] = (kernel[:, a] @ dual[j - 1, a] +
                    kernel[:, b] @ dual[i, b] + state["intercept"][pair])
                pair += 1
    return votes_from_decisions(decisions, len(classes)), decisions


def row_scores(votes):
    """Predictions use first-max votes, including exact zero-margin ties."""
    return {"vote_tie": bool(np.count_nonzero(votes == votes.max()) > 1),
            "tie_break_policy": TIE_POLICY}
