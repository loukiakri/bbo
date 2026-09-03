"""
Shared utilities for Bayesian Optimisation

Functions included:
- Data Loader
- Evaluation grid builder 
- Averaging of posterior samples 
- Cross validation 
- Lengthscale statistics
- PCA metrics

Conventions
-----------
- X : (n, d) array of observation inputs already scaled to the unit cube [0,1]^d
- y : (n,) array of observation outputs

"""

import numpy as np
from scipy.stats import qmc
from pathlib import Path

def load_function_data(function_number, data_path, query_round=None):
    """
    Load the original data for a black-box function and append the
    query points

    Parameters
    ----------
    function_number : int, e.g. 1, 2, ..., 8.

    data_path : str or pathlib.Path
        Path to the folder containing the function data

    query_round : int or None, optional
        How many query rounds to append.

        None (default)
            append every round present in inputs.txt / outputs.txt, i.e. the
            full history. This reproduces the original behaviour exactly.
        0
            the initial design only, with no queries appended. Useful for
            re-running the very first fit.
        k >= 1
            the initial design plus query rounds 1..k, so the returned arrays
            are exactly the data the model saw when it proposed round k+1.
            Reproducing the model behind res_f<N>_query<k>.joblib means
            passing query_round=k-1, since round k's file was fitted on
            everything up to round k-1.

        Rounds are numbered from 1 in file order: line 1 of inputs.txt is
        round 1. A value larger than the number of rounds on file raises
        ValueError rather than silently returning fewer.

    Returns
    -------
    inputs : np.ndarray
        Original inputs with the requested query points appended

    outputs : np.ndarray
        Original outputs with the requested query outputs appended

    original_inputs, original_outputs : np.ndarray
        The initial design, unchanged, for reference

    """

    data_path = Path(data_path)

    def _read_rounds(path):
        """Parse one record per line-group, returning a list of rounds.

        A record can span several physical lines, so lines are accumulated
        until the brackets balance -- the same rule as before.
        """
        rounds = []
        current = ""
        with open(path, "r") as f:
            for line in f:
                current += line

                # Check whether the brackets are balanced
                if current.count("[") == current.count("]"):
                    rounds.append(
                        eval(current, {"array": np.array, "np": np})
                    )
                    current = ""
        return rounds

    # ---------------------------------------------------------
    # 1. Load the original inputs and outputs
    # ---------------------------------------------------------

    function_folder = data_path / f"function_{function_number}"

    original_inputs = np.load(function_folder / "initial_inputs.npy")
    original_outputs = np.load(function_folder / "initial_outputs.npy")

    # ---------------------------------------------------------
    # 2. Read inputs.txt and outputs.txt
    # ---------------------------------------------------------

    input_rounds = _read_rounds(data_path / "inputs.txt")
    output_rounds = _read_rounds(data_path / "outputs.txt")

    n_rounds = min(len(input_rounds), len(output_rounds))
    if len(input_rounds) != len(output_rounds):
        raise ValueError(
            f"inputs.txt has {len(input_rounds)} rounds but outputs.txt has "
            f"{len(output_rounds)}; the two files must stay in step.")

    # ---------------------------------------------------------
    # 3. Decide how many rounds to append
    # ---------------------------------------------------------

    if query_round is None:
        n_take = n_rounds
    else:
        n_take = int(query_round)
        if n_take < 0:
            raise ValueError(
                f"query_round must be >= 0 (or None for all rounds); "
                f"got {query_round}")
        if n_take > n_rounds:
            raise ValueError(
                f"query_round={query_round} was requested but only {n_rounds} "
                f"round(s) are recorded for function {function_number}")

    # ---------------------------------------------------------
    # 4. Extract the requested query points and outputs
    # ---------------------------------------------------------

    newpt_x = []
    newpt_y = []

    for r in range(n_take):
        group_x = input_rounds[r]
        group_y = output_rounds[r]
        if len(group_x) < function_number or len(group_y) < function_number:
            raise ValueError(
                f"round {r + 1} has no entry for function {function_number} "
                f"(found {len(group_x)} input and {len(group_y)} output "
                f"entries on that line)")
        newpt_x.append(np.asarray(group_x[function_number - 1], dtype=float))
        newpt_y.append(group_y[function_number - 1])

    # ---------------------------------------------------------
    # 5. Append the new query points to the original data
    # ---------------------------------------------------------

    if not newpt_x:                       # query_round=0: initial design only
        return (np.array(original_inputs, dtype=float, copy=True),
                np.array(original_outputs, dtype=float, copy=True),
                original_inputs, original_outputs)

    newpt_x = np.vstack(newpt_x)
    if newpt_x.shape[1] != original_inputs.shape[1]:
        raise ValueError(
            f"function {function_number}: the initial design has "
            f"{original_inputs.shape[1]} dimensions but the query records "
            f"have {newpt_x.shape[1]}")

    updated_inputs = np.vstack((original_inputs, newpt_x))
    updated_outputs = np.append(original_outputs, newpt_y)

    return updated_inputs, updated_outputs, original_inputs, original_outputs

def make_eval_points(dim, bounds=None, n_per_dim=80, n_sobol=4096, seed=0,
                     upper_eps=1e-10):
    """
    Build points at which to evaluate the GP posterior / acquisition.

    dim == 2  -> regular meshgrid (n_per_dim x n_per_dim), returned flat as
                 (n_per_dim**2, 2)
                 
    dim  > 2  -> Sobol sequence of n_sobol points in the unit cube.

    bounds : (2, dim) array [[lo...],[hi...]] in the original space, or None
             to stay in the unit cube. Grid is built in the unit cube then
             mapped to bounds if given.
             
    upper_eps : the grid (in unit-cube space) is capped at 1 - upper_eps in
             every dimension, so the exact upper bound is never sampled.
             
    n_per_dim : number of points per dimension (only used for 2D)

    Returns
    -------
    pts   : (N, dim) array of evaluation points (in original space if bounds given)
    mesh  : (XX, YY) tuple for dim==2 (for plotting), else None
    """
    top = 1.0 - float(upper_eps)
    
    if dim == 2:
        g = np.linspace(0.0, top, n_per_dim, endpoint=False) # Create single dimension
        XX, YY = np.meshgrid(g, g) # Convert to grid
        unit = np.column_stack([XX.ravel(), YY.ravel()]) # Flatten and stack points as x,y
        mesh = (XX, YY)
    else:
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed) # Scramble adds randomisation to break potential patterns
        m = int(np.ceil(np.log2(max(n_sobol, 2)))) # Rounding to a power of two for Sobol
        unit = sampler.random_base2(m=m)[:n_sobol] # Capping number of points
        unit = np.minimum(unit, top) # Clipping anything at/above upper bound
        mesh = None

    if bounds is not None: #Pass bounds for inputs not in the 0-1 range
        bounds = np.asarray(bounds, dtype=float)
        lo, hi = bounds[0], bounds[1]
        pts = lo + unit * (hi - lo)
    else:
        pts = unit
    return pts, mesh

def combine_mixture(means, stds):
    """
    Combines the mean and std of hyperparameter combination samples into one
    predictive mean and std using the law of total variance:
    
         S = sets of hyperparameter samples / posterior samples
         E[y]   = mean_s averaged over s                                          
         Var[y] = E_s[var_s]  +  Var_s[mean_s]                                    
              (within-sample)   (between-sample = hyperparameter uncertainty)
    
    The second term is the hyperparameter-uncertainty contribution
    
    means, stds : (S, N) arrays (S samples, N candidate points).
    Returns combined (mean, std) each of shape (N,).
    
    """
    means = np.asarray(means)
    stds = np.asarray(stds)
    mean = means.mean(axis=0)
    within = (stds ** 2).mean(axis=0)
    between = means.var(axis=0) # spread of the sample means
    var = within + between
    return mean, np.sqrt(np.maximum(var, 0.0))

def loo_calibration(X, y, fit_predict_fn, verbose=True,
                    cv="loo", n_splits=10, seed=0):
    """
    Cross-validation:
    LOO - Leave one out
    k-fold

    Fit quality:   LOO-RMSE (lower is better)
    Calibration:   Standardised residuals z_i = (y_i - mu_-i)/sigma_-i
    
    If the model is well-calibrated, the z_i mean ~0, std ~1, and ~95% within +/-1.96.                              
    Overconfidence shows up as std(z) >> 1 and coverage well below 95% (intervals too narrow). 
    Underconfidence is the reverse.

    cv : "loo", or "kfold".
    n_splits : number of folds when K-fold is used
    
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1) # Force 1-D: a (n,1) column y

    n = len(y)

    mus = np.empty(n)
    sigmas = np.empty(n)

    if cv == "loo":
        folds = [np.array([i]) for i in range(n)]   # Each test fold = 1 point
        label = f"LOO ({n} refits)"
    else:
        # Shuffled, roughly equal folds.
        rng = np.random.default_rng(seed) # Initialise random generator
        perm = rng.permutation(n) # Shuffle order of indices
        k = int(min(n_splits, n))
        folds = [perm[i::k] for i in range(k)] # Split indices into k test folds by taking every kth element
        label = f"{k}-fold ({k} refits, n={n})"

    for test_idx in folds: # Held out fold/point
        train_mask = np.ones(n, dtype=bool) # Create/initialise mask with all True values
        train_mask[test_idx] = False # Overwrite test fold to False
        mu, sig = fit_predict_fn(X[train_mask], y[train_mask], X[test_idx]) # Fit model on training folds and predict on test folds
        mu = np.ravel(mu); sig = np.ravel(sig) # Flatten mu, sigma arrays
        m = len(test_idx)
        # Check: one prediction per held-out point, in test_idx order. A length
        # mismatch means the closure collapsed the (S,N) posterior on the wrong
        # axis (returning ~the global mean, wrong length)
        if mu.size != m or sig.size != m:
            raise ValueError(
                f"fit_predict_fn returned {mu.size} predictions for a fold of "
                f"{m} points. The posterior was likely collapsed on the wrong "
                f"axis (points vs MCMC samples). Expected (S, N) with N={m}.")
        mus[test_idx] = mu
        sigmas[test_idx] = sig

    sigmas = np.maximum(sigmas, 1e-12) # Avoid zero division
    resid = y - mus # Calculate residuals
    z = resid / sigmas # Calculate z-score

    rmse = float(np.sqrt(np.mean(resid ** 2)))
    coverage95 = float(np.mean(np.abs(z) <= 1.96))
    nlpd = float(np.mean(0.5 * np.log(2 * np.pi * sigmas ** 2) + resid ** 2 / (2 * sigmas ** 2))) # Negative log predictive density

    report = {
        "loo_rmse": rmse,
        "z_mean": float(z.mean()),
        "z_std": float(z.std(ddof=1)) if n > 1 else float("nan"),
        "coverage_95": coverage95,
        "loo_nlpd": nlpd,
        "z_scores": z,
        "cv": label,
    }
    if verbose:
        print(f"  --- calibration: {label} ---")
        print(f"    RMSE        : {rmse:.4g}")
        print(f"    z mean/std  : {report['z_mean']:.3f} / {report['z_std']:.3f}  (target ~0 / ~1)")
        print(f"    95% coverage: {coverage95*100:.0f}%  (target ~95%)")
        print(f"    NLPD        : {nlpd:.4g}  (lower is better)")
        if report["z_std"] > 1.5:
            print("    -> z-std > 1.5: intervals look TOO NARROW (overconfident / possible overfit).")
        elif report["z_std"] < 0.6:
            print("    -> z-std < 0.6: intervals look TOO WIDE (underconfident).")
        else:
            print("    -> calibration looks reasonable.")
    return report

def get_lengthscales(source):
    """
    Extracts lengthscales from the model objects
    
    Returns an (S, dim) array of lengthscales from a fitted BoTorch model.

    `source` may be a dict from fit_and_suggest (uses source["model"]) or a
    fitted model directly.

    The lengthscale's location in the kernel tree can differ across method
    model_type variants (single_task / single_task_tight / saas /
    saas_tight) 

    The code below was generated with the help of an LLM for searching through 
    the kernel structure of each model type to find the stored lengthscale values
    
    """
    model = source["model"] if isinstance(source, dict) else source

    def find_lengthscale(module, depth=0):
        # direct hit
        ls = getattr(module, "lengthscale", None)
        if ls is not None:
            return ls
        # recurse into common wrappers / submodules
        if depth > 6:
            return None
        for attr in ("base_kernel", "data_covar_module", "covar_module"):
            sub = getattr(module, attr, None)
            if sub is not None:
                found = find_lengthscale(sub, depth + 1)
                if found is not None:
                    return found
        # generic: walk named children (covers ScaleKernel, AdditiveKernel...)
        if hasattr(module, "named_modules"):
            for name, sub in module.named_modules():
                if sub is module:
                    continue
                ls = getattr(sub, "lengthscale", None)
                if ls is not None:
                    return ls
        return None

    covar = getattr(model, "covar_module", model)
    ls_t = find_lengthscale(covar)
    if ls_t is None:
        raise AttributeError(
            "Could not locate a `lengthscale` in the model's kernel. "
            "Inspect the structure with:  print(model.covar_module)  and tell "
            "me what you see, and I'll point get_lengthscales at the right path.")

    ls = np.asarray(ls_t.detach().cpu().numpy())
    if ls.ndim == 3:                              # (S, 1, dim)
        ls = ls.squeeze(1)
    elif ls.ndim == 2 and ls.shape[0] == 1:       # (1, dim) MAP
        pass
    elif ls.ndim == 1:                            # (dim,)
        ls = ls[None, :]
    elif ls.ndim == 2:                            # (S, dim) already
        pass
    return ls                                     # (S, dim)


def lengthscale_stats(ls, dim_labels=None, verbose=True):
    """
    ls : (S, dim) array of lengthscales.

    Returns a dict with per-dimension arrays:
    
      mean        : linear mean across samples            (dim,)
      std         : linear std across samples             (dim,)
      median      : median across samples                 (dim,)
      geo_mean    : geometric mean = exp(mean(log ls))    (dim,)
      geo_std     : multiplicative std = exp(std(log ls)) (dim,) 
      
    """
    ls = np.asarray(ls)
    if ls.ndim == 1:
        ls = ls[None, :]
    S, dim = ls.shape
    labels = dim_labels or [f"x{i+1}" for i in range(dim)]

    mean = ls.mean(axis=0)
    std = ls.std(axis=0, ddof=1) if S > 1 else np.zeros(dim)
    median = np.median(ls, axis=0)
    logls = np.log(np.maximum(ls, 1e-300))
    geo_mean = np.exp(logls.mean(axis=0))
    geo_std = np.exp(logls.std(axis=0, ddof=1)) if S > 1 else np.ones(dim)

    stats = {"labels": labels, "mean": mean, "std": std, "median": median,
             "geo_mean": geo_mean, "geo_std": geo_std, "n_samples": S}

    if verbose:
        print(f"  --- Lengthscale stats ({S} samples) ---")
        print(f"    {'dim':<6}"
              f"{'geo_mean':>12}{'x/÷ factor':>12}")
        for j in range(dim):
            print(f"    {labels[j]:<6}"
                  f"{geo_mean[j]:>12.4g}{geo_std[j]:>12.4g}")
    return stats


def _pca(X):
    """
    PCA via sklearn.decomposition.PCA.

    Returns
    -------
    scores   : (n, dim) projection of every row of X onto ALL principal
               components, ordered by variance explained (descending).
               
    components : (dim, dim) principal axes as ROWS, in X's own units. To
               project a NEW point (e.g. a candidate next_x) into this SAME
               space: (x - mean) @ components.T 
               
    explained_variance_ratio : (dim,) fraction of TOTAL variance each
                               component explains
               
    mean     : (dim,) centring offset, needed for projecting new points.

    """
    from sklearn.decomposition import PCA
    X = np.asarray(X, dtype=float)
    dim = X.shape[1]
    pca = PCA(n_components=dim)
    scores = pca.fit_transform(X)
    return scores, pca.components_, pca.explained_variance_ratio_, pca.mean_


def pca_loadings(X, dim_labels=None, n_components=None, top_k=None, verbose=True,
                 lengthscales=None):
    """
    Defines which original input dimensions make up each principal component, and
    by how much.

    A PC's score is score = (x - mean) @ component, where `component` is
    that PC's (unit-norm) direction vector over ORIGINAL dimensions.
    component[j] is d(score)/d(x_j): how much a one-unit move
    in dimension j (in already-normalised [0,1] input space) shifts
    this PC's score, all else in that direction held fixed. 
    
    Sign matters, positive means increasing that input pushes the score up along this PC.

    n_components : how many PCs to report (default: all).
    
    top_k        : if given, only list the top_k largest-|loading|
                   dimensions per PC.
                   
    lengthscales : optional (dim,) array
    
                   get_lengthscales(model).median(axis=0), or
                   report["lengthscale_median"] from compare_candidates.py.
                   
                   If given, X is divided by lengthscales BEFORE fitting
                   PCA, so this answers "which combined direction matters
                   most TO THE MODEL" rather than "which combined direction
                   did I explore the most" (the default, unweighted view).
                   Loadings in this mode are per LENGTHSCALE-NORMALISED unit, not
                   per raw [0,1] input unit 

    Returns {pc_index (0-based): [(dim_label, loading), ...]}, sorted by
    |loading| descending. 
    
    Prints a readable summary if verbose.
    
    """
    X = np.asarray(X, dtype=float)
    dim = X.shape[1]
    labels = dim_labels or [f"x{i+1}" for i in range(dim)]
    Xin = X
    mode = "raw [0,1] input units"
    if lengthscales is not None:
        Xin = X / np.maximum(np.asarray(lengthscales, dtype=float), 1e-8)
        mode = "lengthscale-normalised units (model-relevant view)"
    _, components, evr, _ = _pca(Xin)
    k = n_components or dim

    report = {}
    if verbose:
        print("PCA LOADINGS (which input dimension makes up each principal component)")
        print(f"  loading = d(PC score)/d(x_j), in {mode}")
        print("  (sign: + means increasing that input increases the PC score)")
    for pc in range(k):
        vec = components[pc]
        order = np.argsort(-np.abs(vec))
        if top_k:
            order = order[:top_k]
        entries = [(labels[j], float(vec[j])) for j in order]
        report[pc] = entries
        if verbose:
            print(f"\n  PC{pc + 1} ({evr[pc] * 100:.1f}% variance):")
            for lbl, val in entries:
                bar = "#" * max(1, int(round(abs(val) * 20)))
                sign = "+" if val >= 0 else "-"
                print(f"    {lbl:8s}: {sign}{abs(val):.3f}  {bar}")
    return report
