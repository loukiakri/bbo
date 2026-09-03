"""
Method: fully Bayesian (NUTS) BoTorch GP + differentiable-acquisition
next-point selection for Bayesian optimisation.

Includes:
fit_and_suggest (Governs the full fit, check calibration, optimise acquisition, suggest next point procedure)
_build_model (fits the GP) ->
NUTS convergence diagnostics (checked automatically after every fit)
_posterior_samples (queries a fitted model) -> 
_ExclusionPenalizedAcquisition (wraps acquisition search to bounds or creates exclusion zones)
_select_next_point (chooses next_x, optimises acquisition function)

The BoTorch method library is highly complex and never covered directly in the course hence
LLMs were used to produce parts of the code below, especially when constructing custom models
from base models within the library to expose hyperparameters and when creating convergence 
metrics for NUTS (Mathematically complex and very specific to the BoTorch sampling methodology)

"""

import math
import numpy as np
from bo_utils import make_eval_points, combine_mixture, loo_calibration

try:
    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from botorch.fit import fit_fully_bayesian_model_nuts
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.acquisition.monte_carlo import (qUpperConfidenceBound,
                                                 qProbabilityOfImprovement)
    from botorch.acquisition.objective import IdentityMCObjective
    from botorch.acquisition import AcquisitionFunction
    from botorch.optim import optimize_acqf
    from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling
    from botorch.sampling import SobolQMCNormalSampler
    try:
        from botorch.models.fully_bayesian import FullyBayesianSingleTaskGP
        _HAVE_NONSPARSE_FB = True
    except Exception:
        FullyBayesianSingleTaskGP = None
        _HAVE_NONSPARSE_FB = False

    try:
        from botorch.models.fully_bayesian import (MaternPyroModel,
                                                   SaasPyroModel,
                                                   MIN_INFERRED_NOISE_LEVEL)
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as numpyro_dist
        from math import log, sqrt
        _HAVE_TIGHT_PYRO_IMPORTS = True
    except Exception:
        _HAVE_TIGHT_PYRO_IMPORTS = False

    _HAVE_BOTORCH = True
except Exception as e: 
    _HAVE_BOTORCH = False
    _IMPORT_ERR = e


#Definition of UCB_beta default. Overwritten by other functions
#All other acquisitions base exploration vs exploitation on posterior uncertainty
UCB_BETA_DEFAULT = 0.1


# Function that calls everything else
def fit_and_suggest(X, y, bounds, acquisition="ei",
                    model_type="single_task",
                    warmup=128, num_samples=128, thinning=8,
                    seed=0, run_loo=True, cv="loo", n_splits=10,
                    ls_loc=None, ls_scale=None, noise_floor=None,
                    noise_prior_scale=0.5, ls_alpha=None,
                    loo_warmup=None, loo_num_samples=None, upper_eps=1e-15,
                    loo_observation_noise=False,
                    compute_grid=True, ucb_beta=UCB_BETA_DEFAULT,
                    num_restarts=10, raw_samples=256,
                    min_dist=0.0, exclude_X=None, exclusion_weight=1e4,
                    posterior_batch_size=256, search_bounds=None,
                    mc_samples=128):
    """
    search_bounds : if provided, next_x is searched only within this dimension
                    range, must be provided as (2,dim) array for upper and lower 
                    bounds.
                    Not providing this defaults to search in the full bounds space
        
    posterior_batch_size : forwarded to _posterior_samples call forcing the processing
                           of evaluation points in chunks of the defined size to bound
                           peak memory and avoid computational resource issues.
                           Lower this if an allocation error comes up.
        
    min_dist : min_dist > 0, every point in X is treated as an exclusion centre and 
               next_x is penalised for approaching within min_dist (Euclidean)
               min_dist == 0 (default), disables behaviour
        
    exclude_X : Pass an (M, dim) array to also exclude points that aren't part of X 
        
    exclusion_weight :  soft penalty on the acquisition surface (see
                        _ExclusionPenalizedAcquisition), not a hard constraint.
                        At the boundary the penalty is 0 and it grows smoothly inside
                        min_dist, controlled by exclusion_weight (default 1e4, large enough
                        relative to typical EI/UCB/TS magnitudes that the excluded
                        neighbourhood is very unlikely to still win, without breaking the
                        gradient-based optimizer the way a hard constraint or an indicator
                        function would). If next_x still lands closer than min_dist to an
                        excluded point, raise exclusion_weight or increase min_dist.

    loo_observation_noise : False (default) - compares held-out y against an assumed noise
                            free posterior. 
                            True - compares against an assumed noisy posterior.

    acquisition : choice of acquisition function to use between UCB/EI/PI and Thompson
                  sampling
        

    compute_grid : True (default) rebuilds grid for plotting and candidate scoring cross-
                   evaluation. 
                   False - Use to skip grid rebuilding if only output required is next_x and 
                   cross validation disgnostics

    ucb_beta : UCB exploration weight, forwarded to qUpperConfidenceBound.
               Defaults to UCB_BETA_DEFAULT 

    num_restarts, raw_samples : passed to optimize_acqf's multi-start continuous optimisation
                                for every acquisition function
    """
    if not _HAVE_BOTORCH:
        raise ImportError(
            "Method needs botorch/gpytorch/torch. Install with:\n"
            "    pip install botorch gpytorch torch\n"
            f"Original import error: {_IMPORT_ERR}")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    dim = X.shape[1]
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = _build_model(X, y, dim, model_type=model_type,
                         ls_loc=ls_loc, ls_scale=ls_scale,
                         noise_floor=noise_floor,
                         noise_prior_scale=noise_prior_scale,
                         ls_alpha=ls_alpha)
    fit_fully_bayesian_model_nuts(
        model, warmup_steps=warmup, num_samples=num_samples,
        thinning=thinning, disable_progbar=True, seed=seed)

    nuts_diagnostics = _check_nuts_convergence(model)

    Xq = mesh = mean = std = None
    if compute_grid:
        Xq, mesh = make_eval_points(dim, bounds=bounds, seed=seed,
                                    upper_eps=upper_eps)
        means, stds = _posterior_samples(model, Xq, batch_size=posterior_batch_size)
        mean, std = combine_mixture(means, stds)

    best = float(y.max())

    _search_bounds = search_bounds if search_bounds is not None else bounds

    exclude_points = None
    if min_dist > 0:
        if exclude_X is not None:
            extra = np.asarray(exclude_X, dtype=float).reshape(-1, dim)
            exclude_points = np.vstack([X, extra]) if extra.size else X
        else:
            exclude_points = X

    next_x_arr, acqf, grid_vals = _select_next_point(
        model, _search_bounds, dim, acquisition, best, seed,
        upper_eps=upper_eps, num_restarts=num_restarts,
        raw_samples=raw_samples, ucb_beta=ucb_beta,
        eval_grid=Xq, exclude_points=exclude_points,
        min_dist=min_dist, exclusion_weight=exclusion_weight,
        mc_samples=mc_samples)
    next_x = next_x_arr.ravel()
    thompson_acqf = acqf if acquisition.lower() == "thompson" else None

    loo = None
    if run_loo:
        _loo_warmup = loo_warmup if loo_warmup is not None else warmup
        _loo_num = loo_num_samples if loo_num_samples is not None else num_samples

        def fp(Xtr, ytr, Xpt):
            m = _build_model(Xtr, ytr, dim, model_type=model_type,
                             ls_loc=ls_loc, ls_scale=ls_scale,
                             noise_floor=noise_floor,
                             noise_prior_scale=noise_prior_scale,
                             ls_alpha=ls_alpha)
            fit_fully_bayesian_model_nuts(
                m, warmup_steps=_loo_warmup, num_samples=_loo_num,
                thinning=thinning, disable_progbar=True, seed=seed)
            # loo_observation_noise controls whether to use noise or noise-free assumption
            mm, ss = _posterior_samples(m, Xpt, observation_noise=loo_observation_noise,
                                        batch_size=posterior_batch_size)
            return combine_mixture(mm, ss)
        loo = loo_calibration(X, y, fp, cv=cv, n_splits=n_splits)

    return {
        "method": "BoTorch FullyBayesianNUTS",
        "model_type": model_type,
        "bounds": bounds, 
        "search_bounds": _search_bounds,
        "eval_points": Xq, "mesh": mesh,
        "mean": mean, "std": std,
        "acquisition": acquisition, "acq_values": grid_vals,
        "ucb_beta": ucb_beta,
        "mc_samples": mc_samples, 
        "next_x": next_x, "best_observed": best,
        "loo": loo, "model": model,
        "acqf": acqf,  # the exact fitted acquisition function that chose next_x
        "thompson_acqf": thompson_acqf,
        "min_dist": min_dist, "exclude_points": exclude_points,
        "exclusion_weight": exclusion_weight, 
        "nuts_diagnostics": nuts_diagnostics,
    }


# =============================================================================
# MODEL FITTING
# =============================================================================
def _build_model(X, y, dim, model_type="single_task",
                 ls_loc=None, ls_scale=None, noise_floor=None,
                 noise_prior_scale=0.5, ls_alpha=None):
    
    """
    Builds requested model (saas, saas_tight, single_task, single_tast_tight)
    
    X : Input dimensions
    y: Output observations
    dim : Function dimensions
    
    """
    tX = torch.as_tensor(X, dtype=torch.double)
    ty = torch.as_tensor(np.asarray(y).reshape(-1, 1), dtype=torch.double)
    unit_bounds = torch.stack([torch.zeros(dim, dtype=torch.double),
                               torch.ones(dim, dtype=torch.double)])
    common = dict(
        input_transform=Normalize(d=dim, bounds=unit_bounds),
        outcome_transform=Standardize(m=1), # standardise outputs
    )

    if model_type == "saas":
        return SaasFullyBayesianSingleTaskGP(tX, ty, **common)

    if model_type == "saas_tight": # Adding manual boundaries to noise and lengthscale
        if not _HAVE_TIGHT_PYRO:
            raise ImportError(
                "saas_tight needs jax/numpyro. Install with:\n"
                "    pip install numpyro jax jaxlib\n"
                "Or use model_type='saas' instead.")
        ConfigurableSaasPyroModel.noise_floor = (
            noise_floor if noise_floor is not None else 0.09)
        ConfigurableSaasPyroModel.ls_alpha = (
            ls_alpha if ls_alpha is not None else 0.1)
        return ConfigurableSaasFullyBayesianSingleTaskGP(tX, ty, **common)

    if model_type not in ("single_task", "single_task_tight"):
        raise ValueError(f"model_type must be 'single_task', "
                         f"'single_task_tight', 'saas', or 'saas_tight', "
                         f"got '{model_type}'.")

    if not (_HAVE_NONSPARSE_FB and _HAVE_TIGHT_PYRO):
        if not _HAVE_NONSPARSE_FB:
            raise ImportError(
                "FullyBayesianSingleTaskGP not available; upgrade botorch or "
                "use model_type='saas'.")
        if model_type == "single_task":
            return FullyBayesianSingleTaskGP(tX, ty, **common)
        raise ImportError(
            "single_task_tight needs jax/numpyro (the configurable pyro"
            "model failed to import). Install with:\n"
            "    pip install numpyro jax jaxlib\n"
            "Or use model_type='single_task' instead.")

    from botorch.models.fully_bayesian import MIN_INFERRED_NOISE_LEVEL
    if model_type == "single_task":
        ConfigurableMaternPyroModel.ls_loc = None
        ConfigurableMaternPyroModel.ls_scale = (
            ls_scale if ls_scale is not None else math.sqrt(3))
    else:
        ConfigurableMaternPyroModel.ls_loc = ls_loc
        ConfigurableMaternPyroModel.ls_scale = (
            ls_scale if ls_scale is not None else 1.0)
    ConfigurableMaternPyroModel.noise_floor = (
        noise_floor if noise_floor is not None else MIN_INFERRED_NOISE_LEVEL)
    ConfigurableMaternPyroModel.noise_prior_scale = noise_prior_scale

    pyro_model = ConfigurableMaternPyroModel()
    try:
        model = FullyBayesianSingleTaskGP(tX, ty, pyro_model=pyro_model,
                                          **common)
    except TypeError:
        model = FullyBayesianSingleTaskGP(tX, ty, **common)
        try:
            tX_t = model.transform_inputs(tX)
        except Exception:
            tX_t = tX
        ty_t = ty
        if getattr(model, "outcome_transform", None) is not None:
            try:
                ty_t, _ = model.outcome_transform(ty)
            except Exception:
                ty_t = ty
        pyro_model.set_inputs(tX_t, ty_t, None)
        model.pyro_model = pyro_model
    return model


# =============================================================================
# NUTS CONVERGENCE DIAGNOSTICS
#
# Fully Bayesian methods only deliver well-calibrated uncertainty if the NUTS
# chain actually converged and mixed. Below is a series of checks on the retained
# lengthscale samples (the hyperparameter most prone to poor mixing in a
# GP). They checks don't change fitting behaviour at all, just evaluate whether
# it can be trusted
# =============================================================================
def _lengthscale_samples(model):
    """
    Return the (S, dim) array of retained per-dimension lengthscale draws
    from a fitted model -- S is the number of NUTS posterior samples kept
    after warmup+thinning. Fully Bayesian BoTorch models store each
    hyperparameter as a BATCHED tensor with that leading S dimension (one
    value per retained MCMC draw), rather than the single point estimate a
    MAP/MLE model would have -- this is exactly what _posterior_samples/
    combine_mixture marginalise over elsewhere in this file.

    Different model_type variants (single_task / single_task_tight / saas /
    saas_tight) can wrap the lengthscale at a different point in the kernel
    tree, so rather than hard-code one path, this searches the tree for the
    first submodule exposing a `.lengthscale` attribute. 
    
    """
    def _find(module):
        ls = getattr(module, "lengthscale", None)
        if ls is not None:
            return ls
        for child in module.children():
            found = _find(child)
            if found is not None:
                return found
        return None

    ls = _find(model.covar_module)
    if ls is None:
        raise AttributeError(
            "Could not find a 'lengthscale' attribute anywhere in this "
            "model's kernel tree.")
    ls = ls.detach().cpu().numpy()
    return ls.reshape(ls.shape[0], -1)


def _split_rhat(samples):
    """
    Split-chain R-hat: the classic Gelman-Rubin convergence diagnostic,
    applied to a SINGLE NUTS chain by splitting it into two equal halves
    and treating them as if they were two independent chains -- the
    standard trick for assessing one chain, since it can catch a chain
    that hadn't settled into its stationary distribution over its own
    length even though there's only one chain available to look at.

    R-hat near 1.0: the two halves look like draws from the same
    distribution (consistent with convergence). R-hat noticeably above 1:
    the first and second half still disagree -- a sign the chain was
    probably still drifting rather than sampling the posterior, i.e. raise
    warmup/num_samples. Classic Gelman & Rubin (1992) used 1.1 as a rule of
    thumb; more recent guidance (Vehtari et al. 2021) recommends a
    stricter 1.01 for publication-quality inference. This file flags at
    1.05 as a practical middle ground given how few samples a typical BO
    iteration retains (see _check_nuts_convergence below).

    samples : (S, dim) array. Returns (dim,) array of R-hat values.
    """
    samples = np.asarray(samples, dtype=float)
    S = samples.shape[0]
    n = S // 2
    if n < 2:
        return np.full(samples.shape[1], np.nan)
    a, b = samples[:n], samples[n:2 * n]
    chain_means = np.stack([a.mean(axis=0), b.mean(axis=0)])       # (2, dim)
    chain_vars = np.stack([a.var(axis=0, ddof=1), b.var(axis=0, ddof=1)])
    W = chain_vars.mean(axis=0)                       # within-chain variance
    B = n * chain_means.var(axis=0, ddof=1)            # between-chain variance
    var_hat = ((n - 1) / n) * W + B / n
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_hat / np.maximum(W, 1e-300))
    return rhat


def _effective_sample_size(samples):
    """
    Approximate per-dimension effective sample size (ESS) via Geyer's
    initial positive sequence estimator: sum paired (lag 2m, 2m+1)
    autocorrelations until a pair sum turns non-positive, then
    ESS = S / (1 + 2 * sum of the retained pair sums). This is the same
    core idea behind Stan/ArviZ's ESS, simplified here (no rank-
    normalisation, no multi-chain pooling). 
    It gets noisier the fewer samples you have: with a typical warmup=num_samples=128,
    thinning=8 setup (16 retained samples), this is a rough signal, not a
    precise number 

    samples : (S, dim) array. Returns (dim,) array of ESS values, each
        clipped to [1, S] (ESS can't exceed the sample count you actually
        have, and the raw estimator can occasionally overshoot slightly
        with very few samples).
    """
    samples = np.asarray(samples, dtype=float)
    S, dim = samples.shape
    ess = np.empty(dim)
    for d in range(dim):
        x = samples[:, d] - samples[:, d].mean()
        var0 = np.dot(x, x) / S
        if var0 <= 1e-300:
            ess[d] = S           # a constant chain -- degenerate, not "poor mixing"
            continue
        rho = []
        for lag in range(1, max(S - 1, 1)):
            c = np.dot(x[:-lag], x[lag:]) / S
            rho.append(c / var0)
        acc, m = 0.0, 0
        while 2 * m + 1 < len(rho):
            pair = rho[2 * m] + rho[2 * m + 1]
            if pair <= 0:
                break
            acc += pair
            m += 1
        ess[d] = S / (1.0 + 2.0 * acc)
    return np.clip(ess, 1.0, float(S))


def _check_nuts_convergence(model, rhat_threshold=1.05, ess_frac_threshold=0.5):
    """
    Post-hoc NUTS convergence check on the fitted model's retained
    lengthscale samples. Called automatically after the main fit in
    fit_and_suggest (NOT after every LOO/K-fold refit).

    Always returns a dict; only prints/warns if something looks off, so a
    healthy fit stays quiet.
    """
    try:
        ls = _lengthscale_samples(model)
    except Exception as e:
        return {"error": str(e)}

    S = ls.shape[0]
    rhat = _split_rhat(ls)
    ess = _effective_sample_size(ls)
    diag = {"n_retained": S, "lengthscale_rhat": rhat, "lengthscale_ess": ess}

    max_rhat = float(np.nanmax(rhat)) if np.any(~np.isnan(rhat)) else float("nan")
    min_ess = float(np.min(ess))
    flagged = ((not np.isnan(max_rhat) and max_rhat > rhat_threshold)
              or (min_ess < ess_frac_threshold * S))
    diag["flagged"] = bool(flagged)

    if flagged:
        import warnings
        warnings.warn(
            f"NUTS convergence check: max split-R-hat={max_rhat:.3f} (want "
            f"~1.0; flagged above {rhat_threshold}), min lengthscale "
            f"ESS={min_ess:.1f} of {S} retained samples (flagged below "
            f"{ess_frac_threshold * S:.0f}) -- the lengthscale posterior may "
            f"not be well mixed. With only {S} retained samples "
            f"(num_samples // thinning), this check itself has limited "
            f"power either way -- if you're relying on this fit for an "
            f"important decision, raise num_samples/warmup and/or lower "
            f"thinning and refit before trusting it.",
            stacklevel=3)
    return diag


# =============================================================================
# QUERYING A FITTED MODEL
# =============================================================================
def _posterior_samples(model, Xq, observation_noise=False, batch_size=256):
    """
    observation_noise: False (default) returns the posterior over a noise-free
                       function value
                       True returns the distribution for a noisy observation by adding 
                       fitted noise variance on top of the latent uncertainty

    batch_size : evaluation points are processed in chunks of this size to avoid 
                 materialising the full (S, N, N) covariance matrix where S is 
                 the number of retained NUTS samples. Lower value if memory allocation
                 errors arise.
                 
    """
    Xq = np.asarray(Xq, dtype=float)
    N = Xq.shape[0]
    tXq = torch.as_tensor(Xq, dtype=torch.double)

    means_chunks, stds_chunks = [], []
    with torch.no_grad():
        for start in range(0, N, batch_size):
            chunk = tXq[start:start + batch_size]
            post = model.posterior(chunk, observation_noise=observation_noise)
            means_chunks.append(post.mean.squeeze(-1).cpu().numpy())
            stds_chunks.append(
                post.variance.clamp_min(1e-12).sqrt().squeeze(-1).cpu().numpy())

    # concatenate along the point axis (Understand which axis that is
    # per-chunk first, since mean/std can come back as (N,) or (S, N))
    def _cat(chunks):
        a0 = np.asarray(chunks[0])
        axis = 0 if a0.ndim == 1 else 1
        return np.concatenate(chunks, axis=axis)

    mean = _cat(means_chunks)
    std = _cat(stds_chunks)

    def _to_SN(a):
        a = np.asarray(a)
        if a.ndim == 1:
            return a.reshape(-1, 1)
        if a.ndim != 2:
            a = a.reshape(a.shape[0], -1)
        if a.shape[1] == N:
            return a
        if a.shape[0] == N:
            return a.T
        return a
    return _to_SN(mean), _to_SN(std)


# =============================================================================
# NEXT-POINT SELECTION
# =============================================================================
class _ExclusionPenalizedAcquisition(AcquisitionFunction):
    """
    Wraps a base acquisition function and subtracts a smooth penalty that
    grows as a candidate approaches any point in `exclude_points`, steering
    optimize_acqf's gradient-based multi-start search away from points (or
    tight neighbourhoods around points) 

    Deliberately a PENALTY added to the objective, not a hard constraint
    This penalty stays fully differentiable, and needs no change to
    how optimize_acqf is called in _select_next_point below

    exclude_points : (M, d) array
    
    min_dist : candidates within this Euclidean distance of any excluded point are penalised; 
                points at or beyond min_dist from every excluded point are unaffected.
                
    weight : how hard the penalty pushes back once inside the radius.
            Default is large relative to typical EI/UCB/TS magnitudes so the
            excluded neighbourhood is close to a hard no-go zone, while staying
            smooth (a quadratic soft barrier, not a step) so gradients remain
            informative right up to the boundary instead of vanishing.
        
    """

    def __init__(self, acq_function, exclude_points, min_dist, weight=1e4):
        super().__init__(model=acq_function.model)
        self.acq_function = acq_function
        self.register_buffer(
            "exclude_points",
            torch.as_tensor(np.asarray(exclude_points, dtype=float),
                            dtype=torch.double))
        self.min_dist = float(min_dist)
        self.weight = float(weight)

    def forward(self, X):
        base = self.acq_function(X) 
        Xc = X.mean(dim=-2)                           
        diff = Xc.unsqueeze(-2) - self.exclude_points 
        dist = diff.pow(2).sum(dim=-1).clamp_min(1e-18).sqrt()
        min_d = dist.min(dim=-1).values                   
        penalty = self.weight * torch.relu(self.min_dist - min_d).pow(2)
        return base - penalty


def _select_next_point(model, bounds, dim, acquisition, best, seed,
                       upper_eps=1e-15, num_restarts=10, raw_samples=256,
                       ucb_beta=UCB_BETA_DEFAULT, q=1, eval_grid=None,
                       exclude_points=None, min_dist=0.0,
                       exclusion_weight=1e4, mc_samples=128):
    """
        
    Chooses the next point by optimising a BoTorch acquisition function
    directly against the fitted fully-Bayesian model with botorch.optim.optimize_acqf.

    mc_samples : ei/ucb/pi are all Monte Carlo acquisition functions, they
        estimate their value by drawing `mc_samples` samples from the
        posterior and averaging, via a `sampler` argument. 

    ei, ucb, pi : differentiable MC acquisition functions
        (qLogExpectedImprovement / qUpperConfidenceBound /
        qProbabilityOfImprovement). These handle the fully Bayesian model's
        extra MCMC-sample batch dimension internally (averaging the
        acquisition value over hyperparameter posterior samples)
        
    thompson : PathwiseThompsonSampling, one Matheron-rule function sample
                per posterior-hyperparameter sample, optimised continuously. Needs no grid at
                all, optimize_acqf explores the continuous space directly.

    eval_grid : optional (M, dim) array. If given (and q == 1), the SAME
                acqf object used for selection is also evaluated at these M points
                and returned as `grid_vals`. This makes the returned surface an
                exact match to what actually produced next_x for every acquisition

    exclude_points, min_dist, exclusion_weight : if min_dist > 0 and
        exclude_points is non-empty, the acqf is wrapped in
        _ExclusionPenalizedAcquisition before optimisation (and before the
        eval_grid evaluation below, so a plotted surface reflects the
        exclusion too) min_dist == 0 (the default) skips the wrap entirely

    Returns (next_x, acqf, grid_vals). grid_vals is None if eval_grid was
    None. acqf is returned so callers (e.g. compare_candidates.py,
    or slice plotting) can reuse the exact fitted acquisition function
    If exclusion was applied, this is the WRAPPED
    (penalised) acqf
    
    """
    bnp = np.asarray(bounds, dtype=float).copy()
    if upper_eps > 0:
        lo, hi = bnp[0], bnp[1]
        bnp[1] = lo + (hi - lo) * (1.0 - float(upper_eps))
    tbounds = torch.as_tensor(bnp, dtype=torch.double)

    torch.manual_seed(seed)
    name = acquisition.lower()
    mc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([int(mc_samples)]), seed=seed)
    if name == "ei":
        acqf = qLogExpectedImprovement(
            model=model, best_f=torch.tensor(best, dtype=torch.double),
            sampler=mc_sampler, objective=IdentityMCObjective())
    elif name == "ucb":
        acqf = qUpperConfidenceBound(
            model=model, beta=float(ucb_beta), sampler=mc_sampler,
            objective=IdentityMCObjective())
    elif name == "pi":
        acqf = qProbabilityOfImprovement(
            model=model, best_f=torch.tensor(best, dtype=torch.double),
            sampler=mc_sampler, objective=IdentityMCObjective())
    elif name == "thompson":
        acqf = PathwiseThompsonSampling(model=model, objective=IdentityMCObjective())
    else:
        raise ValueError(
            f"Unknown acquisition '{acquisition}'. Choose from "
            f"'ei', 'ucb', 'pi', 'thompson'.")

    if min_dist > 0 and exclude_points is not None and len(exclude_points) > 0:
        acqf = _ExclusionPenalizedAcquisition(
            acqf, exclude_points, min_dist=min_dist, weight=exclusion_weight)

    candidate, _ = optimize_acqf(
        acq_function=acqf,
        bounds=tbounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    next_x = candidate.detach().cpu().numpy().reshape(q, dim)

    grid_vals = None
    if eval_grid is not None and q == 1:
        Xg = torch.as_tensor(np.asarray(eval_grid, dtype=float), dtype=torch.double)
        with torch.no_grad():
            grid_vals = acqf(Xg.unsqueeze(-2)).detach().cpu().numpy().ravel()
        if name == "ei":
            # qLogExpectedImprovement returns log(EI), not EI, for numerical
            # stability during optimize_acqf's gradient-based search
            # Undoing the log here after next_x was already chosen above so 
            # this only changes what gets displayed
            grid_vals = np.exp(grid_vals)

    return next_x, acqf, grid_vals


# =============================================================================
# Tight-prior Pyro models
# Only used when model_type is "single_task_tight" or "saas_tight" 

# LLM help was used to extract the noise and lengthscale controls by setting up
# custom models, as BoTorch by design hides these levers.

# Extracting these hyperparameters and exposing them for manual use by the user is
# what is being implemented in the code below.
# =============================================================================
if _HAVE_TIGHT_PYRO_IMPORTS:
    class ConfigurableMaternPyroModel(MaternPyroModel):
        ls_loc = None
        ls_scale = 0.2#1.0
        noise_floor = MIN_INFERRED_NOISE_LEVEL
        noise_prior_scale = 0.5

        def sample_lengthscale(self, dim, **tkwargs):
            loc = (self.ls_loc if self.ls_loc is not None
                   else sqrt(2) + log(dim) * 0.5)
            loc = float(loc)
            scale = float(self.ls_scale)
            return numpyro.sample(
                "lengthscale",
                numpyro_dist.LogNormal(
                    loc=jnp.full((dim,), loc),
                    scale=jnp.full((dim,), scale),
                ),
            )

        def sample_noise(self, **tkwargs):
            from math import log as _pylog
            loc = _pylog(float(self.noise_floor))
            scale = float(self.noise_prior_scale)
            return numpyro.sample(
                "noise",
                numpyro_dist.LogNormal(
                    loc=jnp.array(loc),
                    scale=jnp.array(scale),
                ),
            )

    class ConfigurableSaasPyroModel(SaasPyroModel):
        """
        SAAS prior with two configurable options

        noise_floor : approximate MEAN of the noise prior, hard-coded as
                      Gamma(0.9, 10), mean 0.09. Assuming FIXED
                    concentration (0.9, matching botorch's stock shape) so that
                    concentration / rate == noise_floor. 
                    RAISE this to widen predictive intervals if the model is genuinely 
                    overconfident (check via cross validation for effect)

        ls_alpha :  scale of the HalfCauchy prior on the GLOBAL shrinkage
                    parameter in SAAS's horseshoe-style lengthscale prior.
                    LOWER alpha => more global shrinkage => lengthscales pushed 
                    longer on average => more dimensions pulled toward "inactive" => 
                    less prone to short- lengthscale overfitting. 
                    This is SAAS's sparsity-strength knob. 
            
        """
        noise_floor = 0.09          # botorch stock mean (0.9 / 10)
        noise_conc = 0.9            # keep stock Gamma shape fixed
        ls_alpha = 0.1              # botorch stock global-shrinkage scale

        def sample_noise(self):
            if self.train_Yvar_jax is None:
                conc = float(self.noise_conc)
                rate = conc / float(self.noise_floor)
                return MIN_INFERRED_NOISE_LEVEL + numpyro.sample(
                    "noise",
                    numpyro_dist.Gamma(jnp.array(conc), jnp.array(rate)))
            return self.train_Yvar_jax

        def sample_lengthscale(self, dim, alpha=0.1):
            # ignore the caller's default alpha=0.1; use the configurable one
            return super().sample_lengthscale(dim, alpha=float(self.ls_alpha))

    class ConfigurableSaasFullyBayesianSingleTaskGP(SaasFullyBayesianSingleTaskGP):
        _pyro_model_class = ConfigurableSaasPyroModel

    _HAVE_TIGHT_PYRO = True
else:
    ConfigurableMaternPyroModel = None
    ConfigurableSaasPyroModel = None
    ConfigurableSaasFullyBayesianSingleTaskGP = None
    _HAVE_TIGHT_PYRO = False
