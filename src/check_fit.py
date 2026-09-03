import warnings; warnings.filterwarnings("ignore")
import numpy as np
from bo_model import (fit_and_suggest, _build_model,
    _posterior_samples, fit_fully_bayesian_model_nuts, UCB_BETA_DEFAULT)
from bo_utils import combine_mixture


def _predict_at(model, x_row):
    """Posterior mean/std at a single point x_row (1D array), marginalised."""
    xq = np.asarray(x_row, float).reshape(1, -1)
    mm, ss = _posterior_samples(model, xq)
    mu, sd = combine_mixture(mm, ss)
    return float(np.ravel(mu)[0]), float(np.ravel(sd)[0])


def check_fit(X, y, bounds, model_type="single_task", seed=0,
                  warmup=128, num_samples=128, thinning=8,
                  cv="kfold", n_splits=10, acquisition="ucb",
                  upper_eps=1e-15,
                  ls_loc=None, ls_scale=None, noise_floor=None,
                  noise_prior_scale=0.5, ls_alpha=None,
                  loo_observation_noise=False, compute_grid=True,
                  min_dist=0.0, exclude_X=None, exclusion_weight=1e4,
                  ucb_beta=UCB_BETA_DEFAULT, posterior_batch_size=256,
                  search_bounds=None, mc_samples=128):
    """
    mc_samples : forwarded to fit_and_suggest unchanged (default = 128). Controls
                 the number of Monte Carlo samples ei/ucb/pi draw internally to estimate
                 their acquisition value. 
                 Lower this (e.g. 32-64) if memory crash occurs especially for 
                 acquisition="ei" where computation is heavier per MC sample 

      ls_loc, ls_scale       -- single_task_tight lengthscale prior (log-space
                                 location/spread; see fit_and_suggest docstring)
      noise_floor             -- single_task_tight / saas_tight noise level
      noise_prior_scale       -- single_task_tight noise prior spread
      ls_alpha                -- saas_tight global-shrinkage scale
      loo_observation_noise   -- see fit_and_suggest docstring; default=False
                                 only set True if evaluations are genuinely stochastic

    compute_grid : forwarded to fit_and_suggest (default True). 
                   pass compute_grid=False here for a faster call for only 
                   the printed diagnostics + next_x.

    min_dist, exclude_X, exclusion_weight : forwarded to fit_and_suggest
        unchanged (default min_dist=0.0, i.e. off). Set min_dist > 0 to keep
        next_x from landing within min_dist (normalised [0,1]^dim Euclidean
        distance) of any row in X, or of exclude_X if you pass extra points
        to avoid too. See fit_and_suggest's docstring for the full picture.

    ucb_beta : forwarded to fit_and_suggest unchanged (default
               UCB_BETA_DEFAULT, imported from bo_model. Only affects
               acquisition="ucb"

    posterior_batch_size : forwarded to fit_and_suggest's internal _posterior_samples
                           calls AND to check_fit's local one below (default = 256)
                            Relevant if num_samples was raised and thinning lowered for a
                            higher-quality fit: the retained-sample count (num_samples //
                            thinning) directly multiplies the memory each chunk costs. See
                            fit_and_suggest's docstring for the full explanation.

    search_bounds : if given, next_x is searched for ONLY within this box.
                    `bounds` still controls what compute_grid's diagnostic surface
                    covers, and the model is still fitted on all of X/y regardless. 
                    Default = None means "search all of bounds"
                    
    """
    X = np.asarray(X, float); y = np.asarray(y, float).reshape(-1); dim = X.shape[1]  # 1-D y

    # Fit + out-of-sample calibration + next point
    out = fit_and_suggest(X, y, bounds, model_type=model_type, seed=seed,
                          warmup=warmup, num_samples=num_samples,
                          thinning=thinning, run_loo=True, cv=cv,
                          n_splits=n_splits, acquisition=acquisition,
                          upper_eps=upper_eps,
                          ls_loc=ls_loc, ls_scale=ls_scale,
                          noise_floor=noise_floor,
                          noise_prior_scale=noise_prior_scale,
                          ls_alpha=ls_alpha,
                          loo_observation_noise=loo_observation_noise,
                          compute_grid=compute_grid,
                          min_dist=min_dist, exclude_X=exclude_X,
                          exclusion_weight=exclusion_weight,
                          ucb_beta=ucb_beta,
                          posterior_batch_size=posterior_batch_size,
                          search_bounds=search_bounds, mc_samples=mc_samples)
    rmse_out = out["loo"]["loo_rmse"]
    zstd = out["loo"]["z_std"]; cov = out["loo"]["coverage_95"]
    next_x = np.asarray(out["next_x"], float).ravel()

    m = _build_model(X, y, dim, model_type=model_type,
                     ls_loc=ls_loc, ls_scale=ls_scale,
                     noise_floor=noise_floor,
                     noise_prior_scale=noise_prior_scale,
                     ls_alpha=ls_alpha)
    fit_fully_bayesian_model_nuts(m, warmup_steps=warmup,
                                  num_samples=num_samples, thinning=thinning,
                                  disable_progbar=True, seed=seed)
    mm, ss = _posterior_samples(m, X, batch_size=posterior_batch_size)
    mu_in, _ = combine_mixture(mm, ss)
    rmse_in = float(np.sqrt(np.mean((y - np.ravel(mu_in)) ** 2)))

    pred_mu, pred_sd = _predict_at(m, next_x)

    span = float(np.ptp(y))
    ystd = float(np.std(y, ddof=1))
    skill = 1.0 - rmse_out / max(ystd, 1e-12)
    n = len(y)
    # Calculate the exact Global Naive Baseline NLPD mathematically
    baseline_nlpd = 0.5 * np.log(2 * np.pi * ystd**2) + (n - 1) / (2 * n)
    print(f"  n={len(y)}, y-span={span:.3g}, y-std={ystd:.3g}, acq={acquisition}")
    print(f"  out-of-sample RMSE : {rmse_out:.4f}  ({100*rmse_out/span:.0f}% of span)  [{out['loo']['cv']}]")
    print(f"  skill vs mean      : {skill:.2f}  (1=perfect, 0=no better than predicting the mean)")
    print(f"    NLPDbase/NLPDmodel  : {baseline_nlpd}/ {out['loo']['loo_nlpd']} ")
    print(f"\n  CALIBRATION:")
    print(f"    z-std       : {zstd:.3f}   (>1.3 overconfident/overfit; <0.7 underconfident; ~1 good)")
    print(f"    coverage95  : {cov*100:.0f}%      (~95% good)")
    
    # if zstd > 1.3:
    #     print("    -> z-std > 1.3: OVERCONFIDENT on held-out data (overfitting signal).")
    # elif zstd < 0.7:
    #     print("    -> z-std < 0.7: underconfident (intervals too wide).")
    # else:
    #     print("    -> well-calibrated: NOT overfitting. Held-out uncertainty is honest.")
    print(f"\n  next_x          : {np.round(next_x, 4)}")
    print(f"  predicted y@next: {pred_mu:.4f} +/- {pred_sd:.4f}")
    print(f"  best observed y : {out['best_observed']:.4f}")

    return {"rmse_in": rmse_in, "rmse_out": rmse_out, "skill": skill,
            "zstd": zstd, "coverage": cov, "next_x": next_x,
            "pred_next_mean": pred_mu, "pred_next_std": pred_sd, "out": out,
           "NLPDmodel":out['loo']['loo_nlpd']}
