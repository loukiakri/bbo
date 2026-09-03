"""
Systematically diagnoses disagreement between EI / UCB / PI / Thompson-sampling
next-point candidates

  1. Cross-evaluates each candidate under EVERY acquisition function (not just
     the one that proposed it), expressing each score as a min-max
     normalised "relative value" against that acquisition function's own
     range over the shared evaluation grid Xq. A candidate scoring close to
     1.0 under a DIFFERENT acquisition function's own scale means that
     function considers it nearly as good as its own optimum

  2. Lengthscale normalised distance between candidates, so a raw coordinate
     gap is judged against what the model itself considers "meaningful" in
     each dimension (short lengthscale = sensitive dimension)

  3. Posterior mean/std (marginalised) at each candidate, is summarised such 
     that any fork can be determined.  "different mean, same confidence" (the
     models disagree on where's best) or "similar mean, different
     confidence" (they disagree on how much to trust an unexplored area).

IMPORTANT: pass the SAME seed / model_type / warmup / num_samples / thinning
to all four fit_and_suggest calls
"""
import warnings
import numpy as np
import torch

from bo_utils import combine_mixture, get_lengthscales, lengthscale_stats
from bo_model import _posterior_samples
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound, qProbabilityOfImprovement
from botorch.acquisition.objective import IdentityMCObjective
from botorch.sampling import SobolQMCNormalSampler

CANDS = ("ei", "ucb", "pi", "thompson")

def _eval_acqf_on_points(acqf, X):
    """Evaluate a BoTorch AcquisitionFunction at an (N, dim) array of points,
    one point at a time. (N,) numpy array."""
    Xt = torch.as_tensor(np.asarray(X, dtype=float), dtype=torch.double)
    with torch.no_grad():
        vals = acqf(Xt.unsqueeze(-2)).detach().cpu().numpy().ravel()
    return vals


def _build_acqf(name, model, best, ucb_beta, mc_samples, seed=0):
    """
    Rebuild an ei/ucb/pi acqf against the REFERENCE model for cross-
    evaluation. 
    
    mc_samples must match what actually produced the candidate
    being scored  
    
    ei/ucb/pi are all Monte Carlo acquisition functions
    """
    name = name.lower()
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([int(mc_samples)]), seed=seed)
    if name == "ei":
        return qLogExpectedImprovement(
            model=model, best_f=torch.tensor(best, dtype=torch.double),
            sampler=sampler, objective=IdentityMCObjective())
    if name == "ucb":
        return qUpperConfidenceBound(
            model=model, beta=float(ucb_beta), sampler=sampler,
            objective=IdentityMCObjective())
    if name == "pi":
        return qProbabilityOfImprovement(
            model=model, best_f=torch.tensor(best, dtype=torch.double),
            sampler=sampler, objective=IdentityMCObjective())
    raise ValueError(f"_build_acqf only builds 'ei'/'ucb'/'pi'; got '{name}'")


def _lengthscale_report(model, dim):
    """
    Full ARD lengthscale diagnostics across retained NUTS samples, reusing
    bo_utils' existing lengthscale_stats(). 
    
    Returns (median_per_dim, full_stats_dict, degenerate_dims) where degenerate_dims
    flags any dimension whose geo_std "x/-- factor" is very large (>~50)
    
    """
    ls = get_lengthscales(model)          # (S, dim)
    stats = lengthscale_stats(ls, verbose=False)
    degenerate = [j for j in range(dim) if stats["geo_std"][j] > 50]
    return stats["median"], stats, degenerate


def compare_candidates(out_ei, out_ucb, out_pi, out_ts, X, y, bounds,
                       reference="ei", verbose=True, posterior_batch_size=256):
    """
    reference : which out-dict's fitted model is used as the single posterior
        for cross-evaluation (default "ei")

    posterior_batch_size : forwarded to every _posterior_samples call this
        function makes internally (default 256 - used to avoid memory issues)
        Lower this (e.g. 64) if candidates are compared from a fit that
        used a large num_samples.
    """
    outs = {"ei": out_ei, "ucb": out_ucb, "pi": out_pi, "thompson": out_ts}
    X = np.asarray(X, float); y = np.asarray(y, float).reshape(-1)
    dim = X.shape[1]
    best = float(y.max())

    # --- 0. sanity check: did the four fits converge to essentially the
    #     same posterior? Compare the combined mean at OUT-OF-SAMPLE points
    #     (the shared eval grid Xq) across the four models
    Xq_ref = outs[reference]["eval_points"]
    means_at_Xq = {}
    stds_at_Xq = {}
    for name, out in outs.items():
        mm, ss = _posterior_samples(out["model"], Xq_ref, batch_size=posterior_batch_size)
        mu, sd = combine_mixture(mm, ss)
        means_at_Xq[name] = mu
        stds_at_Xq[name] = sd
    ref_mu, ref_sd = means_at_Xq[reference], stds_at_Xq[reference]
    for name in means_at_Xq:
        if name == reference:
            continue
        mu, sd = means_at_Xq[name], stds_at_Xq[name]
        # z-score-style gap: normalise the mean disagreement at each grid
        # point by the COMBINED std of the two models at that point, rather
        # than a fixed fraction of the y-span.
        pooled_sd = np.sqrt(ref_sd ** 2 + sd ** 2)
        pooled_sd = np.maximum(pooled_sd, 1e-8)
        z_gap = np.abs(mu - ref_mu) / pooled_sd
        p95_z_gap = float(np.percentile(z_gap, 95))
        if p95_z_gap > 2.0:
            warnings.warn(
                f"fitted model for acquisition='{name}' disagrees with "
                f"'{reference}' by up to {p95_z_gap:.2f} pooled-std at the "
                f"95th percentile of the shared out-of-sample grid",
                stacklevel=2)

    model = outs[reference]["model"]
    Xq = outs[reference]["eval_points"]

    # --- 1. build fresh, callable acqf objects on the reference model. Thompson
    #     reuses its own cached path (drawn once, during its own fit) rather
    #     than being rebuilt here as a freshly-built PathwiseThompsonSampling
    #     would be a different random draw
    ucb_beta = outs[reference].get("ucb_beta", 0.1)
    mc_samples = outs[reference].get("mc_samples", 128)
    acqfs = {
        "ei": _build_acqf("ei", model, best, ucb_beta, mc_samples),
        "ucb": _build_acqf("ucb", model, best, ucb_beta, mc_samples),
        "pi": _build_acqf("pi", model, best, ucb_beta, mc_samples),
    }
    ts_acqf = outs["thompson"].get("thompson_acqf")
    if ts_acqf is None:
        raise ValueError(
            "out_ts['thompson_acqf'] is None -- out_ts must come from "
            "fit_and_suggest(acquisition='thompson', acq_mode='integrated').")
    acqfs["thompson"] = ts_acqf

    candidates = {name: np.asarray(outs[name]["next_x"], float).ravel()
                 for name in CANDS}

    # --- 2. grid range per acquisition function, for min-max normalisation.
    grid_vals = {
        "ei": _eval_acqf_on_points(acqfs["ei"], Xq),
        "ucb": _eval_acqf_on_points(acqfs["ucb"], Xq),
        "pi": _eval_acqf_on_points(acqfs["pi"], Xq),
        "thompson": np.asarray(outs["thompson"]["acq_values"], float),
    }

    # --- 3. cross-evaluation table: rows = "scored by", cols = "candidate from"
    # Compute raw scores first.
    raw = {r: {} for r in CANDS}
    for scorer in CANDS:
        for cand_name in CANDS:
            x = candidates[cand_name]
            v = float(_eval_acqf_on_points(acqfs[scorer], x.reshape(1, -1))[0])
            raw[scorer][cand_name] = v

    # Fold each scorer's own diagonal value into its own "hi" so the diagonal
    # is guaranteed to equal exactly 1.0 by construction.
    grid_lo = {k: float(np.min(v)) for k, v in grid_vals.items()}
    grid_hi = {k: max(float(np.max(v)), raw[k][k]) for k, v in grid_vals.items()}

    def rel(scorer_name, value):
        lo, hi = grid_lo[scorer_name], grid_hi[scorer_name]
        span = hi - lo
        if span <= 1e-12:
            return float("nan")
        return (value - lo) / span

    relative = {r: {cand: rel(r, raw[r][cand]) for cand in CANDS} for r in CANDS}

    # --- 4. ARD-lengthscale-normalised pairwise distances + full stats.
    ls_med, ls_stats, degenerate_dims = _lengthscale_report(model, dim)
    if degenerate_dims:
        warnings.warn(
            f"dimension(s) {degenerate_dims} have a very large lengthscale "
            f"geo_std factor (>50x) across retained NUTS samples -- the "
            f"model's belief about these dimensions swings wildly from one "
            f"posterior sample to the next. A sign of an under-mixed NUTS chain -- try "
            f"increasing warmup/num_samples, or check the raw per-sample "
            f"lengthscales directly with bo_plot.plot_lengthscales(ls).",
            stacklevel=2)
    dist = {}
    for a in CANDS:
        for b in CANDS:
            if a >= b:
                continue
            diff = (candidates[a] - candidates[b]) / np.maximum(ls_med, 1e-8)
            dist[(a, b)] = float(np.linalg.norm(diff))

    # --- 5. posterior mean/std at each candidate (marginalised).
    pred = {}
    for name, x in candidates.items():
        mm, ss = _posterior_samples(model, x.reshape(1, -1))
        mu, sd = combine_mixture(mm, ss)
        pred[name] = (float(mu[0]), float(sd[0]))

    # --- 6. distance from each candidate to (a) the current best observed
    #     point, and (b) the nearest existing observation overall (not
    #     necessarily the best one)
    #     Euclidean distance and lengthscale-normalised (reusing ls_med from
    #     step 4, same convention as the candidate-vs-candidate distances
    #     above).
    x_best = X[np.argmax(y)]
    dist_to_best = {}
    dist_to_nearest_obs = {}
    for name, x in candidates.items():
        d_best_raw = float(np.linalg.norm(x - x_best))
        d_best_ls = float(np.linalg.norm((x - x_best) / np.maximum(ls_med, 1e-8)))
        obs_diff = X - x.reshape(1, -1)
        obs_dist_raw = np.linalg.norm(obs_diff, axis=1)
        obs_dist_ls = np.linalg.norm(obs_diff / np.maximum(ls_med, 1e-8), axis=1)
        nearest_idx = int(np.argmin(obs_dist_raw))
        dist_to_best[name] = {"raw": d_best_raw, "lengthscale_normalised": d_best_ls}
        dist_to_nearest_obs[name] = {
            "raw": float(obs_dist_raw[nearest_idx]),
            "lengthscale_normalised": float(obs_dist_ls.min()),
            "nearest_idx": nearest_idx,
        }

    report = {
        "candidates": candidates,
        "raw_scores": raw, "relative_scores": relative,
        "lengthscale_normalised_distance": dist,
        "predicted_at_candidate": pred,
        "lengthscale_median": ls_med,
        "lengthscale_stats": ls_stats,
        "lengthscale_degenerate_dims": degenerate_dims,
        "reference_model": reference,
        "dist_to_best": dist_to_best,
        "dist_to_nearest_obs": dist_to_nearest_obs,
    }

    if verbose:
        _print_report(report, CANDS)

    return report


def _print_report(report, cands):
    print("=" * 70)
    print("CANDIDATE POINTS")
    for name in cands:
        print(f"  {name:10s}: {np.round(report['candidates'][name], 4)}")

    print("\nCROSS-EVALUATION (~0-1 scale per scorer;")
    print(" 1.0 = as good as that acquisition function's own optimum on the grid)")
    header = "  scored by \\ candidate  " + "".join(f"{c:>12s}" for c in cands)
    print(header)
    for scorer in cands:
        row = f"  {scorer:22s}"
        for cand_name in cands:
            row += f"{report['relative_scores'][scorer][cand_name]:>12.2f}"
        print(row)

    print("\n  Read by ROW")

    print("\nLENGTHSCALE-NORMALISED DISTANCE BETWEEN CANDIDATES")
    print("  (in units of the model's own median ARD lengthscale per dim --")
    print("   >~1 means the gap exceeds a lengthscale e.g")
    print("   the model would treat these as meaningfully separated)")
    for (a, b), d in report["lengthscale_normalised_distance"].items():
        print(f"  {a:10s} <-> {b:<10s}: {d:.3f}")

    stats = report["lengthscale_stats"]
    print("\nARD LENGTHSCALE STATS (per dimension, across retained NUTS samples)")
    print(f"  {'dim':<6}{'geo_mean':>12}{'x/ factor':>12}")
    for j, lbl in enumerate(stats["labels"]):
        flag = "  <-- degenerate (>50x)" if j in report["lengthscale_degenerate_dims"] else ""
        print(f"  {lbl:<6}{stats['geo_mean'][j]:>12.4g}{stats['geo_std'][j]:>12.4g}{flag}")
    #if report["lengthscale_degenerate_dims"]:

    print("\nPOSTERIOR PREDICTION AT EACH CANDIDATE (marginalised mean +/- std)")
    for name in cands:
        mu, sd = report["predicted_at_candidate"][name]
        print(f"  {name:10s}: {mu:.4f} +/- {sd:.4f}")

    print("\nDISTANCE TO CURRENT BEST OBSERVED POINT")
    print("  (raw = normalised [0,1]^dim Euclidean; ls = in units of the")
    print("   model's own median ARD lengthscale per dim)")
    for name in cands:
        d = report["dist_to_best"][name]
        print(f"  {name:10s}: raw={d['raw']:.4f}   ls={d['lengthscale_normalised']:.3f}")

    print("\nDISTANCE TO NEAREST EXISTING OBSERVATION (any point, not just best)")
    for name in cands:
        d = report["dist_to_nearest_obs"][name]
        print(f"  {name:10s}: raw={d['raw']:.4f}   ls={d['lengthscale_normalised']:.3f}"
             f"   (nearest: observation #{d['nearest_idx']})")
    print("=" * 70)
