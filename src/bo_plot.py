"""
Visualisation for the BO methods: 

-Lengthscales
-Parallel coordinates
-Side-by-side panels (mean | std | acquisition) with the observed points, the incumbent best,
and the suggested next point overlaid.
-PCA Diagnostics


TWO CASES
---------
2D problems
    The eval grid is already a meshgrid (out["mesh"] is not None), so the
    combined mean / std / acquisition surfaces are reshaped and contoured
    straight from the returned dict. Nothing is recomputed.

Higher-D problems (3D..8D)
    A full field cannot be drawn, so we plot a 2D SLICE over two chosen
    dimensions (dims=(i, j)) with the other dimensions FIXED -- by default at
    the incumbent best point's coordinates. Building that slice means querying
    the model on a fresh meshgrid, which the method's own posterior must do, so
    `reevaluate` is passed.

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from bo_utils import get_lengthscales, lengthscale_stats, _pca, pca_loadings

# 2D case: everything comes straight from the returned dict.                  
def plot_bo_slice(out, X, y, bounds, dims=(0, 1), fixed="best",
                  reevaluate=None, n_grid=60, contour_levels=80,
                  title=None, savepath=None):
    """
    out        : dict from fit_and_suggest
    X, y       : the observed design (n, d) and values (n,)
    bounds     : (2, d) array [[lo...],[hi...]]
    dims       : the two input dims to plot (only used when d > 2)
    fixed      : "best" -> fix other dims at the best point;
                 or an array of length d giving the fixed slice location.
    reevaluate : callable(grid_points (M,d)) -> (mean (M,), std (M,)).
                 required when d > 2 (to query the posterior on the slice).
                 Ignored for d == 2. See slice_predict_from_botorch below.
    n_grid     : slice resolution per axis (d > 2 case). 
    contour_levels : number of contourf color bands

    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    d = X.shape[1]
    best_idx = int(np.argmax(y))
    best_x = X[best_idx]

    if d == 2 and out.get("mesh") is not None:
        XX, YY = out["mesh"]
        mean_g = out["mean"].reshape(XX.shape)
        std_g = out["std"].reshape(XX.shape)
        acq_g = out["acq_values"].reshape(XX.shape)
        ax_lab = ("x1", "x2")
        plot_X = X
        i, j = 0, 1
    else:
        if reevaluate is None:
            raise ValueError(
                "For d > 2 you must pass a `reevaluate(grid)->(mean,std)` "
                "callback so the slice can be queried from the posterior. "
                )
        i, j = dims
        lo, hi = np.asarray(bounds, float)[0], np.asarray(bounds, float)[1]
        gi = np.linspace(lo[i], hi[i], n_grid)
        gj = np.linspace(lo[j], hi[j], n_grid)
        XX, YY = np.meshgrid(gi, gj)

        # Fixed location for the other dimensions.
        if isinstance(fixed, str) and fixed == "best":
            base = best_x.copy()
        else:
            base = np.asarray(fixed, float)
        grid = np.tile(base, (XX.size, 1))
        grid[:, i] = XX.ravel()
        grid[:, j] = YY.ravel()

        mean_flat, std_flat = reevaluate(grid)
        acqf = out.get("acqf")
        if acqf is None:
            raise ValueError(
                "out['acqf'] is missing -- the d > 2 acquisition panel needs "
                "the actual fitted acquisition function that chose next_x. "
                "Every current fit_and_suggest() call populates this; if "
                "you're passing in an older or hand-built out-dict, re-run "
                "fit_and_suggest to get one.")
        import torch
        Xg = torch.as_tensor(np.asarray(grid, dtype=float), dtype=torch.double)
        with torch.no_grad():
            acq_flat = acqf(Xg.unsqueeze(-2)).detach().cpu().numpy().ravel()
        if out["acquisition"].lower() == "ei":
            # Undo log just for EI just for plotting
            acq_flat = np.exp(acq_flat)
        mean_g = mean_flat.reshape(XX.shape)
        std_g = std_flat.reshape(XX.shape)
        acq_g = acq_flat.reshape(XX.shape)
        ax_lab = (f"x{i+1}", f"x{j+1}")
        plot_X = X  # scatter uses the two plotted dims below

    next_x = np.asarray(out["next_x"], float)

    # ------------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)
    panels = [("Posterior mean", mean_g, "viridis"),
              ("Posterior std (marginalised)", std_g, "magma"),
              (f"Acquisition ({out['acquisition'].upper()})", acq_g, "cividis")]

    for ax, (name, grid_vals, cmap) in zip(axes, panels):
        cf = ax.contourf(XX, YY, grid_vals, levels=contour_levels, cmap=cmap)
        fig.colorbar(cf, ax=ax, shrink=0.85)
        # observed points
        ax.scatter(plot_X[:, i], plot_X[:, j], c="white", edgecolors="black",
                   s=45, zorder=3, label="observed")
        # best
        ax.scatter(best_x[i], best_x[j], marker="*", c="lime",
                   edgecolors="black", s=320, zorder=5, label="best")
        # suggested next point
        ax.scatter(next_x[i], next_x[j], marker="X", c="red",
                   edgecolors="black", s=180, zorder=5, label="next")
        ax.set_xlabel(ax_lab[0]); ax.set_ylabel(ax_lab[1]); ax.set_title(name)

    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    sup = title or (f"{out.get('method','BO')}  |"
                    f"{'2D' if d==2 else f'slice dims {dims}, others fixed'}")
    fig.suptitle(sup, fontsize=11)

    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return savepath
    return fig

def slice_predict_from_botorch(out, batch_size=128):
    """
    Uses the already-fitted model stored in out["model"] to predict 
    the mean and std on a 2D slice for the d>2 functions (only to 
    enable 2D slice plotting). 
    
    Returns the COMBINED metrics from the posterior samples

    batch_size : grid points are queried in chunks of this size to avoid
                 materialising the full (S, M, M) covariance in one shot
                 (M = number of grid points, S = number of MCMC samples) 
    """
    import torch
    from bo_utils import combine_mixture
    model = out["model"]

    def reevaluate(grid):
        grid = np.asarray(grid, float)
        M = grid.shape[0]
        tg = torch.as_tensor(grid, dtype=torch.double)
        model.eval()

        mean_chunks, std_chunks = [], []
        with torch.no_grad():
            for start in range(0, M, batch_size):
                chunk = tg[start:start + batch_size]
                post = model.posterior(chunk)
                mean_chunks.append(post.mean.squeeze(-1).cpu().numpy())
                std_chunks.append(
                    post.variance.clamp_min(1e-12).sqrt().squeeze(-1).cpu().numpy())

        mean = np.concatenate(mean_chunks, axis=-1)
        std = np.concatenate(std_chunks, axis=-1)

        if mean.ndim == 2:           # (S, M) fully Bayesian -> combine
            return combine_mixture(mean, std)
        return mean, std             # (M,) MAP
    return reevaluate

def plot_lengthscales(ls, dim_labels=None, logy=True, title=None, savepath=None):
    """
    ls : (S, dim) array of lengthscales (S samples, dim dimensions).
    Box plot per dimension when S > 1; scatter of single points when S == 1.
    Log y-axis by default (lengthscales are sampled in log space and span
    orders of magnitude).
    """
    ls = np.asarray(ls)
    if ls.ndim == 1:
        ls = ls[None, :]
    S, dim = ls.shape
    labels = dim_labels or [f"x{i+1}" for i in range(dim)]

    fig, ax = plt.subplots(figsize=(max(5, 0.7 * dim + 2), 4.2),
                           constrained_layout=True)
    if S > 1:
        ax.boxplot([ls[:, j] for j in range(dim)], labels=labels,
                   showfliers=False, medianprops=dict(color="crimson"))
        # overlay the raw samples for a sense of the distribution
        for j in range(dim):
            jitter = (np.random.default_rng(j).random(S) - 0.5) * 0.25
            ax.scatter(np.full(S, j + 1) + jitter, ls[:, j], s=10,
                       alpha=0.35, color="steelblue", zorder=3)
        ax.set_ylabel("lengthscale (per MCMC sample)")
        sub = f"{S} samples per dimension"
    else:
        ax.scatter(range(1, dim + 1), ls[0], s=70, color="steelblue",
                   zorder=3)
        ax.set_xticks(range(1, dim + 1)); ax.set_xticklabels(labels)
        ax.set_ylabel("lengthscale (MAP point estimate)")
        sub = "single MAP estimate"

    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("input dimension")
    ax.set_title(title or f"ARD lengthscales  ({sub})\n"
                 "long = input matters less (slow variation); "
                 "short = matters more")
    ax.grid(axis="y", alpha=0.3)

    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return savepath
    return fig

def plot_bayesian_optimization(fcn_outputs, fcn_outputs_up):
    # Determine the starting best value
    max_orig = np.max(fcn_outputs)
    init_obs = fcn_outputs.shape[0]
    # Combine the original best with the new observations
    pts = np.concatenate(([max_orig], fcn_outputs_up[init_obs:]))
    
    # Dynamically set nquery based on the size of the points array
    nquery = np.arange(len(pts+1))
    
    plt.figure(figsize=(8, 5))
    
    # Plot the continuous connecting line first (zorder=1 pushes it behind markers)
    plt.plot(nquery, pts, linestyle='-', color='#1f77b4', zorder=1)
    
    # Plot the 0th point as an unfilled circle (white facecolor) and add it to the legend
    plt.plot(nquery[0], pts[0], marker='o', color='red', 
             linestyle='None', markersize=7, label='Best initial observation point', zorder=3)
    
    # Plot the remaining points as standard filled circles
    if len(pts) > 1:
        plt.plot(nquery[1:], pts[1:], marker='o', color='#1f77b4', 
                 linestyle='None', markersize=7, label='Queries', zorder=3)
                 
    # Dashed horizontal reference line representing the initial baseline
    plt.axhline(y=pts[0], color='gray', linestyle='--', alpha=0.7, zorder=0)
    
    # Force the x-axis to start exactly at 0 (removes the empty space on the left)
    plt.xlim(left=0)
    
    # Labels and aesthetics
    plt.xlabel("Query round")
    plt.ylabel("Function output")
    plt.title("Bayesian Optimization Progress")
    
    plt.xticks(nquery)
    plt.grid(True, linestyle=':', alpha=0.7, zorder=0)
    
    plt.legend()
    plt.tight_layout()
    
    plt.show()

def plot_parallel_coords(X, y, next_x, bounds=None, dim_labels=None,
                         normalize_axes=True, title=None, savepath=None):
    """
    Parallel-coordinates plot across all dimensions, with the best and the 
    suggested next point highlighted.

    X        : (n, dim) observed inputs
    y        : (n,)     observed values (used to colour lines and find best)
    next_x   : (dim,)   suggested next point (from out["next_x"])
    bounds   : (2, dim) for axis scaling when normalize_axes=True; defaults to
               the unit cube if None.
    normalize_axes : if True, each axis is scaled to its [lo,hi] so axes are
                     comparable; raw values are still annotated on the ticks.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    next_x = np.asarray(next_x, float).ravel()
    n, dim = X.shape
    labels = dim_labels or [f"x{i+1}" for i in range(dim)]
    best_idx = int(np.argmax(y))

    if bounds is None:
        lo, hi = np.zeros(dim), np.ones(dim)
    else:
        bounds = np.asarray(bounds, float)
        lo, hi = bounds[0], bounds[1]
    rng = np.where((hi - lo) == 0, 1.0, hi - lo)

    def scale(row):
        return (row - lo) / rng if normalize_axes else row

    axx = np.arange(dim)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * dim + 2), 4.8),
                           constrained_layout=True)

    # colour observed lines by their y value
    ynorm = (y - y.min()) / (np.ptp(y) or 1.0)
    cmap = plt.get_cmap("viridis")
    for i in range(n):
        if i == best_idx:
            continue
        ax.plot(axx, scale(X[i]), color=cmap(ynorm[i]), alpha=0.5,
                lw=1.3, zorder=2)

    # best (green) and next (red, dashed)
    ax.plot(axx, scale(X[best_idx]), color="lime", lw=3, zorder=5,
            marker="*", markersize=14, markeredgecolor="black", label="best")
    ax.plot(axx, scale(next_x), color="red", lw=2.5, ls="--", zorder=6,
            marker="X", markersize=11, markeredgecolor="black", label="next")

    ax.set_xticks(axx); ax.set_xticklabels(labels)
    if normalize_axes:
        ax.set_ylabel("value")
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.set_ylabel("value")
    ax.set_xlabel("Input dimension")
    ax.set_title(title or "Parallel coordinates plot")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    # colourbar for the objective value
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=y.min(), vmax=y.max()))
    fig.colorbar(sm, ax=ax, shrink=0.85, label="Function value")

    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return savepath
    return fig


def _as_candidate_dict(candidates):
    """Normalise `candidates` into {label: (dim,) array}, accepting either a
    dict (e.g. compare_candidates' report["candidates"]) or a single
    (dim,) array (e.g. out["next_x"])."""
    if candidates is None:
        return {}
    if isinstance(candidates, dict):
        return {k: np.asarray(v, float).ravel() for k, v in candidates.items()}
    return {"next": np.asarray(candidates, float).ravel()}


def plot_pca_scatter(X, y, candidates=None, dims=(0, 1), title=None,
                     savepath=None, show_loadings=False, dim_labels=None,
                     highlight_top=None, lengthscales=None):
    """
    2D PCA scatter of design points, coloured by observed y, with the
    incumbent best marked (star) and any candidate next_x points
    overlaid (one marker per label).

    X, y       : as everywhere else in this codebase -- X already normalised
                 to [0,1]^dim.
    candidates : optional -- a dict {label: (dim,) array} (pass
                 report["candidates"] straight from compare_candidates.py to
                 show all three acquisitions' picks at once) or a single
                 (dim,) array (e.g. out["next_x"] from check_fit). Each is
                 projected through the SAME PCA fit as the observations
                 (via (x - mean) @ components.T) -- not a separate fit --
                 so its position is directly comparable to where the actual
                 data sits
    dims       : which two principal components to plot, 0-indexed. (0, 1)
                 for PC1 vs PC2 (the usual choice); any pair is valid, e.g.
                 (0, 2) for PC1 vs PC3.
    show_loadings : if True, overlay biplot-style arrows showing each
                 original dimension's contribution to these two PCs
                 (direction = which way that input pushes the score;
                 length = how strongly, relative to the others shown).
                 Arrows are scaled for visibility, not literal coordinates
                 
    dim_labels : labels for the arrows when show_loadings=True (default
                 x1..xN).
    highlight_top : optional int or float. Ring-highlights a top subset of
                 points BY Y VALUE on this SAME full-data PCA fit -- e.g.
                 highlight_top=10 rings the top 10 points, highlight_top=0.2
                 rings the top 20%. This is the safe way to see where your
                 best points sit without refitting PCA on just that subset
                 (which, with few points relative to dim, gives an unstable,
                 often near-degenerate fit
    lengthscales : optional (dim,) array. If given, X (and any candidates)
                 are divided by lengthscales BEFORE fitting/projecting PCA
                 -- the "model-relevant" view instead of the default "raw
                 exploration" view. The title is annotated to make
                 clear which view you're looking at.

    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    dim = X.shape[1]
    w = None
    Xin = X
    if lengthscales is not None:
        w = np.maximum(np.asarray(lengthscales, dtype=float), 1e-8)
        Xin = X / w
    scores, components, evr, mean = _pca(Xin)
    cand_dict = _as_candidate_dict(candidates)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    sc = ax.scatter(scores[:, dims[0]], scores[:, dims[1]], c=y, cmap="viridis",
                    s=70, edgecolor="black", linewidth=0.6, zorder=3)

    if highlight_top is not None:
        n_top = (int(round(highlight_top * len(y))) if isinstance(highlight_top, float)
                else int(highlight_top))
        n_top = max(1, min(n_top, len(y)))
        top_idx = np.argsort(-y)[:n_top]
        ax.scatter(scores[top_idx, dims[0]], scores[top_idx, dims[1]],
                  s=170, facecolors="none", edgecolor="orangered",
                  linewidth=1.6, zorder=4, label=f"top {n_top} by y")

    best_idx = int(np.argmax(y))
    ax.scatter(scores[best_idx, dims[0]], scores[best_idx, dims[1]],
              marker="*", s=380, color="lime", edgecolor="black",
              linewidth=1.0, zorder=5, label="best observed")

    marker_cycle = ["X", "P", "D", "^", "v", "s"]
    for i, (label, x) in enumerate(cand_dict.items()):
        x_in = (x / w) if w is not None else x
        xc = (x_in - mean) @ components.T
        ax.scatter(xc[dims[0]], xc[dims[1]], marker=marker_cycle[i % len(marker_cycle)],
                  s=220, color="red", edgecolor="black", linewidth=1.0,
                  zorder=6, label=f"{label} next_x")

    if show_loadings:
        labels = dim_labels or [f"x{i+1}" for i in range(dim)]
        vx, vy = components[dims[0]], components[dims[1]]
        # scale arrows to a fixed fraction of the current axes range, purely
        # for visibility -- these are directions, not literal coordinates.
        span = max(np.ptp(scores[:, dims[0]]), np.ptp(scores[:, dims[1]])) or 1.0
        arrow_scale = 0.4 * span / max(np.max(np.abs(np.stack([vx, vy]))), 1e-8)
        for j in range(dim):
            ax.annotate("", xy=(vx[j] * arrow_scale, vy[j] * arrow_scale),
                       xytext=(0, 0),
                       arrowprops=dict(arrowstyle="->", color="dimgray", lw=1.4),
                       zorder=7)
            ax.annotate(labels[j], xy=(vx[j] * arrow_scale * 1.12, vy[j] * arrow_scale * 1.12),
                       color="dimgray", fontsize=9, ha="center", va="center", zorder=7)

    ax.set_xlabel(f"PC{dims[0]+1} ({evr[dims[0]]*100:.1f}% var)")
    ax.set_ylabel(f"PC{dims[1]+1} ({evr[dims[1]]*100:.1f}% var)")
    view_note = "lengthscale-weighted (model-relevant) view" if w is not None else "raw exploration view"
    ax.set_title(title or "Design space in PCA projection "
                f"({view_note})\n"
                "colour = observed y; star = best; marker(s) = suggested next point(s)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.colorbar(sc, ax=ax, shrink=0.85, label="objective value (y)")

    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return savepath
    return fig


def plot_pca_scree(X, title=None, savepath=None, lengthscales=None):
    """
    Scree plot: % variance explained by each principal component (bars)
    plus cumulative variance (line, right axis, with a 95% reference line)
    -- how lossy a 2D PCA projection of design points actually is. 

    lengthscales : optional (dim,) array. If given, this reports the spectrum
        in the lengthscale-weighted (model-relevant) view instead of the
        default raw-exploration view
    """
    X = np.asarray(X, dtype=float)
    if lengthscales is not None:
        X = X / np.maximum(np.asarray(lengthscales, dtype=float), 1e-8)
    _, _, evr, _ = _pca(X)
    dims = np.arange(1, len(evr) + 1)
    cum = np.cumsum(evr) * 100

    fig, ax1 = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax1.bar(dims, evr * 100, color="steelblue", zorder=3)
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("% variance explained (individual)", color="steelblue")
    ax1.set_xticks(dims)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    ax2 = ax1.twinx()
    ax2.plot(dims, cum, "o-", color="firebrick", zorder=4)
    ax2.set_ylabel("cumulative % variance explained", color="firebrick")
    ax2.tick_params(axis="y", labelcolor="firebrick")
    ax2.set_ylim(0, 105)
    ax2.axhline(95, color="firebrick", ls=":", lw=1, alpha=0.6)
    for d, c in zip(dims, cum):
        ax2.annotate(f"{c:.0f}%", (d, c), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8, color="firebrick")

    ax1.set_title(title or "PCA scree plot -- variance explained per component")

    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return savepath
    return fig
