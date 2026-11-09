"""Figures for the four experiments, drawn in one consistent style."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import matplotlib.patheffects  # noqa: E402,F401
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

# Palette: validated categorical slots (fixed order), one-hue blue ramp, neutral inks.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURFACE = "#e6e5e1", "#fcfcfb"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
CELLS = ListedColormap([SURFACE, "#1f2a37"])
HALO = [matplotlib.patheffects.withStroke(linewidth=3, foreground=SURFACE)]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.markersize": 5,
    })


def headline(fig, title, subtitle):
    h = fig.get_figheight()
    fig.text(0.01, 0.995, title, ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.01, 0.995 - 0.34 / h, subtitle, ha="left", va="top", fontsize=10.5, color=INK2)


def save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"saved {path}")


def panel(ax, title, subtitle=None):
    """Bold panel title with an optional lighter second line."""
    ax.set_title(title, fontsize=10.5, pad=20 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=INK2)


def show_cells(ax, grid, title=None, color=INK2):
    ax.imshow(grid, cmap=CELLS, interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(GRID)
    if title:
        ax.set_title(title, fontsize=9, fontweight="normal", color=color, loc="center", pad=4)


# ------------------------------------------------------------------- soups
def plot_soups(d, path):
    style()
    fig = plt.figure(figsize=(14, 4.6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.32, width_ratios=[1, 1, 1.1])

    ax = fig.add_subplot(gs[0])
    ps = sorted(d["density_sweep"], key=float)
    x = np.array([float(p) for p in ps])
    final = [np.array(d["density_sweep"][p]["final_density"]) for p in ps]
    mean = np.array([f.mean() for f in final])
    sd = np.array([f.std() for f in final])
    ax.fill_between(x, mean - sd, mean + sd, color=BLUE, alpha=0.15, lw=0)
    ax.plot(x, mean, color=BLUE, marker="o")
    plateau = mean[(x >= 0.2) & (x <= 0.6)].mean()
    ax.axhline(plateau, color=MUTED, lw=1)
    ax.text(0.42, 0.3, f"plateau ≈ {plateau:.3f}\n(mean for starts of 20–60%)", color=INK2, va="center",
            ha="center", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("initial density (fraction of cells alive)")
    ax.set_ylabel("final density")
    ax.set_ylim(0, max(0.045, (mean + sd).max() * 1.15))
    size = d["config"]["density_size"]
    panel(ax, "A  Almost any start ends about 3% alive", f"{size}×{size} torus, mean ± SD over soups")

    ax = fig.add_subplot(gs[1])
    sizes = sorted(d["size_sweep"], key=int)
    L = np.array([int(s) for s in sizes])
    lifetimes = [np.array(d["size_sweep"][s]["transient"]) for s in sizes]
    lifetimes = [t[t >= 0] for t in lifetimes]
    med = np.array([np.median(t) for t in lifetimes])
    q1 = np.array([np.percentile(t, 25) for t in lifetimes])
    q3 = np.array([np.percentile(t, 75) for t in lifetimes])
    ax.fill_between(L, q1, q3, color=BLUE, alpha=0.15, lw=0)
    ax.plot(L, med, color=BLUE, marker="o")
    slope, icpt = np.polyfit(np.log(L), np.log(med), 1)
    ax.plot(L, np.exp(icpt) * L ** slope, color=MUTED, lw=1, ls="--")
    k = len(L) // 2  # label the middle size (64 in the full run; the quick run has only two sizes)
    ax.text(L[k] * 1.15, np.exp(icpt) * L[k] ** slope * 0.62, f"$\\propto L^{{{slope:.2f}}}$", color=INK2,
            fontsize=10, path_effects=HALO)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(L)
    ax.set_xticklabels([f"{v}" for v in L])
    ax.set_xlabel("grid side L (cells)")
    ax.set_ylabel("generations until it settles")
    panel(ax, "B  Bigger worlds live longer", "median and interquartile range, 35% initial density")

    ax = fig.add_subplot(gs[2])
    census = d["census"]
    total = sum(census.values())
    named = [(n, c) for n, c in census.items() if "(" not in n and n != "unstable"][:10]
    other = total - sum(c for _, c in named)
    items = named + [("everything else", other)]
    names = [n.replace("_", " ") for n, _ in items][::-1]
    share = np.array([100 * c / total for _, c in items])[::-1]
    ax.barh(names, share, color=BLUE, height=0.7)
    for i, v in enumerate(share):
        ax.text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("share of all objects (%)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, share.max() * 1.18)
    big = max(d["config"]["sizes"], key=int)
    n_soups = d["config"]["sizes"][big]
    panel(ax, "C  What the ash is made of", f"{total:,} objects from {n_soups} settled {big}×{big} soups")

    headline(fig, "How does a random world live and die?",
             "Random soups of Conway's Life churn for a while, then freeze into sparse \"ash\" of "
             "still lifes and blinkers.")
    fig.subplots_adjust(top=0.78)
    save(fig, path)


# --------------------------------------------------------------- rule space
STEPS_MUTANTS = 4000
CLASS_ORDER = ["dies out", "freezes", "oscillates", "never settles"]
CLASS_COLORS = {"dies out": YELLOW, "freezes": BLUE, "oscillates": AQUA, "never settles": ORANGE}


def plot_rule_space(d, summary, worlds, finals, path):
    from rule_space import LONG_TRANSIENT, classify

    style()
    cls = classify(d["density"], d["period"])
    lam = d["lam"]
    life = summary.get("life", {})

    fig = plt.figure(figsize=(14, 5.2))
    gs = GridSpec(1, 2, figure=fig, wspace=0.22, width_ratios=[1, 1.15])

    ax = fig.add_subplot(gs[0])
    bins = np.linspace(0, 1, 21)
    idx = np.clip(np.digitize(lam, bins) - 1, 0, len(bins) - 2)
    centers = (bins[:-1] + bins[1:]) / 2
    counts = np.array([[np.sum((idx == b) & (cls == c)) for c in CLASS_ORDER] for b in range(len(bins) - 1)])
    keep = counts.sum(1) > 0
    frac = counts[keep] / counts[keep].sum(1, keepdims=True)
    bottom = np.zeros(keep.sum())
    width = (bins[1] - bins[0]) * 0.9
    for k, c in enumerate(CLASS_ORDER):
        ax.bar(centers[keep], frac[:, k], width=width, bottom=bottom, color=CLASS_COLORS[c],
               edgecolor=SURFACE, linewidth=1.2, label=c)
        bottom += frac[:, k]
    ax.axvline(life.get("lam", 0.2734), color=INK, lw=1.2)
    ax.text(life.get("lam", 0.2734) + 0.01, 1.02, "Conway's Life (λ = 0.273)", color=INK, fontsize=9,
            ha="left", va="bottom")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Langton's λ (share of neighbourhoods that produce a live cell)")
    ax.set_ylabel("share of rules")
    ax.grid(axis="x", visible=False)
    n = summary["n_rules"]
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, handlelength=1.2)
    ax.set_title(f"A  How all {n:,} rules end, by λ", fontsize=10.5)

    ax = fig.add_subplot(gs[1])
    settled = (d["period"] > 0) & (d["transient"] > 0)
    hb = ax.hexbin(lam[settled], d["transient"][settled], gridsize=(40, 22), yscale="log",
                   bins="log", cmap=BLUE_RAMP, mincnt=1, linewidths=0.2, edgecolors=SURFACE)
    cb = fig.colorbar(hb, ax=ax, pad=0.01, fraction=0.04)
    cb.set_label("rules per cell (log)", color=INK2)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    cb.ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axhline(LONG_TRANSIENT, color=MUTED, lw=1)
    ax.text(0.99, LONG_TRANSIENT * 1.1, f"long-lived: ≥ {LONG_TRANSIENT} generations",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8.5, color=INK2)
    if life:
        ax.scatter([life["lam"]], [life["transient"]], marker="*", s=260, color=ORANGE,
                   edgecolor=SURFACE, linewidth=1.5, zorder=5)
        ax.annotate("Conway's Life", (life["lam"], life["transient"]), xytext=(18, 10),
                    textcoords="offset points", fontsize=9.5, fontweight="bold", color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8), path_effects=HALO)
    for label in ("highlife", "drylife", "pseudo_life", "move"):
        info = summary["named"].get(label)
        if info and info["period"] > 0:
            ax.scatter([info["lam"]], [info["transient"]], s=28, color=INK, zorder=4)
            ax.annotate(label.replace("_", " ").title().replace("Highlife", "HighLife").replace("Drylife", "DryLife"),
                        (info["lam"], info["transient"]), xytext=(6, -12), textcoords="offset points",
                        fontsize=8.5, color=INK, path_effects=HALO)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Langton's λ")
    ax.set_ylabel("generations before settling")
    ax.set_title("B  Life churns far longer than almost any rule that settles", fontsize=10.5)

    long_pct = 100 * summary["long_lived"] / n
    sparse = summary["long_lived_sparse"]
    headline(fig, "How special is Conway's rule?",
             f"Every Life-like rule (B0 excluded) run on the same {summary['size']}×{summary['size']} soup. "
             f"Only {long_pct:.1f}% churn for ≥ {LONG_TRANSIENT} generations and then settle; "
             f"{sparse} of those end sparse like Life.")
    fig.subplots_adjust(top=0.8)
    save(fig, path)

    # Life and its one-change mutants
    fig, axes = plt.subplots(3, 6, figsize=(12.6, 7.6))
    for ax, world, grid in zip(axes.ravel(), worlds, finals):
        label = "Conway's Life  B3/S23" if world["label"] == "Life" else f"{world['label']}  {world['rule']}"
        if world.get("name") and world["label"] != "Life":
            label += f"\n({world['name'].replace('_', ' ')})"
        if world["period"] < 0:
            status = f"still changing at {STEPS_MUTANTS:,}"
        else:
            status = {"dies out": "died", "freezes": "froze", "oscillates": "settled"}[world["cls"]]
            status += f" at t = {world['transient']:,}"
        show_cells(ax, grid)
        ax.set_title(label, fontsize=8.5, color=INK, loc="center", pad=3,
                     fontweight="bold" if world["label"] == "Life" else "normal")
        ax.set_xlabel(f"{status}\n{100 * world['density']:.1f}% alive", fontsize=8, color=INK2, labelpad=3)
        if world["label"] == "Life":
            for spine in ax.spines.values():
                spine.set_color(ORANGE)
                spine.set_linewidth(2.5)
    mutants = worlds[1:]
    busy = sum(w["period"] < 0 for w in mutants)
    quick = sum(0 < w["period"] and w["transient"] < 300 for w in mutants)
    slow = sum(0 < w["period"] and w["transient"] >= 300 for w in mutants)
    headline(fig, "One change away from Life",
             f"Conway's rule and its 17 one-change neighbours from the same 128×128 soup. {busy} are still changing "
             f"at t = {STEPS_MUTANTS:,}, {quick} settle within 300 generations, and {slow} churn and then settle like Life.")
    fig.subplots_adjust(top=0.86, hspace=0.55, wspace=0.1)
    save(fig, str(path).replace("rule_space", "life_mutants"))


# --------------------------------------------------------------- avalanches
def _ccdf(t):
    t = np.sort(np.asarray(t))
    y = 1.0 - np.arange(len(t)) / len(t)
    return t, y


def plot_avalanches(d, summary, path):
    style()
    burn = summary["burn_in"]
    tmin = summary["tmin"]
    fits = {(f["size"], f["boundary"]): f for f in summary["fits"]}
    sizes = sorted({f["size"] for f in summary["fits"]})
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.22)

    for col, (boundary, nice) in enumerate([("wrap", "wrap-around (torus)"), ("dead", "dead (cold) edges")]):
        ax = fig.add_subplot(gs[0, col])
        for k, size in enumerate(sizes):
            sel = (d["size"] == size) & (d["boundary"] == (boundary == "dead")) & (d["index"] >= burn)
            t = d["lifetime"][sel]
            t = t[t >= tmin]
            if not len(t):
                continue
            x, y = _ccdf(t)
            ax.plot(x, y, color=SERIES[k], lw=1.8, label=f"L = {size}")
            f = fits.get((size, boundary))
            if f:
                xs = np.logspace(np.log10(tmin), np.log10(x.max()), 200)
                ys = (xs / tmin) ** (1 - f["b"]) * np.exp(-(xs - tmin) / f["tc"])
                ax.plot(xs, ys, color=SERIES[k], lw=1, ls="--", alpha=0.9)
        xs = np.logspace(np.log10(tmin), 4.5, 50)
        ax.plot(xs, (xs / tmin) ** (1 - 1.377) * 0.9, color=MUTED, lw=1)
        ax.text(xs[30], (xs[30] / tmin) ** (1 - 1.377) * 1.25, "pure power law (b = 1.377)", fontsize=8.5,
                color=INK2, rotation=-17)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.3)
        ax.set_xlabel("avalanche lifetime T (generations)")
        ax.set_ylabel("P(lifetime ≥ T)")
        ax.legend(loc="lower left")
        panel(ax, f"{'AB'[col]}  Avalanche lifetimes, {nice}", "solid: data · dashed: power law with cutoff (fit)")

    ax = fig.add_subplot(gs[1, 0])
    for k, (boundary, label) in enumerate([("wrap", "wrap-around"), ("dead", "dead edges")]):
        xs = [s for s in sizes if (s, boundary) in fits]
        b = np.array([fits[(s, boundary)]["b"] for s in xs])
        lo = np.array([fits[(s, boundary)]["b_ci"][0] for s in xs])
        hi = np.array([fits[(s, boundary)]["b_ci"][1] for s in xs])
        off = 1.04 if k else 0.96
        ax.errorbar(np.array(xs) * off, b, yerr=[b - lo, hi - b], color=SERIES[k], marker="o",
                    capsize=3, lw=1.5, label=label)
    ax.axhspan(1.377 - 0.009, 1.377 + 0.009, color=MUTED, alpha=0.25, lw=0)
    ax.text(0.98, 1.377 - 0.011, "Blok & Bergersen 1997: b = 1.377 ± 0.009", fontsize=8.5, color=INK2,
            ha="right", va="top", transform=ax.get_yaxis_transform())
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("grid side L")
    ax.set_ylabel("power-law exponent b")
    ax.legend(loc="lower right")
    panel(ax, "C  The power-law exponent", "maximum-likelihood fit, 95% bootstrap intervals")

    ax = fig.add_subplot(gs[1, 1])
    for k, (boundary, label) in enumerate([("wrap", "wrap-around"), ("dead", "dead edges")]):
        xs = [s for s in sizes if (s, boundary) in fits]
        tc = np.array([fits[(s, boundary)]["tc"] for s in xs])
        lo = np.array([fits[(s, boundary)]["tc_ci"][0] for s in xs])
        hi = np.array([fits[(s, boundary)]["tc_ci"][1] for s in xs])
        off = 1.04 if k else 0.96
        ax.errorbar(np.array(xs) * off, tc, yerr=[tc - lo, hi - tc], color=SERIES[k], marker="o",
                    capsize=3, lw=1.5, label=label)
    ax.axhspan(1905 - 59, 1905 + 59, color=MUTED, alpha=0.25, lw=0)
    ax.text(0.02, 1905 * 1.06, "Blok & Bergersen 1997, torus, L ≥ 128: 1905 ± 59",
            fontsize=8.5, color=INK2, ha="left", va="bottom", transform=ax.get_yaxis_transform())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}" if str(int(round(v)))[0] in "2468" else ""))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("grid side L")
    ax.set_ylabel("cutoff lifetime t_c (generations)")
    ax.legend(loc="lower right")
    panel(ax, "D  The cutoff: a typical avalanche scale", "maximum-likelihood fit, 95% bootstrap intervals")

    n = sum(f["n"] for f in summary["fits"])
    big = [fits[(s, "wrap")]["tc"] for s in sizes[-2:] if (s, "wrap") in fits]
    plateau = f" On a torus the cutoff stops growing at about {np.mean(big):,.0f} generations." if big else ""
    headline(fig, "Is the Game of Life critical?",
             f"{n:,} avalanches, each set off by flipping one cell of a settled world. Lifetimes follow a "
             f"power law only up to a cutoff.{plateau}")
    fig.subplots_adjust(top=0.88)
    save(fig, path)


# --------------------------------------------------------------- asynchrony
def async_snapshots(sigmas=(0.75, 0.9, 1.0), size=128, steps=3000, seed=4):
    from lifelab.engine import random_grid, step

    out = []
    for s in sigmas:
        rng = np.random.default_rng(seed)
        g = random_grid((size, size), 0.5, rng)
        for _ in range(steps):
            g = step(g, "B3/S23", "wrap", synchrony=s, rng=rng)
        out.append(g)
    return out


def plot_asynchrony(d, path, refine=None):
    style()
    fits = d.get("fits", {})
    sizes = sorted({p["size"] for p in d["points"]})
    fig = plt.figure(figsize=(14, 8.6))
    gs = GridSpec(2, 6, figure=fig, hspace=0.45, wspace=0.5, height_ratios=[1, 0.95])
    big = str(max(sizes))
    for col, (key, ylabel, title) in enumerate([
        ("density", "steady-state density", "A  Density of live cells"),
        ("activity", "flips per updated cell per tick", "B  Activity"),
    ]):
        ax = fig.add_subplot(gs[0, 3 * col: 3 * col + 3])
        for k, size in enumerate(sizes):
            pts = [p for p in d["points"] if p["size"] == size]
            sig = np.array([p["sigma"] for p in pts])
            val = np.array([np.mean(p[key]) for p in pts])
            if key == "activity":
                val = val / sig
            ax.plot(sig, val, color=SERIES[k], marker="o", ms=3.5, lw=1.5, label=f"L = {size}")
        fit = fits.get(big, {}).get(key)
        if refine is not None:
            rpts = [p for p in refine["points"] if p["size"] == int(big)]
            rs = np.array([p["sigma"] for p in rpts])
            rv = np.array([np.mean(p[key]) for p in rpts])
            if key == "activity":
                rv = rv / rs
            ax.plot(rs, rv, ls="none", marker="D", ms=3.5, color=INK, label=f"L = {big}, 30,000 ticks")
            fit = refine.get("fits", {}).get(big, {}).get(key, fit)
        if fit:
            xs = np.linspace(0.75, 1.0, 400)
            ys = fit["rho0"] + fit["amp"] * np.clip(fit["sigma_c"] - xs, 0, None) ** fit["beta"]
            ax.plot(xs, ys, color=INK, lw=1, ls="--", label="fit")
            ax.axvline(fit["sigma_c"], color=INK, lw=0.8)
            ax.text(fit["sigma_c"] - 0.006, ax.get_ylim()[1] * 0.92 if key == "density" else ax.get_ylim()[1] * 0.92,
                    f"σc = {fit['sigma_c']:.3f}\n(β = {fit['beta']:.2f})", ha="right", va="top", fontsize=9,
                    color=INK)
        ax.set_xlabel("synchrony σ (chance each cell updates per tick)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="center left")
        ax.set_title(title, fontsize=10.5)

    sigmas = (0.75, 0.9, 1.0)
    snaps = async_snapshots(sigmas)
    names = ["σ = 0.75: dense labyrinth", "σ = 0.90: near the transition", "σ = 1.00: Conway's Life"]
    for k, (g, name) in enumerate(zip(snaps, names)):
        ax = fig.add_subplot(gs[1, 2 * k: 2 * k + 2])
        show_cells(ax, g)
        ax.set_title(f"{'CDE'[k]}  {name}", fontsize=10, loc="left")
        ax.set_xlabel(f"{100 * g.mean():.1f}% alive after 3,000 ticks", fontsize=8.5, color=INK2)

    if refine is not None:
        fits = refine.get("fits", fits)
    sc = fits.get(big, {}).get("density", {}).get("sigma_c")
    sub = ("Each cell updates with probability σ per tick. Below a critical synchrony the world never freezes; "
           "Blok & Bergersen (1999) put it at σc ≈ 0.906")
    if sc:
        sub += f"; these runs give {sc:.3f} (L = {big})."
    headline(fig, "What if the clock is not perfect?", sub)
    fig.subplots_adjust(top=0.88)
    save(fig, path)
