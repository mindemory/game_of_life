"""Render the animations used in the README, the project page and the video.

    python experiments/make_media.py            # GIFs + MP4 clips + the explainer video
    python experiments/make_media.py --gifs     # GIFs only (no ffmpeg needed)

GIFs go to ``docs/assets/`` (shared by the README and the project page). MP4
clips and the explainer video go to ``media/`` and need ffmpeg (see
``lifelab.render.find_ffmpeg``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import RESULTS, ROOT, load_json  # noqa: E402

from lifelab.analysis import find_cycle
from lifelab.engine import neighbor_counts, random_grid, step
from lifelab.patterns import get_pattern, place
from lifelab.render import THEMES, Painter, _rgb, find_ffmpeg, save_mp4

MEDIA = ROOT / "media"
DOCS_ASSETS = ROOT / "docs" / "assets"
MEDIA.mkdir(exist_ok=True)
DOCS_ASSETS.mkdir(parents=True, exist_ok=True)

BG = tuple(int(v) for v in _rgb(THEMES["dark"]["background"]))
TEXT = (244, 247, 251)
TEXT2 = (165, 177, 194)
ACCENT = (235, 104, 52)
BLUE = (42, 120, 214)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    try:
        from matplotlib import font_manager

        candidates.append(font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")))
    except Exception:  # pragma: no cover
        pass
    for path in candidates:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def wrap(draw, text, fnt, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=fnt) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


def to_gif(frames_rgb, path, fps=20, colors=64):
    """Quantise RGB frames to one shared palette and write a looping GIF."""
    sample = np.concatenate([f.reshape(-1, 3) for f in frames_rgb[:: max(1, len(frames_rgb) // 8)]])
    pal_img = Image.fromarray(sample.reshape(1, -1, 3)).quantize(colors=colors, method=Image.MEDIANCUT)
    images = [Image.fromarray(f).quantize(palette=pal_img, dither=0) for f in frames_rgb]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0,
                   optimize=True, disposal=1)
    print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1e6:.1f} MB, {len(images)} frames)")


def label_strip(width, text, height=34, size=17, color=TEXT, sub=None):
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    fnt = font(size, bold=True)
    d.text((8, (height - size) // 2 - 1), text, font=fnt, fill=color)
    if sub:
        x = 8 + d.textlength(text, font=fnt) + 10
        d.text((x, (height - size) // 2 + 1), sub, font=font(size - 3), fill=TEXT2)
    return np.asarray(img)


# ------------------------------------------------------------------- clips
def clip_soup(size=(84, 150), frames=200, scale=4, seed=3, theme="dark"):
    rng = np.random.default_rng(seed)
    grid = random_grid(size, 0.35, rng)
    painter = Painter(theme, trail=0.75, scale=scale)
    out = []
    for _ in range(frames):
        out.append(painter.rgb(grid))
        grid = step(grid)
    return out


def clip_gun(frames=150, scale=6):
    grid = place(np.zeros((52, 90), np.uint8), "gosper", 3, 3)
    painter = Painter("dark", trail=0.7, scale=scale)
    out = []
    for _ in range(frames):
        out.append(painter.rgb(grid))
        grid = step(grid, boundary="dead")
    return out


ZOO = [("Life", "B3/S23"), ("HighLife", "B36/S23"), ("Day & Night", "B3678/S34678"),
       ("Seeds", "B2/S"), ("Maze", "B3/S12345"), ("Coral", "B3/S45678"),
       ("Replicator", "B1357/S1357"), ("2x2", "B36/S125"), ("Diamoeba", "B35678/S5678")]


def clip_zoo(frames=180, cells=72, scale=3, seed=11, cols=3):
    rng = np.random.default_rng(seed)
    soup = random_grid((cells, cells), 0.35, rng)
    seeds_start = np.zeros((cells, cells), np.uint8)  # sparse start for explosive rules
    seeds_start[cells // 2 - 4: cells // 2 + 4, cells // 2 - 4: cells // 2 + 4] = rng.integers(0, 2, (8, 8))
    grids, painters, strips = [], [], []
    for name, rule in ZOO:
        start = seeds_start.copy() if name in ("Seeds", "Replicator") else soup.copy()
        if name == "Diamoeba":
            start = random_grid((cells, cells), 0.5, np.random.default_rng(seed + 1))
        grids.append(start)
        painters.append(Painter("dark", trail=0.7, scale=scale))
        strips.append(label_strip(cells * scale, name, sub=rule))
    gap = 6
    rows = (len(ZOO) + cols - 1) // cols
    tile_h = cells * scale + strips[0].shape[0]
    h = rows * tile_h + (rows + 1) * gap
    w = cols * cells * scale + (cols + 1) * gap
    out = []
    for _ in range(frames):
        canvas = np.empty((h, w, 3), np.uint8)
        canvas[:] = BG
        for k, (g, p, s) in enumerate(zip(grids, painters, strips)):
            r, c = divmod(k, cols)
            y = gap + r * (tile_h + gap)
            x = gap + c * (cells * scale + gap)
            canvas[y: y + s.shape[0], x: x + s.shape[1]] = s
            canvas[y + s.shape[0]: y + tile_h, x: x + cells * scale] = p.rgb(g)
        out.append(canvas)
        grids = [step(g, rule) for g, (_, rule) in zip(grids, ZOO)]
    return out


def clip_async(frames=220, cells=110, scale=3, seed=5, sigmas=(1.0, 0.85), skip=2):
    rng0 = np.random.default_rng(seed)
    start = random_grid((cells, cells), 0.5, rng0)
    grids = [start.copy() for _ in sigmas]
    rngs = [np.random.default_rng(seed + k) for k in range(len(sigmas))]
    painters = [Painter("dark", trail=0.6, scale=scale) for _ in sigmas]
    labels = ["Synchronous (σ = 1.00): Conway's Life", f"Imperfect clock (σ = {sigmas[1]:.2f})"]
    strips = [label_strip(cells * scale, lab, size=15) for lab in labels]
    gap = 8
    tile_h = cells * scale + strips[0].shape[0]
    w = len(sigmas) * cells * scale + (len(sigmas) + 1) * gap
    out = []
    for _ in range(frames):
        canvas = np.empty((tile_h + 2 * gap, w, 3), np.uint8)
        canvas[:] = BG
        for k in range(len(sigmas)):
            x = gap + k * (cells * scale + gap)
            canvas[gap: gap + strips[k].shape[0], x: x + cells * scale] = strips[k]
            canvas[gap + strips[k].shape[0]: gap + tile_h, x: x + cells * scale] = painters[k].rgb(grids[k])
        out.append(canvas)
        for _ in range(skip):
            grids = [step(g, "B3/S23", "wrap", s, r) for g, s, r in zip(grids, sigmas, rngs)]
    return out


def clip_avalanche(cells=120, scale=4, seed=21, frames=None, max_len=420):
    """A settled world, one flipped cell, and the avalanche that follows."""
    rng = np.random.default_rng(seed)
    best = None
    for attempt in range(40):  # find a perturbation with a watchable avalanche
        world = find_cycle(random_grid((cells, cells), 0.5, rng), max_steps=20_000).grid
        near = np.flatnonzero((world > 0) | (neighbor_counts(world) > 0))
        for _ in range(30):
            site = rng.choice(near)
            pert = world.copy()
            pert.flat[site] ^= 1
            cyc = find_cycle(pert, max_steps=3000)
            if cyc.settled and 150 <= cyc.transient <= max_len:
                best = (world, site, cyc.transient)
                break
        if best:
            break
    world, site, lifetime = best
    r0, c0 = divmod(int(site), cells)
    base, pert = world.copy(), world.copy()
    pert.flat[site] ^= 1
    touched = np.zeros_like(world, bool)
    ash, hot, scar = np.array([92, 104, 122]), np.array(ACCENT), np.array([86, 46, 34])
    n = frames or lifetime + 40
    out = []
    for t in range(-20, n):
        diff = (pert != base) if t >= 0 else np.zeros_like(world, bool)
        touched |= diff
        img = np.empty((cells, cells, 3), np.uint8)
        img[:] = BG
        img[touched] = scar
        show = pert if t >= 0 else base
        img[show.astype(bool)] = ash
        img[diff & show.astype(bool)] = hot
        img[diff & ~show.astype(bool)] = (hot * 0.55 + np.array(BG) * 0.45).astype(np.uint8)
        frame = np.repeat(np.repeat(img, scale, 0), scale, 1)
        if -20 <= t < 0 and (t // 4) % 2 == 0:  # blink a ring where the flip will happen
            y, x = r0 * scale + scale // 2, c0 * scale + scale // 2
            pil = Image.fromarray(frame)
            ImageDraw.Draw(pil).ellipse([x - 14, y - 14, x + 14, y + 14], outline=ACCENT, width=3)
            frame = np.asarray(pil)
        out.append(frame)
        if t >= 0:
            base, pert = step(base), step(pert)
    return out, lifetime, int(touched.sum())


# ------------------------------------------------------------- the video
def fit_into(frame, size, top=0, bottom=0):
    """Scale an RGB frame (nearest neighbour) to fit a square canvas with margins."""
    h, w = frame.shape[:2]
    avail_h = size - top - bottom
    k = min(size / w, avail_h / h)
    img = Image.fromarray(frame).resize((int(w * k), int(h * k)), Image.NEAREST)
    canvas = Image.new("RGB", (size, size), BG)
    canvas.paste(img, ((size - img.width) // 2, top + (avail_h - img.height) // 2))
    return canvas


def caption(canvas, text, size, y, fnt_size=40, color=TEXT, bold=False, margin=70, align="left"):
    d = ImageDraw.Draw(canvas)
    fnt = font(fnt_size, bold)
    for line in wrap(d, text, fnt, size - 2 * margin):
        x = margin if align == "left" else (size - d.textlength(line, font=fnt)) / 2
        d.text((x, y), line, font=fnt, fill=color)
        y += int(fnt_size * 1.3)
    return y


def card(size, title, body=None, kicker=None, background=None):
    canvas = Image.fromarray(background) if background is not None else Image.new("RGB", (size, size), BG)
    y = size // 2 - 170
    if kicker:
        y = caption(canvas, kicker, size, y, 34, ACCENT, bold=True) + 18
    y = caption(canvas, title, size, y, 74, TEXT, bold=True) + 24
    if body:
        caption(canvas, body, size, y, 38, TEXT2)
    return np.asarray(canvas)


def scene(frames, size, text, kicker=None, top=150, bottom=230):
    out = []
    for f in frames:
        canvas = fit_into(f, size, top=top, bottom=bottom)
        y = 52
        if kicker:
            y = caption(canvas, kicker, size, y, 30, ACCENT, bold=True) + 2
        caption(canvas, text, size, size - bottom + 30, 36, TEXT)
        out.append(np.asarray(canvas))
    return out


def dimmed(frames, amount=0.25):
    bg = np.array(BG, float)
    return [(bg + (f.astype(float) - bg) * amount).astype(np.uint8) for f in frames]


def hold(frame, seconds, fps):
    return [frame] * int(seconds * fps)


def explainer(size=1080, fps=30, clips=None):
    rs = load_json("rule_space.json") if (RESULTS / "rule_space.json").exists() else None
    av = load_json("avalanches.json")
    asy = load_json("asynchrony.json")
    big = str(max(int(k) for k in asy["fits"]))
    sigma_c = asy["fits"][big]["density"]["sigma_c"]
    tc = np.mean([f["tc"] for f in av["fits"] if f["boundary"] == "wrap" and f["size"] >= 256])
    b = np.mean([f["b"] for f in av["fits"]])

    seq = []
    soup = clips["soup"]
    title_bg = [np.asarray(fit_into(f, size)) for f in dimmed(soup[:120], 0.22)]
    for k, bgf in enumerate(title_bg):
        seq.append(card(size, "Is the Game of Life special?",
                        "Notes from simulating every one of its 131,072 sibling universes.",
                        kicker="A SMALL SIMULATION STUDY", background=bgf))
    seq += scene(soup[:150], size, "Each cell counts its 8 neighbours. Born with exactly 3. Survives with 2 or 3. "
                 "That's the whole rule.", kicker="CONWAY'S GAME OF LIFE (1970)")
    seq += scene(clips["zoo"][:170], size, "Change which counts mean birth or survival and you get a different "
                 "universe. There are 262,144 of these rules.", kicker="CHANGE THE RULE, CHANGE THE UNIVERSE",
                 top=120, bottom=200)
    if rs:
        n = rs["n_rules"]
        long_lived = 100 * rs["long_lived"] / n
        seq += hold(card(size, f"{long_lived:.1f}%",
                         f"of the {n:,} rules I ran keep churning for 300+ generations and then settle down. "
                         f"Conway's rule is one of them: most rules either freeze fast or never stop.",
                         kicker="HOW SPECIAL IS LIFE?"), 5, fps)
    av_frames = clips["avalanche"]
    seq += scene(av_frames[: min(len(av_frames), 260)], size,
                 f"Flip one cell of a frozen world and the disturbance can run for thousands of generations. "
                 f"But there's a typical scale (≈{round(tc, -2):,.0f} generations): Life is near-critical, not critical.",
                 kicker="IS LIFE CRITICAL?")
    seq += scene(clips["async"][:200], size,
                 f"Let each cell miss the clock tick at random. Below ≈{100 * sigma_c:.0f}% synchrony, Life never "
                 f"freezes. It becomes a living labyrinth.", kicker="WHAT IF THE CLOCK IS IMPERFECT?",
                 top=170, bottom=230)
    end_bg = [np.asarray(fit_into(f, size)) for f in dimmed(soup[150:200], 0.18)]
    for bgf in end_bg + [end_bg[-1]] * int(2.5 * fps):
        seq.append(card(size, "Code, data, figures and an interactive demo",
                        "github.com/mindemory/game_of_life", kicker="TRY IT YOURSELF", background=bgf))
    out = MEDIA / "lifelab_explainer.mp4"
    save_mp4(seq, out, fps=fps, rgb_frames=True)
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB, {len(seq) / fps:.0f} s)  b={b:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gifs", action="store_true", help="only write the GIFs")
    args = parser.parse_args()
    clips = {
        "soup": clip_soup(),
        "gun": clip_gun(),
        "zoo": clip_zoo(),
        "async": clip_async(),
    }
    av, lifetime, area = clip_avalanche()
    clips["avalanche"] = av
    print(f"avalanche clip: lifetime {lifetime}, area {area}")
    to_gif(clips["soup"][:130], DOCS_ASSETS / "soup.gif", fps=18, colors=24)
    to_gif(clips["gun"][:120], DOCS_ASSETS / "gosper_gun.gif", fps=20, colors=32)
    to_gif(clips["zoo"][:100], DOCS_ASSETS / "rule_zoo.gif", fps=12, colors=24)
    to_gif(clips["async"][:130], DOCS_ASSETS / "asynchrony.gif", fps=12, colors=24)
    to_gif(av[::2], DOCS_ASSETS / "avalanche.gif", fps=15, colors=32)
    if not args.gifs:
        if find_ffmpeg() is None:
            print("ffmpeg not found; skipping the MP4s")
            return
        for name in ("soup", "zoo", "async", "gun"):
            save_mp4(clips[name], MEDIA / f"{name}.mp4", fps=24, rgb_frames=True)
        save_mp4(av, MEDIA / "avalanche.mp4", fps=24, rgb_frames=True)
        explainer(clips=clips)


if __name__ == "__main__":
    main()
