"""Command line interface: ``lifelab <command>`` (or ``python -m lifelab``).

Examples::

    lifelab play                                  # interactive window (needs pygame)
    lifelab play --rule highlife --size 160x100
    lifelab render --pattern gosper --size 120x70 --steps 300 --out gun.gif
    lifelab render --density 0.35 --size 200 --steps 600 --out soup.mp4 --fps 30
    lifelab census --size 256 --seed 1
    lifelab rule B36/S23
    lifelab rules
    lifelab patterns
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np

from .analysis import census, find_cycle
from .engine import Life, random_grid
from .patterns import CATEGORY, PATTERNS, get_pattern, load, place
from .rules import NAMED_RULES, Rule


def parse_size(text: str) -> tuple[int, int]:
    """``"200"`` -> (200, 200); ``"160x90"`` -> (90 rows, 160 columns)."""
    if "x" in text.lower():
        w, h = text.lower().split("x")
        return int(h), int(w)
    return int(text), int(text)


def build_grid(args) -> np.ndarray:
    shape = parse_size(args.size)
    rng = np.random.default_rng(args.seed)
    if getattr(args, "pattern", None):
        grid = np.zeros(shape, np.uint8)
        for spec in args.pattern:
            name, _, at = spec.partition("@")
            pat = load(name) if name.endswith((".rle", ".cells")) else get_pattern(name)
            if at:
                top, left = (int(v) for v in at.split(","))
            else:
                top, left = (shape[0] - pat.shape[0]) // 2, (shape[1] - pat.shape[1]) // 2
            place(grid, pat, top, left, wrap=args.boundary == "wrap")
        if args.density:
            grid |= random_grid(shape, args.density, rng)
        return grid
    return random_grid(shape, args.density or 0.35, rng)


def cmd_render(args) -> None:
    from .render import save_animation, save_png

    world = Life(build_grid(args), args.rule, args.boundary, args.synchrony, seed=args.seed)
    if args.out.lower().endswith(".png"):
        world.step(args.steps)
        save_png(world.grid, args.out, theme=args.theme, scale=args.scale)
    else:
        def frames():
            yield world.grid
            for _ in range(args.steps):
                for _ in range(args.skip):
                    world.step()
                yield world.grid

        kwargs = dict(fps=args.fps, theme=args.theme, trail=args.trail, scale=args.scale)
        save_animation(frames(), args.out, **kwargs)
    print(f"wrote {args.out}  ({world.rule}, {world.generation} generations, population {world.population})")


def cmd_play(args) -> None:
    from .viewer import run_viewer

    run_viewer(build_grid(args), rule=args.rule, boundary=args.boundary, synchrony=args.synchrony,
               cell=args.cell, fps=args.fps, theme=args.theme, trail=args.trail, skulls=args.skulls,
               max_frames=args.frames, seed=args.seed)


def cmd_census(args) -> None:
    grid = build_grid(args)
    cyc = find_cycle(grid, args.rule, args.boundary, max_steps=args.max_steps)
    if not cyc.settled:
        print(f"did not settle within {args.max_steps} generations")
        return
    counts: Counter = census(cyc.grid, wrap=args.boundary == "wrap")
    total = sum(counts.values())
    print(f"settled after {cyc.transient} generations into a cycle of period {cyc.period}; "
          f"{int(cyc.grid.sum())} live cells, {total} objects")
    for name, n in counts.most_common():
        print(f"  {name:28s} {n:6d}  {100 * n / total:5.1f}%")


def cmd_rule(args) -> None:
    rule = Rule.parse(args.rule)
    print(f"{rule}  ({rule.name or 'unnamed'})")
    print(f"  births on {sorted(rule.birth)} neighbours, survives on {sorted(rule.survival)}")
    print(f"  Langton's lambda = {rule.langton_lambda:.4f}, 18-bit code = {rule.code}")
    print("  one change away:")
    for label, mutant in rule.mutants():
        print(f"    {label:4s} {str(mutant):18s} {mutant.name or ''}")


def cmd_rules(_args) -> None:
    for name, text in NAMED_RULES.items():
        rule = Rule.parse(text)
        print(f"  {name:20s} {text:16s} lambda = {rule.langton_lambda:.3f}")


def cmd_patterns(_args) -> None:
    for name, pat in PATTERNS.items():
        print(f"  {name:20s} {CATEGORY[name]:11s} {pat.shape[1]}x{pat.shape[0]}, {int(pat.sum())} cells")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="lifelab", description="A small laboratory for the Game of Life.")
    sub = parser.add_subparsers(dest="command", required=True)

    def world_args(p, size="120x80"):
        p.add_argument("--rule", default="B3/S23", help="B/S rule or name (default: B3/S23)")
        p.add_argument("--size", default=size, help="WIDTHxHEIGHT or a single number")
        p.add_argument("--density", type=float, default=None, help="random soup density (default 0.35)")
        p.add_argument("--pattern", action="append", help="pattern name or .rle file, optionally @row,col")
        p.add_argument("--boundary", choices=("wrap", "dead"), default="wrap")
        p.add_argument("--synchrony", type=float, default=1.0, help="chance each cell updates per tick")
        p.add_argument("--seed", type=int, default=None)

    p = sub.add_parser("render", help="write a GIF, MP4 or PNG")
    world_args(p)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--skip", type=int, default=1, help="generations per frame")
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--scale", type=int, default=4, help="pixels per cell")
    p.add_argument("--theme", default="dark", choices=("dark", "light", "ember", "mint"))
    p.add_argument("--trail", type=float, default=0.8, help="glow kept per frame, 0-1")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("play", help="interactive window (needs pygame)")
    world_args(p, size="160x100")
    p.add_argument("--cell", type=int, default=6, help="pixels per cell")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--theme", default="dark", choices=("dark", "light", "ember", "mint"))
    p.add_argument("--trail", type=float, default=0.75)
    p.add_argument("--skulls", action="store_true", help="draw live cells as skulls, like the 2023 original")
    p.add_argument("--frames", type=int, default=None, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("census", help="run a soup until it settles and count the objects")
    world_args(p, size="256")
    p.add_argument("--max-steps", type=int, default=50_000)
    p.set_defaults(func=cmd_census)

    p = sub.add_parser("rule", help="describe a rule")
    p.add_argument("rule")
    p.set_defaults(func=cmd_rule)

    sub.add_parser("rules", help="list named rules").set_defaults(func=cmd_rules)
    sub.add_parser("patterns", help="list built-in patterns").set_defaults(func=cmd_patterns)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
