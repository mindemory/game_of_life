"""Interactive pygame window: draw cells, switch rules, change synchrony.

Run with ``lifelab play`` (needs ``pip install pygame``). Press ``h`` in
the window for the key list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .engine import Life, neighbor_counts, random_grid
from .patterns import get_pattern, place
from .render import THEMES, Painter
from .rules import Rule

#: Rules on the number keys 1-9.
PRESETS = ["life", "highlife", "day_and_night", "seeds", "maze", "coral", "replicator", "2x2", "diamoeba"]
#: Patterns cycled with Tab and stamped with s.
STAMPS = ["glider", "lwss", "r_pentomino", "acorn", "pulsar", "pentadecathlon", "gosper_glider_gun", "diehard"]

HELP = [
    "space  play / pause      n  single step",
    "r  random soup     c  clear     f  flip one cell",
    "mouse  draw (left), erase (right)",
    "tab  next stamp    s  stamp it at the mouse",
    "1-9  rule presets  w  wrap-around / dead edges",
    "[ ]  synchrony     - =  speed     t  colours",
    "p  save screenshot     h  help     q  quit",
]


def run_viewer(
    grid: np.ndarray,
    rule="B3/S23",
    boundary: str = "wrap",
    synchrony: float = 1.0,
    cell: int = 6,
    fps: int = 20,
    theme: str = "dark",
    trail: float = 0.75,
    skulls: bool = False,
    max_frames: Optional[int] = None,
    seed: Optional[int] = None,
) -> Life:
    import pygame

    world = Life(grid, rule, boundary, synchrony, seed=seed)
    h, w = world.shape
    hud = 26
    pygame.init()
    screen = pygame.display.set_mode((w * cell, h * cell + hud))
    pygame.display.set_caption("lifelab · Game of Life")
    font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 13)
    clock = pygame.time.Clock()
    themes = list(THEMES)
    theme_i = themes.index(theme)
    painter = Painter(theme, trail, scale=1)

    skull = None
    if skulls:  # the 2023 original drew every live cell as a skull
        path = Path(__file__).resolve().parents[1] / "classic" / "img" / "skull.png"
        if path.exists():
            skull = pygame.transform.smoothscale(pygame.image.load(str(path)), (cell, cell))

    playing, show_help, running = True, False, True
    stamp_i, frame = 0, 0
    shots = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                key = event.key
                if key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif key == pygame.K_SPACE:
                    playing = not playing
                elif key == pygame.K_n:
                    world.step()
                elif key == pygame.K_r:
                    world.grid = random_grid(world.shape, 0.35, world.rng)
                    world.generation = 0
                elif key == pygame.K_c:
                    world.grid = np.zeros(world.shape, np.uint8)
                    world.generation = 0
                elif key == pygame.K_f:
                    near = np.flatnonzero((world.grid > 0) | (neighbor_counts(world.grid, world.boundary) > 0))
                    if len(near):
                        world.grid.flat[world.rng.choice(near)] ^= 1
                elif key == pygame.K_w:
                    world.boundary = "dead" if world.boundary == "wrap" else "wrap"
                elif key == pygame.K_LEFTBRACKET:
                    world.synchrony = max(0.0, round(world.synchrony - 0.01, 2))
                elif key == pygame.K_RIGHTBRACKET:
                    world.synchrony = min(1.0, round(world.synchrony + 0.01, 2))
                elif key == pygame.K_MINUS:
                    fps = max(1, fps - 5)
                elif key == pygame.K_EQUALS:
                    fps = min(240, fps + 5)
                elif key == pygame.K_TAB:
                    stamp_i = (stamp_i + 1) % len(STAMPS)
                elif key == pygame.K_s:
                    mx, my = pygame.mouse.get_pos()
                    place(world.grid, get_pattern(STAMPS[stamp_i]), (my - hud) // cell, mx // cell,
                          wrap=world.boundary == "wrap")
                elif key == pygame.K_t:
                    theme_i = (theme_i + 1) % len(themes)
                    painter = Painter(themes[theme_i], trail, scale=1)
                elif key == pygame.K_p:
                    shots += 1
                    name = f"lifelab_{world.generation:06d}_{shots}.png"
                    pygame.image.save(screen, name)
                    print(f"saved {name}")
                elif key == pygame.K_h:
                    show_help = not show_help
                elif pygame.K_1 <= key <= pygame.K_9:
                    world.rule = Rule.parse(PRESETS[key - pygame.K_1])

        buttons = pygame.mouse.get_pressed()
        if buttons[0] or buttons[2]:
            mx, my = pygame.mouse.get_pos()
            r, c = (my - hud) // cell, mx // cell
            if 0 <= r < h and 0 <= c < w:
                world.grid[r, c] = 1 if buttons[0] else 0

        if playing:
            world.step()

        if skull is not None:
            screen.fill(tuple(painter.palette[0]))
            for r, c in np.argwhere(world.grid):
                screen.blit(skull, (c * cell, hud + r * cell))
        else:
            rgb = painter.rgb(world.grid)
            surf = pygame.surfarray.make_surface(np.ascontiguousarray(rgb.swapaxes(0, 1)))
            screen.blit(pygame.transform.scale(surf, (w * cell, h * cell)), (0, hud))

        bar = pygame.Rect(0, 0, w * cell, hud)
        pygame.draw.rect(screen, (17, 24, 39), bar)
        name = world.rule.name or ""
        status = (f"{world.rule} {name}  gen {world.generation}  pop {world.population}  "
                  f"sync {world.synchrony:.2f}  {world.boundary}  {fps} fps  "
                  f"stamp: {STAMPS[stamp_i]}  {'' if playing else '[paused]'}  h: help")
        screen.blit(font.render(status, True, (229, 231, 235)), (8, 6))
        if show_help:
            box = pygame.Surface((min(w * cell - 20, 430), 22 * len(HELP) + 16), pygame.SRCALPHA)
            box.fill((17, 24, 39, 225))
            for i, line in enumerate(HELP):
                box.blit(font.render(line, True, (229, 231, 235)), (12, 10 + 22 * i))
            screen.blit(box, (10, hud + 10))
        pygame.display.flip()

        frame += 1
        if max_frames is not None and frame >= max_frames:
            running = False
        clock.tick(fps)
    pygame.quit()
    return world
