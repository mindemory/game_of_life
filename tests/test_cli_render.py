import os

import numpy as np
import pytest

from lifelab.cli import main, parse_size
from lifelab.render import Painter, save_gif, save_png


def test_parse_size():
    assert parse_size("200") == (200, 200)
    assert parse_size("160x90") == (90, 160)


def test_painter_trail_and_scale():
    painter = Painter("dark", trail=0.5, scale=3)
    grid = np.zeros((4, 5), np.uint8)
    grid[1, 1] = 1
    first = painter.indices(grid)
    assert first.shape == (12, 15)
    assert first.max() == len(painter.palette) - 1  # live colour
    faded = painter.indices(np.zeros_like(grid))
    assert 0 < faded[4, 4] < first[4, 4]  # dead cell still glowing
    assert painter.rgb(grid).shape == (12, 15, 3)


def test_save_gif_and_png(tmp_path):
    frames = [np.random.default_rng(i).integers(0, 2, (10, 12)).astype(np.uint8) for i in range(4)]
    gif = save_gif(frames, tmp_path / "a.gif", fps=10, scale=2)
    assert gif.stat().st_size > 0
    png = save_png(frames[0], tmp_path / "a.png", scale=2)
    assert png.stat().st_size > 0


def test_cli_commands(tmp_path, capsys):
    main(["rule", "B36/S23"])
    assert "highlife" in capsys.readouterr().out
    main(["rules"])
    main(["patterns"])
    out = tmp_path / "gun.gif"
    main(["render", "--pattern", "gosper@2,2", "--size", "60x40", "--steps", "10", "--out", str(out)])
    assert out.exists()
    main(["census", "--size", "48", "--seed", "3", "--density", "0.35"])
    assert "objects" in capsys.readouterr().out


def test_viewer_headless():
    pygame = pytest.importorskip("pygame")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from lifelab.viewer import run_viewer

    world = run_viewer(np.zeros((20, 30), np.uint8), cell=4, max_frames=5, fps=200)
    assert world.generation == 5
    world = run_viewer(np.ones((10, 10), np.uint8), cell=4, max_frames=2, fps=200, skulls=True)
    assert world.generation == 2
