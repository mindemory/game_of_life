"""Turn simulations into pictures, GIFs and MP4 videos.

Cells are drawn with a fading "trail": a cell that just died glows for a
few frames, which makes motion (gliders, waves of activity) easy to follow.
GIFs need only Pillow. MP4s need an ``ffmpeg`` executable, found on the
``PATH``, through the ``LIFELAB_FFMPEG`` variable, or via the optional
``imageio-ffmpeg`` package.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
from PIL import Image

__all__ = ["THEMES", "Painter", "save_gif", "save_mp4", "save_png", "save_animation", "find_ffmpeg"]

#: Colour themes: background, live cell, and the colour a dying cell fades from.
THEMES = {
    "dark": dict(background="#0b1220", alive="#f4f7fb", trail="#2a78d6"),
    "light": dict(background="#fcfcfb", alive="#1f2a37", trail="#86b6ef"),
    "ember": dict(background="#120c0a", alive="#fff4e6", trail="#eb6834"),
    "mint": dict(background="#08130f", alive="#effbf6", trail="#1baf7a"),
}

_LEVELS = 24  # shades used for the fading trail


def _rgb(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.array([int(color[i: i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


class Painter:
    """Keeps the fading trail between frames and converts grids to images.

    ``trail`` is the fraction of glow kept per frame (0 = no trail).
    """

    def __init__(self, theme: str = "dark", trail: float = 0.8, scale: int = 4, gap: bool = False) -> None:
        if theme not in THEMES:
            raise ValueError(f"theme must be one of {sorted(THEMES)}")
        colors = THEMES[theme]
        self.scale = int(scale)
        self.trail = float(trail)
        self.gap = gap and self.scale >= 4
        bg, alive, glow = (_rgb(colors[k]) for k in ("background", "alive", "trail"))
        shades = [bg + (glow - bg) * (k / _LEVELS) ** 1.3 for k in range(_LEVELS + 1)]
        palette = np.array(shades + [alive]).round().astype(np.uint8)  # index 0 = background
        self.palette = palette
        self.heat: Optional[np.ndarray] = None

    def indices(self, grid: np.ndarray) -> np.ndarray:
        """Palette indices for one grid (updates the trail)."""
        grid = np.asarray(grid, dtype=bool)
        if self.heat is None or self.heat.shape != grid.shape:
            self.heat = np.zeros(grid.shape)
        self.heat *= self.trail
        self.heat[grid] = 1.0
        idx = np.rint(self.heat * _LEVELS).astype(np.uint8)
        idx[grid] = _LEVELS + 1
        if self.scale > 1:
            idx = np.repeat(np.repeat(idx, self.scale, 0), self.scale, 1)
            if self.gap:
                idx[self.scale - 1:: self.scale, :] = np.minimum(idx[self.scale - 1:: self.scale, :], _LEVELS // 3)
                idx[:, self.scale - 1:: self.scale] = np.minimum(idx[:, self.scale - 1:: self.scale], _LEVELS // 3)
        return idx

    def rgb(self, grid: np.ndarray) -> np.ndarray:
        """An ``(H, W, 3)`` RGB image for one grid."""
        return self.palette[self.indices(grid)]

    def image(self, grid: np.ndarray) -> Image.Image:
        img = Image.fromarray(self.indices(grid), mode="P")
        img.putpalette(self.palette.ravel().tolist())
        return img


def save_png(grid: np.ndarray, path: str | Path, theme: str = "light", scale: int = 4) -> Path:
    Painter(theme, trail=0.0, scale=scale).image(grid).convert("RGB").save(path)
    return Path(path)


def save_gif(frames: Iterable[np.ndarray], path: str | Path, fps: int = 20, theme: str = "dark",
             trail: float = 0.8, scale: int = 4, hold_last: int = 0) -> Path:
    """Write grids to an animated GIF (looping forever)."""
    painter = Painter(theme, trail, scale)
    images = [painter.image(f) for f in frames]
    durations = [int(1000 / fps)] * len(images)
    if hold_last and images:
        durations[-1] += int(hold_last * 1000 / fps)
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0,
                   optimize=False, disposal=1)
    return Path(path)


def find_ffmpeg() -> Optional[str]:
    """Locate an ffmpeg executable, or return None."""
    exe = os.environ.get("LIFELAB_FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - optional dependency
        return None


def save_mp4(frames: Iterable[np.ndarray] | Iterator[np.ndarray], path: str | Path, fps: int = 30,
             theme: str = "dark", trail: float = 0.8, scale: int = 4, crf: int = 18,
             rgb_frames: bool = False) -> Path:
    """Write grids (or ready-made RGB frames with ``rgb_frames=True``) to an H.264 MP4."""
    exe = find_ffmpeg()
    if exe is None:
        raise RuntimeError("ffmpeg not found: install it, set LIFELAB_FFMPEG, or `pip install imageio-ffmpeg`")
    painter = Painter(theme, trail, scale)
    proc = None
    try:
        for frame in frames:
            img = frame if rgb_frames else painter.rgb(frame)
            h, w = img.shape[:2]
            if proc is None:
                # H.264 with yuv420p needs even dimensions.
                pad_h, pad_w = h % 2, w % 2
                cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{w + pad_w}x{h + pad_h}", "-r", str(fps), "-i", "-",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
                       "-preset", "slow", "-movflags", "+faststart", str(path)]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            if pad_h or pad_w:
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            proc.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())
    finally:
        if proc is not None:
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError("ffmpeg failed")
    return Path(path)


def save_animation(frames, path: str | Path, **kwargs) -> Path:
    """Pick GIF or MP4 from the file extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".gif":
        return save_gif(frames, path, **kwargs)
    if suffix in (".mp4", ".m4v", ".mov"):
        return save_mp4(frames, path, **kwargs)
    raise ValueError("output must end in .gif or .mp4")
