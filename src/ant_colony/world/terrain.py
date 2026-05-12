"""Terrain map with Perlin-noise elevation and colour mapping."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


def _fade(t: float) -> float:
    return t * t * t * (t * (t * 6 - 15) + 10)

def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)

def _gradient(hash_val: int, x: float, y: float) -> float:
    h = hash_val & 3
    u = x if h < 2 else -x if h == 2 else y
    v = y if h == 0 else -y if h == 1 else x
    return u + v

class _PerlinNoise:
    def __init__(self, seed: int = 0):
        p = list(range(256))
        rng = random.Random(seed)
        rng.shuffle(p)
        self.perm = p + p

    def noise2d(self, x: float, y: float) -> float:
        x, y = float(x), float(y)
        X = int(math.floor(x)) & 255
        Y = int(math.floor(y)) & 255
        xf = x - math.floor(x)
        yf = y - math.floor(y)
        u = _fade(xf)
        v = _fade(yf)
        aa = self.perm[self.perm[X] + Y]
        ab = self.perm[self.perm[X] + Y + 1]
        ba = self.perm[self.perm[X + 1] + Y]
        bb = self.perm[self.perm[X + 1] + Y + 1]
        x1 = _lerp(_gradient(aa, xf, yf), _gradient(ba, xf - 1, yf), u)
        x2 = _lerp(_gradient(ab, xf, yf - 1), _gradient(bb, xf - 1, yf - 1), u)
        return _lerp(x1, x2, v)


_COLOUR_RAMP = [
    (0.25, 0.35, 0.55),
    (0.30, 0.45, 0.65),
    (0.55, 0.70, 0.45),
    (0.65, 0.80, 0.40),
    (0.55, 0.55, 0.35),
    (0.50, 0.45, 0.35),
    (0.70, 0.70, 0.70),
]


@dataclass
class TerrainMap:
    """Elevation map with colour rendering."""

    width: int = 200
    height: int = 200
    scale: float = 12.0
    octaves: int = 4
    persistence: float = 0.5
    seed: int = 42

    elevation: list[list[float]] = field(default_factory=list)
    colours: list[list[tuple[float, float, float]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.elevation:
            self._generate()

    def _generate(self) -> None:
        noise = _PerlinNoise(self.seed)
        elev = [[0.0] * self.height for _ in range(self.width)]
        colours = [[(0.0, 0.0, 0.0)] * self.height for _ in range(self.width)]
        min_e = float("inf")
        max_e = float("-inf")

        for x in range(self.width):
            for y in range(self.height):
                nx = x / self.scale
                ny = y / self.scale
                val = 0.0
                amp = 1.0
                freq = 1.0
                for _ in range(self.octaves):
                    val += noise.noise2d(nx * freq, ny * freq) * amp
                    amp *= self.persistence
                    freq *= 2.0
                elev[x][y] = val
                if val < min_e: min_e = val
                if val > max_e: max_e = val

        span = max_e - min_e if max_e > min_e else 1.0
        ramp_len = len(_COLOUR_RAMP)

        for x in range(self.width):
            for y in range(self.height):
                nv = (elev[x][y] - min_e) / span
                elev[x][y] = nv
                idx = nv * (ramp_len - 1)
                i = int(idx)
                frac = idx - i
                if i >= ramp_len - 1:
                    colours[x][y] = _COLOUR_RAMP[-1]
                else:
                    c0 = _COLOUR_RAMP[i]
                    c1 = _COLOUR_RAMP[i + 1]
                    colours[x][y] = (
                        c0[0] + frac * (c1[0] - c0[0]),
                        c0[1] + frac * (c1[1] - c0[1]),
                        c0[2] + frac * (c1[2] - c0[2]),
                    )

        self.elevation = elev
        self.colours = colours

    def get_elevation(self, x: float, y: float) -> float:
        ix, iy = int(x), int(y)
        if 0 <= ix < self.width and 0 <= iy < self.height:
            return self.elevation[ix][iy]
        return 0.5

    def get_colour(self, x: float, y: float) -> tuple[float, float, float]:
        ix, iy = int(x), int(y)
        if 0 <= ix < self.width and 0 <= iy < self.height:
            return self.colours[ix][iy]
        return (0.5, 0.5, 0.5)

    def get_rgb(self, x: float, y: float) -> tuple[int, int, int]:
        r, g, b = self.get_colour(x, y)
        return (int(r * 255), int(g * 255), int(b * 255))

    def summary(self) -> dict:
        e_min = min(min(row) for row in self.elevation)
        e_max = max(max(row) for row in self.elevation)
        return {
            "width": self.width, "height": self.height,
            "scale": self.scale, "octaves": self.octaves,
            "elevation_range": (e_min, e_max),
        }

    def __repr__(self) -> str:
        return f"TerrainMap({self.width}x{self.height}, scale={self.scale})"
