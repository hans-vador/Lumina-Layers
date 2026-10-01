"""BandSchedule: the global layer/filament plan of a Band-mode print.

Bambu layer numbering: layer n (1-based) has top z_n = FL + lh*(n-1).
Band b (bottom -> top) has layer_counts[b] layers; band b >= 1 starts at
layer s_b = cum(b-1) + 1 and the swap entry for it is
(top_z(s_b), extruder = b+1, hex_b).

Reference parity (Astroworld): names (Black, Red, Sunny Orange, White),
counts (9, 4, 6, 8), FL 0.16 -> swaps at 0.88 / 1.20 / 1.68 mm.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Mapping

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from core.band.optics import Filament

MAX_SLOTS = 5   # width of the stacks() matrix (Lumina 5-slot material_matrix)


@dataclass(frozen=True)
class BandSchedule:
    filament_names: tuple[str, ...]       # bottom -> top
    layer_counts: tuple[int, ...]         # one per band, base first
    first_layer_mm: float = 0.16
    layer_h: float = 0.08
    base_min_layers: int = 5

    def __post_init__(self):
        names = tuple(str(n) for n in self.filament_names)
        counts = tuple(int(c) for c in self.layer_counts)
        object.__setattr__(self, 'filament_names', names)
        object.__setattr__(self, 'layer_counts', counts)
        object.__setattr__(self, 'first_layer_mm', float(self.first_layer_mm))
        object.__setattr__(self, 'layer_h', float(self.layer_h))
        object.__setattr__(self, 'base_min_layers', int(self.base_min_layers))
        if len(names) != len(counts):
            raise ValueError("filament_names and layer_counts must have the same length")
        if not 1 <= len(names) <= 32:
            raise ValueError(f"need 1..32 bands, got {len(names)}")
        if any(c < 1 for c in counts):
            raise ValueError("every band needs at least 1 layer")
        if self.layer_h <= 0 or self.first_layer_mm <= 0:
            raise ValueError("layer heights must be positive")
        if not 1 <= self.base_min_layers <= counts[0]:
            raise ValueError("base_min_layers must be in 1..layer_counts[0]")

    # ----- basic geometry ------------------------------------------------- #
    @property
    def n_bands(self) -> int:
        return len(self.layer_counts)

    @property
    def n_layers(self) -> int:
        return int(sum(self.layer_counts))

    @property
    def n0(self) -> int:
        return self.layer_counts[0]

    @property
    def total_height_mm(self) -> float:
        return self.top_z(self.n_layers)

    def cum(self, b: int) -> int:
        """Cumulative layer count through band b (inclusive); cum(-1) == 0."""
        if b < 0:
            return 0
        return int(sum(self.layer_counts[:b + 1]))

    def top_z(self, n) -> float | np.ndarray:
        """Top z of layer n (1-based). Works on ints and arrays."""
        n = np.asarray(n, dtype=np.float64)
        z = self.first_layer_mm + self.layer_h * (n - 1)
        return float(z) if z.ndim == 0 else z

    def layer_of_height(self, h) -> np.ndarray:
        """Number of printed layers in a column of height h (slice-plane rule).

        Layer n (>=2) is present iff h > top_z(n) - lh/2; layer 1 iff h > FL/2.
        """
        h = np.asarray(h, dtype=np.float64)
        n = np.floor((h - self.first_layer_mm) / self.layer_h + 1.5).astype(np.int64)
        n = np.where(h > self.first_layer_mm / 2.0, np.maximum(n, 1), 0)
        return n

    def band_first_layer(self, b: int) -> int:
        """1-based index of the first layer printed with band b."""
        if not 0 <= b < self.n_bands:
            raise IndexError(b)
        return self.cum(b - 1) + 1

    def band_of_layer(self, n) -> np.ndarray:
        """Band index for layer number n (1-based), clipped to the last band."""
        n = np.asarray(n)
        cums = np.array([self.cum(b) for b in range(self.n_bands)])
        return np.minimum((n[..., None] > cums).sum(-1), self.n_bands - 1)

    # ----- ramp rows ------------------------------------------------------ #
    def k_rows(self) -> range:
        """Column layer counts that have a distinct colour: n0..K."""
        return range(self.n0, self.n_layers + 1)

    def stacks(self) -> np.ndarray:
        """(K',5) int32: per-band layer counts of a k-layer column, zero-padded."""
        k = np.arange(self.n0, self.n_layers + 1, dtype=np.int32)
        out = np.zeros((k.size, max(MAX_SLOTS, self.n_bands)), dtype=np.int32)
        out[:, 0] = np.minimum(k, self.n0)
        for b in range(1, self.n_bands):
            out[:, b] = np.clip(k - self.cum(b - 1), 0, self.layer_counts[b])
        return out

    def heights(self) -> np.ndarray:
        """(K',) float top z of each ramp row."""
        return np.asarray(self.top_z(np.arange(self.n0, self.n_layers + 1)), dtype=np.float64)

    # ----- swaps ---------------------------------------------------------- #
    def swap_entries(self, filaments: Mapping[str, 'Filament']) -> list[tuple[float, int, str]]:
        """[(top_z, extruder 1-based = band index + 1, '#RRGGBB'), ...] for bands >= 1."""
        entries = []
        slots = {name: i+1 for i, name in enumerate(dict.fromkeys(self.filament_names))}
        for b in range(1, self.n_bands):
            z = round(self.top_z(self.band_first_layer(b)), 8)
            entries.append((z, slots[self.filament_names[b]], filaments[self.filament_names[b]].hex))
        return entries

    def swap_instructions(self, filaments: Mapping[str, 'Filament']) -> str:
        slots = {name: i+1 for i, name in enumerate(dict.fromkeys(self.filament_names))}
        lines = [
            f"Start with {self.filament_names[0]} "
            f"({filaments[self.filament_names[0]].hex}, extruder 1) - "
            f"{self.layer_counts[0]} layers."
        ]
        for b in range(1, self.n_bands):
            n = self.band_first_layer(b)
            name = self.filament_names[b]
            lines.append(
                f"At layer {n} ({self.top_z(n):.2f} mm) swap to {name} "
                f"({filaments[name].hex}, extruder {slots[name]}) - {self.layer_counts[b]} layers."
            )
        lines.append(
            f"Total {self.n_layers} layers = {self.total_height_mm:.2f} mm "
            f"(first layer {self.first_layer_mm:.2f}, layer height {self.layer_h:.2f}); "
            f"{self.n_bands - 1} filament change(s)."
        )
        td = ', '.join(
            f"{n} TD {filaments[n].td_mm:g} ({filaments[n].td_source})" for n in self.filament_names
        )
        lines.append(f"Transmission distances (mm): {td}.")
        return '\n'.join(lines)

    # ----- identity / serialisation --------------------------------------- #
    def key(self) -> str:
        payload = json.dumps(
            [list(self.filament_names), list(self.layer_counts),
             round(self.first_layer_mm, 6), round(self.layer_h, 6)],
            separators=(',', ':'), ensure_ascii=False,
        )
        return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:10]

    def to_dict(self) -> dict:
        return {
            'filament_names': list(self.filament_names),
            'layer_counts': list(self.layer_counts),
            'first_layer_mm': self.first_layer_mm,
            'layer_h': self.layer_h,
            'base_min_layers': self.base_min_layers,
            'n_layers': self.n_layers,
            'total_height_mm': self.total_height_mm,
            'key': self.key(),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> 'BandSchedule':
        return cls(
            filament_names=tuple(d['filament_names']),
            layer_counts=tuple(int(c) for c in d['layer_counts']),
            first_layer_mm=float(d.get('first_layer_mm', 0.16)),
            layer_h=float(d.get('layer_h', 0.08)),
            base_min_layers=int(d.get('base_min_layers', 5)),
        )

    def with_counts(self, layer_counts) -> 'BandSchedule':
        return replace(self, layer_counts=tuple(int(c) for c in layer_counts))

    def __str__(self) -> str:
        bands = ', '.join(f"{n}x{c}" for n, c in zip(self.filament_names, self.layer_counts))
        return f"BandSchedule[{bands}; FL {self.first_layer_mm:g} lh {self.layer_h:g}; K={self.n_layers}]"
