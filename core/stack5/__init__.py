"""Stack5 mode: Lumina's native per-pixel 5-layer material stacks with a custom
5-filament palette chosen from the user's spool library.

* :mod:`core.stack5.lut`      - enumerate the 5^5 stacks, synthesise the LUT
                                (face-down Beer-Lambert compositing) in Lumina's
                                .npz {rgb, stacks} format, register the colour mode.
* :mod:`core.stack5.palette`  - exhaustive C(12,5) x backing palette search scored the
                                way Lumina matches (OpenCV 8-bit Lab nearest LUT colour
                                per histogram bin, reported as true dE76).
* :mod:`core.stack5.flush`    - Bambu Studio flush-volume formula, Auto-For-Flush tool
                                ordering, prime-tower sizing and plate layout.
* :mod:`core.stack5.cleanup`  - sub-nozzle island removal + printability statistics.
* :mod:`core.stack5.pipeline` - image -> palette -> LUT -> core.converter (no mask
                                dilation, min-region cleanup) -> X2D 3MF (imported
                                lazily: it pulls in core.converter / gradio).
"""
from core.stack5.lut import (enumerate_stacks, synth_lut, save_lut_npz, register_stack5_mode,
                             choose_backing, pure_stack_indices, stack5_key)
from core.stack5.palette import select_palette, image_hist

__all__ = ['enumerate_stacks', 'synth_lut', 'save_lut_npz', 'register_stack5_mode',
           'choose_backing', 'pure_stack_indices', 'stack5_key', 'select_palette',
           'image_hist', 'convert_album_stack5']


def __getattr__(name):
    if name == 'convert_album_stack5':
        from core.stack5.pipeline import convert_album_stack5
        return convert_album_stack5
    raise AttributeError(name)
