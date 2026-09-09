"""Relief Stack5: full-colour 3D album covers.

Two independent branches, combined per pixel:

* colour   - the ORIGINAL image through the unchanged Stack5 matcher
             (:mod:`core.relief.color_stage`, delegating to :mod:`core.stack5`);
* geometry - a :class:`~core.relief.provider.GeometryProvider`
             (imported grey depth map, Depth Anything V2, Wonder3D front depth)
             -> :mod:`core.relief.depth_processing` -> 0..4 mm of whole layers.

:mod:`core.relief.relief_stack` stacks them face up (base, relief body,
stack[4] .. stack[0]), :mod:`core.relief.mesh` meshes one closed box-union per
filament, :mod:`core.relief.export3mf` writes the X2D 3MF with the existing
writers and :mod:`core.relief.validate` re-reads it.  Entry point:
:func:`core.relief.pipeline.convert_album_relief` / scripts/convert_relief.py.

Art direction (optional, geometry only): :mod:`core.relief.heightfield` edits a
float32 mm height map inside object masks (dome / extrude / raise-lower /
feather / reset), :mod:`core.relief.art_direct` records them as a replayable
script and applies it in the pipeline (``height_edits=``), :mod:`core.relief.segment`
provides SAM click-to-select + brush masks and :mod:`core.relief.object_provider`
is the stub interface for future AI object-geometry providers.

Nothing in core/stack5, core/band, core/converter.py or config.py is modified.
"""
from core.relief.depth_processing import DepthParams, ReliefField, process_depth
from core.relief.provider import (PROVIDER_NAMES, GeometryError, GeometryProvider, GeometryResult,
                                  MissingDependencyError, get_provider)
from core.relief.relief_stack import ReliefDims, build_relief_voxels, check_relief_voxels

__all__ = ['DepthParams', 'ReliefField', 'process_depth', 'PROVIDER_NAMES', 'GeometryError', 'GeometryProvider',
           'GeometryResult', 'MissingDependencyError', 'get_provider', 'ReliefDims', 'build_relief_voxels',
           'check_relief_voxels', 'convert_album_relief', 'EditScript', 'EditSession', 'apply_edit_script',
           'relief_for_editor']


def __getattr__(name):
    if name == 'convert_album_relief':          # pulls in core.converter / gradio: import lazily
        from core.relief.pipeline import convert_album_relief
        return convert_album_relief
    if name in ('EditScript', 'EditSession', 'apply_edit_script', 'relief_for_editor'):
        import core.relief.art_direct as _ad
        return getattr(_ad, name)
    raise AttributeError(name)
