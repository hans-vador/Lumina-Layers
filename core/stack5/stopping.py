"""Match independent, supported face-up columns from zero to five layers.

Every pixel may choose its own materials AND stopping height. Candidate columns
contain air only above their exposed surface. Simpler recipes win when their
predicted colour distance is within one metric unit of the best candidate.
"""
from __future__ import annotations
import numpy as np
from core.band.optics import srgb_to_lab_d65
from core.stack5.lut import synth_lut
from core.stack5.metric import hue_dist2, weighted_lab_dist2
from core.stack5.cleanup import min_region_cleanup


def variable_stack_candidates(filaments, backing_slot, model, layer_h, layers):
    stacks = [np.full((1,layers),-1,np.int32)]
    colours = [np.asarray([filaments[backing_slot].rgb8],np.uint8)]
    for depth in range(1,layers+1):
        rgb, recipes = synth_lut(filaments,backing_slot,model,layers=depth,layer_h=layer_h)
        padded = np.full((len(recipes),layers),-1,np.int32)
        padded[:,-depth:] = recipes
        stacks.append(padded); colours.append(rgb)
    stacks, colours = np.concatenate(stacks), np.concatenate(colours)
    occupied = stacks >= 0
    depths = occupied.sum(axis=1)
    changes = ((stacks[:,1:] != stacks[:,:-1]) & occupied[:,1:] & occupied[:,:-1]).sum(axis=1)
    distinct = np.array([len(set(row[row>=0])) for row in stacks])
    # Global order makes tie-breaking deterministic: fewer changes, then fewer
    # materials, then fewer layers. Preserve enumeration order for exact ties.
    order = np.lexsort((np.arange(len(stacks)),depths,distinct,changes))
    return stacks[order],colours[order],depths[order],changes[order]


def stop_at_best_match(result, filaments, backing_slot, model, layer_h,
                       metric='lumina', wL=1.0, hue_params=None,
                       min_region_px=0, simplicity_tolerance=1.0):
    import cv2
    if not np.isfinite(simplicity_tolerance) or simplicity_tolerance < 0:
        raise ValueError('simplicity tolerance must be finite and nonnegative')
    mm = np.asarray(result['material_matrix'])
    mask = np.asarray(result['mask_solid'], bool)
    original = mm[mask]
    layers = mm.shape[-1]
    if not len(original):
        raise ValueError('early stopping requires at least one solid pixel')
    if np.any(original < 0):
        raise ValueError('early stopping expects complete input colour stacks')
    stacks,colours,depths,changes = variable_stack_candidates(filaments,backing_slot,model,layer_h,layers)
    targets = np.asarray(result['quantized_image'], dtype=np.uint8)[mask]
    unique_targets, target_ids = np.unique(targets, axis=0, return_inverse=True)
    if metric == 'lumina':
        candidate_lab = cv2.cvtColor(colours[:,None,:],cv2.COLOR_RGB2LAB)[:,0].astype(float)
        target_lab = cv2.cvtColor(unique_targets[:,None,:],cv2.COLOR_RGB2LAB)[:,0].astype(float)
        costs = np.sum((candidate_lab[:,None,:]-target_lab[None,:,:])**2,axis=-1)
    else:
        candidate_lab = srgb_to_lab_d65(colours / 255.)
        target_lab = srgb_to_lab_d65(unique_targets / 255.)
        if metric == 'hue':
            costs = hue_dist2(target_lab,candidate_lab,hue_params)
        elif metric == 'lab':
            costs = weighted_lab_dist2(target_lab,candidate_lab,wL)
        else:
            raise ValueError(f'unknown stopping metric: {metric}')
    best = costs.min(axis=0)
    eligible = costs <= (np.sqrt(best)+simplicity_tolerance)**2 + 1e-9
    chosen = eligible.argmax(axis=0) # first in simplicity order
    selected = chosen[target_ids]
    protected = np.asarray(result.get('protected_edges', np.zeros(mask.shape, bool)), bool)
    if protected.any():
        # Brightness separation matters at outlines even when hue matching
        # elsewhere is willing to trade lightness for colour-family accuracy.
        edge_costs = weighted_lab_dist2(srgb_to_lab_d65(unique_targets / 255.),
                                       srgb_to_lab_d65(colours / 255.), 2.0)
        edge_ids = edge_costs.argmin(axis=0)
        edge_pixels = protected[mask]
        selected[edge_pixels] = edge_ids[target_ids[edge_pixels]]
    id_map = np.full(mask.shape,-1,np.int32); id_map[mask] = selected
    cleanup = None
    if min_region_px > 1:
        # Copy WHOLE recipes, never independent layers: no floating islands or
        # resuming extrusion above a stopped column. Preview follows cleanup.
        clean,cleanup = min_region_cleanup(id_map[...,None],mask,min_region_px, protected_mask=protected)
        selected = clean[...,0][mask]
    lookup = {tuple(row):i for i,row in enumerate(stacks)}
    unique_original, inv = np.unique(original,axis=0,return_inverse=True)
    baseline_ids = np.array([lookup[tuple(row)] for row in unique_original])[inv]
    before = costs[baseline_ids,target_ids]
    after = costs[selected,target_ids]
    count = depths[selected]
    out = mm.copy(); out[mask] = stacks[selected]
    preview = np.asarray(result['matched_rgb']).copy(); preview[mask] = colours[selected]
    height = np.zeros(mask.shape,dtype=np.uint8); height[mask] = count
    report = {
        'enabled': True, 'method': 'independent variable-depth stack search',
        'candidate_count': len(stacks), 'metric': metric,
        'protected_edge_pixels': int(np.count_nonzero(protected & mask)),
        'simplicity_tolerance': float(simplicity_tolerance),
        'tie_break': 'fewest material changes, fewest materials, fewest layers',
        'pixels_by_colour_layer_count': np.bincount(count,minlength=layers+1).tolist(),
        'optical_voxels_before': int(len(original)*layers),
        'optical_voxels_after': int(count.sum()),
        'optical_material_saved_fraction': float(1-count.mean()/layers),
        'mean_match_cost_before': float(before.mean()), 'mean_match_cost_after': float(after.mean()),
        'max_match_cost_increase': float(np.max(after-before)),
        'max_colour_layers': int(count.max()),
        'single_material_column_share': float(np.mean((changes[selected]==0)&(count>0))),
        'blended_column_share': float(np.mean(changes[selected]>0)),
        'whole_recipe_cleanup': cleanup,
    }
    return out, preview, height, report
