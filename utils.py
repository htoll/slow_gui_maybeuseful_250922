import sif_parser
import numpy as np
import pandas as pd
from skimage.feature import blob_log

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize

import seaborn as sns

from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

from datetime import date

import streamlit as st
import io
import re
import os
import textwrap

import plotly.express as px
import plotly.graph_objects as go


_PEAK_FOOTPRINT = np.ones((3, 3), dtype=int)
_INTERP_GRID_CACHE = {}
_SKIP_RESULT = object()


def _sklearn_peak_local_max(
    image,
    *,
    min_distance=1,
    threshold_abs=None,
    num_peaks=None,
    exclude_border=True,
    footprint=None,
):
    """Lightweight peak finder backed by scikit-learn's KDTree."""

    arr = np.asarray(image, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, arr.ndim), dtype=int)

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.empty((0, arr.ndim), dtype=int)

    finite_vals = arr[finite_mask]
    if threshold_abs is None:
        threshold_abs_val = float(finite_vals.min())
    else:
        threshold_abs_val = float(threshold_abs)

    candidate_mask = (arr > threshold_abs_val) & finite_mask
    max_peaks = None
    if num_peaks is not None and not np.isinf(num_peaks):
        max_peaks = max(0, int(num_peaks))
        if max_peaks == 0:
            return np.empty((0, arr.ndim), dtype=int)

    if not candidate_mask.any():
        if max_peaks is not None and max_peaks > 0:
            flat_idx = int(np.nanargmax(arr))
            coord = np.array(np.unravel_index(flat_idx, arr.shape), dtype=int)
            return coord.reshape(1, -1)
        return np.empty((0, arr.ndim), dtype=int)

    coords = np.argwhere(candidate_mask)
    intensities = arr[candidate_mask]

    if coords.shape[0] == 0:
        return np.empty((0, arr.ndim), dtype=int)

    if exclude_border:
        border = int(np.ceil(min_distance))
        if border > 0:
            valid_mask = np.ones(coords.shape[0], dtype=bool)
            for axis, size in enumerate(arr.shape):
                valid_mask &= coords[:, axis] >= border
                valid_mask &= coords[:, axis] < size - border
            coords = coords[valid_mask]
            intensities = intensities[valid_mask]
            if coords.shape[0] == 0:
                if max_peaks is not None and max_peaks > 0:
                    flat_idx = int(np.nanargmax(arr))
                    coord = np.array(np.unravel_index(flat_idx, arr.shape), dtype=int)
                    return coord.reshape(1, -1)
                return np.empty((0, arr.ndim), dtype=int)

    order = np.argsort(intensities)[::-1]
    coords = coords[order]
    intensities = intensities[order]

    if coords.shape[0] == 0:
        return np.empty((0, arr.ndim), dtype=int)

    radius = float(max(min_distance, 0.0))
    if footprint is not None:
        footprint = np.asarray(footprint)
        if footprint.size > 0:
            offsets = [(dim - 1) / 2.0 for dim in footprint.shape]
            radius = max([radius, *offsets])

    if radius <= 0.0:
        if max_peaks is None:
            return coords.copy()
        return coords[:max_peaks].copy()

    coords_float = coords.astype(np.float64, copy=False)
    tree = KDTree(coords_float, leaf_size=16, metric="minkowski", p=np.inf)
    suppressed = np.zeros(coords.shape[0], dtype=bool)
    accepted = []

    for idx, point in enumerate(coords_float):
        if suppressed[idx]:
            continue
        accepted.append(coords[idx])
        if max_peaks is not None and len(accepted) >= max_peaks:
            break
        neighbor_idx = tree.query_radius(point.reshape(1, -1), r=radius, return_distance=False)[0]
        suppressed[neighbor_idx] = True

    return np.asarray(accepted, dtype=int)


def _estimate_sigma_from_moment(sub_img, pix_size_um, baseline, min_sigma, max_sigma):
    """Estimate Gaussian sigma along x/y via intensity-weighted second moments."""
    if sub_img.size == 0:
        return (min_sigma, min_sigma)

    # Subtract a baseline so that background fluctuations are de-emphasised.
    weights = np.asarray(sub_img, dtype=np.float64) - float(baseline)
    if not np.isfinite(weights).all():
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        fallback = max(min_sigma, min(max_sigma, 0.5 * (min_sigma + max_sigma)))
        return (fallback, fallback)

    y_idx, x_idx = np.indices(sub_img.shape, dtype=np.float64)
    mean_x = float((weights * x_idx).sum() / total)
    mean_y = float((weights * y_idx).sum() / total)

    var_x = float((weights * (x_idx - mean_x) ** 2).sum() / total)
    var_y = float((weights * (y_idx - mean_y) ** 2).sum() / total)

    var_x = max(var_x, 0.0)
    var_y = max(var_y, 0.0)

    sigma_x = np.sqrt(var_x) * pix_size_um
    sigma_y = np.sqrt(var_y) * pix_size_um

    sigma_x = float(np.clip(sigma_x, min_sigma, max_sigma))
    sigma_y = float(np.clip(sigma_y, min_sigma, max_sigma))

    return (sigma_x, sigma_y)


def _gaussian_residuals(params, x, y, z):
    A, x0, sx, y0, sy, offset = params
    model = A * np.exp(-((x - x0) ** 2 / (2 * sx ** 2) + (y - y0) ** 2 / (2 * sy ** 2))) + offset
    return model - z


def _get_interp_grids(orig_shape, interp_shape):
    key = (orig_shape[0], orig_shape[1], interp_shape[0], interp_shape[1])
    cached = _INTERP_GRID_CACHE.get(key)
    if cached is None:
        H, W = interp_shape
        if W > 1:
            x_lin = np.linspace(0, orig_shape[1] - 1, W, dtype=np.float64)
        else:
            x_lin = np.zeros(W, dtype=np.float64)
        if H > 1:
            y_lin = np.linspace(0, orig_shape[0] - 1, H, dtype=np.float64)
        else:
            y_lin = np.zeros(H, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(x_lin, y_lin)
        cached = (grid_x.ravel(), grid_y.ravel())
        _INTERP_GRID_CACHE[key] = cached
    return cached


def HWT_aesthetic():
    sns.set_style("ticks")
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5,
                        "axes.labelsize": 14,
                        "axes.titlesize": 16})
    palette = sns.color_palette("colorblind")  # pref'd are colorblind, tab20c, muted6
    sns.set_palette(palette)
    sns.despine()
    return palette 

def integrate_sif(
    sif,
    threshold=1,
    region='all',
    signal='UCNP',
    pix_size_um=0.1,
    sig_threshold=0.25,
    min_sigma_um = 0.15,
    *,
    min_fit_separation_px=3,
    min_r2 = 0.85
):
    image_data, metadata = sif_parser.np_open(sif)
    image_data = image_data[0]  # (H, W)

    gainDAC = metadata['GainDAC']
    exposure_time = metadata['ExposureTime']
    accumulate_cycles = metadata['AccumulatedCycles']

    # Normalize counts → photons
    image_data_cps = image_data * (5.0 / gainDAC) / exposure_time / accumulate_cycles

    radius_um_fine = 0.3
    radius_pix_fine = int(radius_um_fine / pix_size_um)

    # --- Crop image if region specified ---
    region = str(region)
    if region == '3':
        image_data_cps = image_data_cps[0:256, 0:256]
    elif region == '4':
        image_data_cps = image_data_cps[0:256, 256:512]
    elif region == '1':
        image_data_cps = image_data_cps[256:512, 0:256]
    elif region == '2':
        image_data_cps = image_data_cps[256:512, 256:512]
    elif region == 'custom': #accounting for misaligned 638 beam on 250610
        image_data_cps = image_data_cps[312:512, 56:256]

    # else → 'all': use full image

    # --- Detect peaks ---
    smoothed_image = gaussian_filter(image_data_cps, sigma=1)
    threshold_abs = np.mean(smoothed_image) + threshold * np.std(smoothed_image)

    if signal == 'UCNP':
        coords = peak_local_max(smoothed_image, min_distance=5, threshold_abs=threshold_abs)
    else:
        blobs = blob_log(smoothed_image, min_sigma=1, max_sigma=3, num_sigma=5, threshold=5 * threshold)
        coords = blobs[:, :2]

    #print(f"{os.path.basename(sif)}: Found {len(coords)} peaks in region {region}")

    results = []
    for center_y, center_x in coords:
        # Extract subregion
        sub_img, x0_idx, y0_idx = extract_subregion(image_data_cps, center_x, center_y, radius_pix_fine)

        # Refine peak
        blurred = gaussian_filter(sub_img, sigma=1)
        local_peak = peak_local_max(blurred, num_peaks=1)
        if local_peak.shape[0] == 0:
            continue
        local_y, local_x = local_peak[0]
        center_x_refined = x0_idx + local_x
        center_y_refined = y0_idx + local_y

        # Extract finer subregion
        sub_img_fine, x0_idx_fine, y0_idx_fine = extract_subregion(
            image_data_cps, center_x_refined, center_y_refined, radius_pix_fine
        )
        # Interpolate to 20x20 grid (like MATLAB)
        interp_size = 20
        zoom_factor = interp_size / sub_img_fine.shape[0]
        sub_img_interp = zoom(sub_img_fine, zoom_factor, order=1)  # bilinear interpolation

        # Prepare grid
        # y_indices, x_indices = np.indices(sub_img_fine.shape)
        # x_coords = (x_indices + x0_idx_fine) * pix_size_um
        # y_coords = (y_indices + y0_idx_fine) * pix_size_um
        interp_shape = sub_img_interp.shape
        y_indices, x_indices = np.indices(interp_shape)
        x_coords = (x_indices / interp_shape[1] * sub_img_fine.shape[1] + x0_idx_fine) * pix_size_um
        y_coords = (y_indices / interp_shape[0] * sub_img_fine.shape[0] + y0_idx_fine) * pix_size_um

        x_flat = x_coords.ravel()
        y_flat = y_coords.ravel()
        z_flat = sub_img_interp.ravel() #∆ variable name 250604

        # Initial guess
        amp_guess = np.max(sub_img_fine)
        offset_guess = np.min(sub_img_fine)
        x0_guess = center_x_refined * pix_size_um
        y0_guess = center_y_refined * pix_size_um
        sigma_guess = 0.15
        p0 = [amp_guess, x0_guess, sigma_guess, y0_guess, sigma_guess, offset_guess]

        # Fit
        try:
            #popt, _ = curve_fit(gaussian2d, (x_flat, y_flat), z_flat, p0=p0)
            def residuals(params, x, y, z):
                A, x0, sx, y0, sy, offset = params
                model = A * np.exp(-((x - x0)**2 / (2 * sx**2) + (y - y0)**2 / (2 * sy**2))) + offset
                return model - z

            # Set bounds similar to MATLAB
            lb = [1, x0_guess - 1, 0.0, y0_guess - 1, 0.0, offset_guess * 0.5]
            ub = [2 * amp_guess, x0_guess + 1, 0.175, y0_guess + 1, 0.175, offset_guess * 1.2]

            # Perform fit
            res = least_squares(residuals, p0, args=(x_flat, y_flat, z_flat), bounds=(lb, ub))
            popt = res.x
            amp_fit, x0_fit, sigx_fit, y0_fit, sigy_fit, offset_fit = popt
            brightness_fit = 2 * np.pi * amp_fit * sigx_fit * sigy_fit / pix_size_um**2
            brightness_integrated = np.sum(sub_img_fine) - sub_img_fine.size * offset_fit

            if brightness_fit > 1e9 or brightness_fit < 10:
                print(f"Excluded peak for brightness {brightness_fit:.2e}")
                continue
            if sigx_fit > sig_threshold or sigy_fit > sig_threshold:
                print(f"Excluded peak for size {sigx_fit:.2f} um x {sigy_fit:.2f} um")
                continue

            # Note: coordinates are already RELATIVE to cropped image
            results.append({
                'x_pix': center_x_refined,
                'y_pix': center_y_refined,
                'x_um': x0_fit,
                'y_um': y0_fit,
                'amp_fit': amp_fit,
                'sigx_fit': sigx_fit,
                'sigy_fit': sigy_fit,
                'brightness_fit': brightness_fit,
                'brightness_integrated': brightness_integrated
            })

        except RuntimeError:
            continue

    df = pd.DataFrame(results)
    return df, image_data_cps
    # """Detect peaks in a SIF image and fit local 2D Gaussian PSFs.

    # Parameters
    # ----------
    # min_fit_separation_px : float, optional
    #     Minimum centre-to-centre distance (in pixels) required between accepted
    #     fits. Candidates closer than this distance to an existing result are
    #     skipped. Default is 3 pixels. Pass ``0`` or ``None`` to disable the
    #     separation filter.

    # min_r2 : float or None, optional
    #     Minimum coefficient of determination (R²) required for a fitted PSF to
    #     be accepted. Set to ``None`` to disable the residual-based quality
    #     filter. Defaults to 0.85.
        
    # min_sigma_um : float, optional
    #     Minimum allowed Gaussian sigma (in microns). Fits smaller than this
    #     threshold are discarded. Defaults to 0.15 µm (≈150 nm).
    # """
    # image_data, metadata = sif_parser.np_open(sif, ignore_corrupt=True)
    # image_data = image_data[0]  # (H, W)

    # gainDAC = metadata['GainDAC']
    # if gainDAC == 0:
    #     gainDAC =1 #account for gain turned off
    # exposure_time = metadata['ExposureTime']
    # accumulate_cycles = metadata['AccumulatedCycles']

    # # Normalize counts → photons
    # image_data_cps = image_data * (5.0 / gainDAC) / exposure_time / accumulate_cycles

    # radius_um_fine = 0.5
    # radius_pix_fine = int(radius_um_fine / pix_size_um)
    # pix_size_um_sq = pix_size_um ** 2

    # # --- Crop image if region specified ---
    # region = str(region)
    # if region == '3':
    #     image_data_cps = image_data_cps[0:256, 0:256]
    # elif region == '4':
    #     image_data_cps = image_data_cps[0:256, 256:512]
    # elif region == '1':
    #     image_data_cps = image_data_cps[256:512, 0:256]
    # elif region == '2':
    #     image_data_cps = image_data_cps[256:512, 256:512]
    # elif region == 'custom': #accounting for misaligned 638 beam on 250610
    #     image_data_cps = image_data_cps[312:512, 56:256]

    # # else → 'all': use full image

    # # --- Detect peaks ---
    # smoothed_image = gaussian_filter(image_data_cps, sigma=1)
    # threshold_abs = np.mean(smoothed_image) + threshold * np.std(smoothed_image)


    # # mean_val = float(np.mean(smoothed_image))
    # # std_val = float(np.std(smoothed_image))
    # # std_val = max(std_val, 1e-6)
    # # noise_floor = std_val
    # # threshold_abs = mean_val + threshold * std_val

    # min_distance = 5


    # if signal == 'UCNP':
    #     coords = peak_local_max(smoothed_image, min_distance=5, threshold_abs=threshold_abs)
    # else:
    #     blobs = blob_log(smoothed_image, min_sigma=1, max_sigma=3, num_sigma=5, threshold=5 * threshold)
    #     coords = blobs[:, :2]

    # # #print(f"{os.path.basename(sif)}: Found {len(coords)} peaks in region {region}")
    # # coords = [tuple(c) for c in coords]  # list of (y, x)
    # # seen = set()  # to avoid infinite loops on duplicates

    # # i = 0
    # # while i < len(coords):
    # #     center_y, center_x = coords[i]
    # #     i += 1
    # #     key = (int(center_y), int(center_x))
    # #     if key in seen:
    # #         continue
    # #     seen.add(key)
    
    # #     # Extract subregion
    # #     sub_img, x0_idx, y0_idx = extract_subregion(image_data_cps, center_x, center_y, radius_pix_fine)
    
    # #     # Quick local check for multiple peaks in this subwindow
    # #     sub_blur = gaussian_filter(sub_img, sigma=0.5)
    # #     # relative threshold: 30% of local max avoids noise, keeps siblings
    # #     loc_thr = sub_blur.max() * 0.3
    # #     local_peaks = _sklearn_peak_local_max(
    # #         sub_blur,
    # #         min_distance=2,
    # #         threshold_abs=loc_thr,
    # #         exclude_border=False,
    # #     )

    # #     if local_peaks.shape[0] > 1:
    # #         # enqueue each local maximum as its own candidate (translated to image coords)
    # #         for ly, lx in local_peaks:
    # #             ny = y0_idx + ly
    # #             nx = x0_idx + lx
    # #             # skip if extremely close to any already-known coord
    # #             if all((abs(ny - cy) > 1 or abs(nx - cx) > 1) for cy, cx in coords):
    # #                 coords.append((ny, nx))
    # #     continue  # don't fit the ambiguous merged center; handle the new ones


    # # results = []
    # # fit_cache = {}
    # for center_y, center_x in coords:
    #     # Extract subregion
    #     sub_img, x0_idx, y0_idx = extract_subregion(image_data_cps, center_x, center_y, radius_pix_fine)

    #     # Refine peak
    #     blurred = gaussian_filter(sub_img, sigma=1)
    #     local_peak = peak_local_max(blurred, num_peaks=1)
    #     if local_peak.shape[0] == 0:
    #         continue
    #     local_y, local_x = local_peak[0]
    #     center_x_refined = x0_idx + local_x
    #     center_y_refined = y0_idx + local_y

    #     # cache_key = (int(center_y_refined), int(center_x_refined))
    #     # cached_result = fit_cache.get(cache_key)
    #     # if cached_result is _SKIP_RESULT:
    #     #     continue
    #     # if cached_result is not None:
    #     #     # A prior candidate already fit this PSF; skip duplicating it.
    #     #     continue

    #     # Extract finer subregion
    #     sub_img_fine, x0_idx_fine, y0_idx_fine = extract_subregion(
    #         image_data_cps, center_x_refined, center_y_refined, radius_pix_fine
    #     )
    #     local_peak_value = float(np.max(sub_img_fine)) if sub_img_fine.size else 0.0

    #     replace_indices = None
    #     replace_cache_keys = None
    #     insert_pos = None
    #     if min_fit_separation_px is not None and min_fit_separation_px > 0:
    #         min_sep_sq = float(min_fit_separation_px) ** 2
    #         # Collect every existing fit within the exclusion radius so we can
    #         # either veto the new candidate (if an existing one is stronger) or
    #         # evict all nearby weaker fits once the new fit succeeds.
    #         conflicts = []
    #         for idx, existing in enumerate(results):
    #             dx = float(existing.get('x_pix', 0.0)) - center_x_refined
    #             dy = float(existing.get('y_pix', 0.0)) - center_y_refined
    #             if dx * dx + dy * dy <= min_sep_sq:
    #                 existing_peak = float(existing.get('local_peak_value', existing.get('brightness_integrated', 0.0)))
    #                 conflicts.append((idx, existing_peak, existing))
    #         if conflicts:
    #             # Skip the candidate entirely when any nearby fit already has a
    #             # higher peak height. Otherwise mark every conflicting fit for
    #             # replacement so we don't keep a ring of weaker duplicates.
    #             strongest_peak = max(conflicts, key=lambda item: item[1])[1]
    #             if strongest_peak >= local_peak_value:
    #                 fit_cache[cache_key] = _SKIP_RESULT
    #                 continue
    #             replace_indices = sorted({idx for idx, _, _ in conflicts})
    #             replace_cache_keys = list(dict.fromkeys([
    #                 (int(existing.get('y_pix', 0)), int(existing.get('x_pix', 0)))
    #                 for idx, _, existing in conflicts
    #             ]))
    #             insert_pos = replace_indices[0]
    #     # # Interpolate to 20x20 grid (like MATLAB)
    #     # interp_size = 20
    #     # zoom_factor = interp_size / sub_img_fine.shape[0]
    #     # sub_img_interp = zoom(sub_img_fine, zoom_factor, order=1)  # bilinear interpolation

    #     # # Prepare grid
    #     # # y_indices, x_indices = np.indices(sub_img_fine.shape)
    #     # # x_coords = (x_indices + x0_idx_fine) * pix_size_um
    #     # # y_coords = (y_indices + y0_idx_fine) * pix_size_um
    #     # interp_shape = sub_img_interp.shape
    #     # y_indices, x_indices = np.indices(interp_shape)
    #     # x_coords = (x_indices / interp_shape[1] * sub_img_fine.shape[1] + x0_idx_fine) * pix_size_um
    #     # y_coords = (y_indices / interp_shape[0] * sub_img_fine.shape[0] + y0_idx_fine) * pix_size_um

    #     # x_flat = x_coords.ravel()
    #     # y_flat = y_coords.ravel()
    #     # z_flat = sub_img_interp.ravel() #∆ variable name 250604
    #     # switch from interp to fitting pixels 250916, noted that we were understimating brightness
    #     interp_size = 20
    #     zoom_factor = interp_size / sub_img_fine.shape[0]
    #     sub_img_interp = zoom(sub_img_fine, zoom_factor, order=1)
    #     base_x_flat, base_y_flat = _get_interp_grids(sub_img_fine.shape, sub_img_interp.shape)
    #     x_flat = (base_x_flat + x0_idx_fine) * pix_size_um
    #     y_flat = (base_y_flat + y0_idx_fine) * pix_size_um
    #     z_flat = sub_img_interp.ravel()

    #     # Initial guess
    #     amp_guess = np.max(sub_img_fine)
    #     offset_guess = np.min(sub_img_fine)
    #     local_baseline = float(np.median(sub_img_fine))
    #     local_mad = float(np.median(np.abs(sub_img_fine - local_baseline)))
    #     if local_mad > 0:
    #         local_noise = local_mad / 0.6745
    #     else:
    #         local_noise = noise_floor
    #     local_noise = max(local_noise, noise_floor)
    #     if amp_guess - local_baseline < 1.5 * local_noise:
    #         fit_cache[cache_key] = _SKIP_RESULT
    #         continue
    #     x0_guess = center_x_refined * pix_size_um
    #     y0_guess = center_y_refined * pix_size_um
    #     sigma_min = max(float(min_sigma_um), 0.0)
    #     sigma_max = max(float(sig_threshold), sigma_min)
    #     sigma_ub = sigma_max  # bounds expressed in microns to match coordinate scaling
    #     baseline_for_sigma = min(local_baseline + local_noise, amp_guess)
    #     sigma_guess_x, sigma_guess_y = _estimate_sigma_from_moment(
    #         sub_img_fine,
    #         pix_size_um,
    #         baseline_for_sigma,
    #         sigma_min,
    #         sigma_max,
    #     )
    #     p0 = [amp_guess, x0_guess, sigma_guess_x, y0_guess, sigma_guess_y, offset_guess]

    #     # Fit
    #     try:

    #         lb = [1, x0_guess - 1, sigma_min, y0_guess - 1, sigma_min, 0.0]
    #         ub = [2 * amp_guess, x0_guess + 1, sigma_ub, y0_guess + 1, sigma_ub, offset_guess * 1.2]

    #         # Perform fit
    #         res = least_squares(_gaussian_residuals, p0, args=(x_flat, y_flat, z_flat), bounds=(lb, ub))
    #         popt = res.x
    #         amp_fit, x0_fit, sigx_fit, y0_fit, sigy_fit, offset_fit = popt
            
    #         # Evaluate residual quality of the fitted Gaussian on the
    #         # interpolated sub-image to filter out noisy / poorly modelled PSFs.
    #         exp_term = -(
    #             (x_flat - x0_fit) ** 2 / (2 * sigx_fit ** 2)
    #             + (y_flat - y0_fit) ** 2 / (2 * sigy_fit ** 2)
    #         )
    #         model_flat = amp_fit * np.exp(exp_term) + offset_fit
    #         residual_flat = z_flat - model_flat
    #         ss_res = float(np.sum(residual_flat ** 2))
    #         ss_tot = float(np.sum((z_flat - z_flat.mean()) ** 2))
    #         if ss_tot <= np.finfo(float).eps:
    #             fit_r2 = 1.0 if ss_res <= np.finfo(float).eps else -np.inf
    #         else:
    #             fit_r2 = 1.0 - ss_res / ss_tot
    #         if min_r2 is not None and fit_r2 < min_r2:
    #             fit_cache[cache_key] = _SKIP_RESULT
    #             continue
    #         brightness_fit = 2 * np.pi * amp_fit * sigx_fit * sigy_fit / pix_size_um_sq
    #         roi_min = np.min(sub_img_fine)
    #         brightness_integrated = np.sum(sub_img_fine) - sub_img_fine.size * roi_min
    #         #brightness_integrated = np.sum(sub_img_fine) - sub_img_fine.size * offset_fit

    #         if brightness_integrated > 1e9 or brightness_integrated < 1:
    #             print(f"Excluded peak for brightness {brightness_integrated:.2e}")
    #             fit_cache[cache_key] = _SKIP_RESULT
    #             continue
    #         EPS = 1e-3
    #         if sigx_fit > sig_threshold + EPS or sigy_fit > sig_threshold + EPS:
    #             fit_cache[cache_key] = _SKIP_RESULT
    #             continue
    #         if sigx_fit < sigma_min - EPS or sigy_fit < sigma_min - EPS:
    #             fit_cache[cache_key] = _SKIP_RESULT
    #             continue

    #         # Note: coordinates are already RELATIVE to cropped image
    #         result = {
    #             'x_pix': center_x_refined,
    #             'y_pix': center_y_refined,
    #             'x_um': x0_fit,
    #             'y_um': y0_fit,
    #             'amp_fit': amp_fit,
    #             'sigx_fit': sigx_fit,
    #             'sigy_fit': sigy_fit,
    #             'brightness_fit': brightness_fit,
    #             'brightness_integrated': brightness_integrated,
    #             'local_peak_value': local_peak_value,
    #             'fit_r2': fit_r2,

    #         }
    #         if replace_indices:
    #             for key in replace_cache_keys:
    #                 if key != cache_key:
    #                     fit_cache.pop(key, None)
    #             for idx in sorted(replace_indices, reverse=True):
    #                 if 0 <= idx < len(results):
    #                     results.pop(idx)
    #             if insert_pos is None or insert_pos > len(results):
    #                 results.append(result)
    #             else:
    #                 results.insert(insert_pos, result)
    #         else:
    #             results.append(result)
    #         fit_cache[cache_key] = result.copy()

    #     except RuntimeError:
    #         fit_cache[cache_key] = _SKIP_RESULT
    #         continue

    # df = pd.DataFrame(results)
    # return df, image_data_cps
    
def gaussian(x, amp, mu, sigma):
  return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))



def plot_brightness(
    image_data_cps,
    df,
    show_fits=True,
    plot_brightness_histogram=False,   
    normalization=False,
    pix_size_um=0.1,
    cmap='magma',
    *,
    interactive=False,                
    dragmode='zoom'                    # 'zoom' (magnifier) or 'pan'
):
    """
    If interactive=False (default): returns a Matplotlib Figure (original behavior).
    If interactive=True: returns a Plotly Figure with box-zoom/pan and crisp vector overlays.
    """
    # --- MATPLOTLIB PATH (UNCHANGED DEFAULT) ---
    if not interactive:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.patches import Circle

        fig_width, fig_height = 5, 5
        scale = fig_width / 5
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        norm = LogNorm() if normalization else None
        im = ax.imshow(image_data_cps + 1, cmap=cmap, norm=norm, origin='lower')
        ax.tick_params(axis='both', length=0, labelleft=False, labelright=False,
                       labeltop=False, labelbottom=False)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10*scale)
        cbar.set_label('pps', fontsize=10*scale)

        if show_fits and df is not None and not df.empty:
            for _, row in df.iterrows():
                x_px = row['x_pix']
                y_px = row['y_pix']
                brightness_kpps = row['brightness_integrated'] / 1000.0
                radius_px = 2 * max(row['sigx_fit'], row['sigy_fit']) / pix_size_um

                circle = Circle((x_px, y_px), radius_px, color='white',
                                fill=False, linewidth=1*scale, alpha=0.7)
                ax.add_patch(circle)

                ax.text(x_px + 7.5, y_px + 7.5, f"{brightness_kpps:.1f} kpps",
                        color='white', fontsize=7*scale, ha='center', va='center')

        plt.tight_layout()
        HWT_aesthetic()
        return fig

    # --- PLOTLY PATH (INTERACTIVE) ---
    import numpy as np

    # Map Matplotlib colormap names to Plotly scales
    cmap_map = {
        "magma": "Magma", "viridis": "Viridis", "plasma": "Plasma",
        "hot": "Hot", "gray": "Gray", "hsv": "HSV", "cividis": "Cividis", "inferno": "Inferno"
    }
    plotly_scale = cmap_map.get(cmap, "Magma")

    # Approximate LogNorm for image display if requested
    img = image_data_cps.astype(float)
    if normalization:
        eps = max(float(np.percentile(img, 0.01)), 1e-9)
        img_display = np.log10(np.clip(img + 1.0, eps, None))
    else:
        img_display = img

    # Base image with stored cps for hover
    fig = px.imshow(
        img_display,
        origin="lower",
        aspect="equal",
        color_continuous_scale=plotly_scale
    )
    # store original cps values for accurate hover information
    img_custom = np.expand_dims(img, axis=-1)
    fig.data[0].customdata = img_custom
    fig.data[0].hovertemplate = (
        "x=%{x:.0f}px<br>y=%{y:.0f}px<br>pps=%{customdata[0]:.1f}<extra></extra>"
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        dragmode=dragmode,
        coloraxis_colorbar=dict(
            title="pps" if not normalization else "log10(pps)",
            yanchor="middle",
            y=0.5,
            lenmode="fraction",
            len=0.8,
            thickness=20,
        ),
        xaxis_title="X (px)",
        yaxis_title="Y (px)"
    )

    xs = ys = rs = br = None
    if df is not None and not df.empty:
        xs = df["x_pix"].to_numpy()
        ys = df["y_pix"].to_numpy()
        rs = (2 * np.maximum(df["sigx_fit"].to_numpy(), df["sigy_fit"].to_numpy()) / pix_size_um).astype(float)
        br = (df["brightness_integrated"].to_numpy() / 1000.0).astype(float)

        custom = np.stack([br], axis=1)
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=1, opacity=0),
            name="Fits",
            customdata=custom,
            hovertemplate="x=%{x:.2f}px<br>y=%{y:.2f}px<br>brightness=%{customdata[0]:.1f} kpps<extra></extra>",
            showlegend=False,
        ))

        if show_fits:
            shapes = []
            for x, y, r in zip(xs, ys, rs):
                shapes.append(dict(
                    type="circle",
                    xref="x", yref="y",
                    x0=x - r, x1=x + r,
                    y0=y - r, y1=y + r,
                    line=dict(width=1.5, color="white"),
                    fillcolor="rgba(0,0,0,0)",
                    layer="above",
                ))
            fig.update_layout(shapes=shapes)

            fig.add_trace(go.Scatter(
                x=xs + 7.5, y=ys + 7.5, mode="text",
                text=[f"{v:.1f} kpps" for v in br],
                textfont=dict(color="white", size=10),
                textposition="middle center",
                showlegend=False,
                hoverinfo="skip",
            ))
# Keep pixel coordinates aligned and square pixels
    h, w = img.shape
    fig.update_xaxes(range=[-0.5, w - 0.5], constrain="domain", showgrid=False, zeroline=False)
    fig.update_yaxes(range=[-0.5, h - 0.5], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False)

    return fig



def plot_histogram(df, min_val=None, max_val=None, num_bins=20, thresholds=None):
    """
    Plots the brightness histogram with a Gaussian fit and optional vertical thresholds.
    
    Args:
        df (pd.DataFrame): DataFrame containing brightness data.
        min_val (float, optional): Minimum brightness value for the histogram.
        max_val (float, optional): Maximum brightness value for the histogram.
        num_bins (int, optional): Number of bins for the histogram.
        thresholds (list, optional): A list of numerical values to plot as vertical lines.
    """
    fig_width, fig_height = 4, 4
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    scale = fig_width / 5

    brightness_vals = df['brightness_integrated'].values

    # Apply min/max filtering if specified
    if min_val is not None and max_val is not None:
        brightness_vals = brightness_vals[(brightness_vals >= min_val) & (brightness_vals <= max_val)]

    # If the filtered data is empty, return an empty figure
    if len(brightness_vals) == 0:
        return fig

    # Use the min/max values to define histogram bin edges
    bins = np.linspace(min_val, max_val, num_bins)

    counts, edges, _ = ax.hist(brightness_vals, bins=bins, color='#88CCEE', edgecolor='#88CCEE', alpha=0.7)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    # Gaussian fit
    mu, sigma = None, None
    p0 = [np.max(counts), np.mean(brightness_vals), np.std(brightness_vals)]
    try:
        popt, _ = curve_fit(gaussian, bin_centers, counts, p0=p0)
        mu, sigma = popt[1], popt[2]
        x_fit = np.linspace(edges[0], edges[-1], 500)
        y_fit = gaussian(x_fit, *popt)
        ax.plot(x_fit, y_fit, color='black', linewidth=0.75, linestyle='--', label=r"μ = {mu:.0f} ± {sigma:.0f} pps".format(mu=mu, sigma=sigma))
        ax.legend(fontsize=10 * scale)
    except RuntimeError:
        pass  # Fail gracefully if fit doesn't converge

    palette = HWT_aesthetic()
    region_colors = palette[:4]

    # Draw shaded background regions first
    if thresholds:
        all_bounds = [min_val] + sorted(thresholds) + [max_val]
        for i in range(len(all_bounds) - 1):
            ax.axvspan(
                all_bounds[i],
                all_bounds[i + 1],
                color=region_colors[i % len(region_colors)],
                alpha=0.2,
                zorder=0  # optional: send even further back
            )

    # Now draw histogram bars on top
    counts, edges, _ = ax.hist(
        brightness_vals,
        bins=np.linspace(min_val, max_val, num_bins),
        color='#88CCEE',
        edgecolor='#88CCEE',
        alpha=0.7,
        zorder=1
    )

    ax.set_xlabel("Brightness (pps)", fontsize=10 * scale)
    ax.set_ylabel("Count", fontsize=10 * scale)
    ax.tick_params(axis='both', labelsize=10 * scale, width=0.75)
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    HWT_aesthetic()
    plt.tight_layout()
    return fig, mu, sigma


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def sort_UCNP_dye_sifs(uploaded_files, signal_ucnp="976", signal_dye="638"):
    ucnp_files = []
    dye_files = []

    for f in uploaded_files:
        fname = f.name
        has_ucnp = signal_ucnp in fname
        has_dye = signal_dye in fname

        if has_ucnp and not has_dye:
            ucnp_files.append(f)
        elif has_dye and not has_ucnp:
            dye_files.append(f)
        elif has_ucnp and has_dye:
            st.warning(f"File contains both excitation numbers → {fname}")
        else:
            st.warning(f"File contains neither excitation number → {fname}")

    return ucnp_files, dye_files


def match_ucnp_dye_files(ucnps, dyes):
    ucnps_sorted = sorted(ucnps, key=lambda f: natural_sort_key(f.name))
    dyes_sorted = sorted(dyes, key=lambda f: natural_sort_key(f.name))

    dye_map = {}
    for f in dyes_sorted:
        m = re.search(r"(\d+)\.sif$", f.name)
        if m:
            dye_map[int(m.group(1))] = f

    pairs = []
    matched_dyes = set()

    for ucnp_file in ucnps_sorted:
        m = re.search(r"(\d+)\.sif$", ucnp_file.name)
        if not m:
            continue
        uidx = int(m.group(1))

        cand = []
        if (uidx + 1) in dye_map and (uidx + 1) not in matched_dyes:
            cand.append(uidx + 1)
        if (uidx - 1) in dye_map and (uidx - 1) not in matched_dyes:
            cand.append(uidx - 1)

        if cand:
            chosen = cand[0]
            pairs.append((ucnp_file, dye_map[chosen]))
            matched_dyes.add(chosen)

    return pairs


def coloc_subplots(ucnp_file, dye_file, df_dict, colocalization_radius=2, show_fits=True, pix_size_um=0.1):
    ucnp_df, ucnp_img = df_dict[ucnp_file.name]
    dye_df, dye_img = df_dict[dye_file.name]

    required_cols = ['x_pix', 'y_pix', 'sigx_fit', 'sigy_fit', 'brightness_integrated']
    ucnp_has_fit = all(col in ucnp_df.columns for col in required_cols)
    dye_has_fit = all(col in dye_df.columns for col in required_cols)

    colocalized_ucnp = np.zeros(len(ucnp_df), dtype=bool) if ucnp_has_fit else None
    colocalized_dye = np.zeros(len(dye_df), dtype=bool) if dye_has_fit else None
    matched_ucnp = set()
    matched_dye = set()
    matched_records = []

    if show_fits and ucnp_has_fit and dye_has_fit:
        for idx_ucnp, row_ucnp in ucnp_df.iterrows():
            x_ucnp, y_ucnp = row_ucnp['x_pix'], row_ucnp['y_pix']
            mask = ~dye_df.index.isin(matched_dye)
            dx = dye_df.loc[mask, 'x_pix'] - x_ucnp
            dy = dye_df.loc[mask, 'y_pix'] - y_ucnp
            distances = np.hypot(dx, dy)

            if distances.min() <= colocalization_radius:
                best_idx = distances.idxmin()
                matched_ucnp.add(idx_ucnp)
                matched_dye.add(best_idx)
                colocalized_ucnp[idx_ucnp] = True
                idx_dye = dye_df.index.get_loc(best_idx)
                colocalized_dye[idx_dye] = True

                matched_records.append({
                    'source_ucnp_file': ucnp_file.name,
                    'source_dye_file': dye_file.name,
                    'ucnp_x': x_ucnp,
                    'ucnp_y': y_ucnp,
                    'ucnp_brightness': row_ucnp['brightness_integrated'],
                    'dye_x': dye_df.loc[best_idx]['x_pix'],
                    'dye_y': dye_df.loc[best_idx]['y_pix'],
                    'dye_brightness': dye_df.loc[best_idx]['brightness_integrated'],
                    'distance': distances[best_idx],
                })

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax_u, ax_d = axes

    ax_u.imshow(ucnp_img + 1, cmap='magma', origin='lower', norm=LogNorm())
    ax_d.imshow(dye_img + 1, cmap='magma', origin='lower', norm=LogNorm())

    if show_fits and ucnp_has_fit:
        for is_coloc, (_, row) in zip(colocalized_ucnp, ucnp_df.iterrows()):
            color = 'lime' if is_coloc else 'white'
            radius_px = 4 * max(row['sigx_fit'], row['sigy_fit']) / pix_size_um
            ax_u.add_patch(Circle((row['x_pix'], row['y_pix']), radius_px, edgecolor=color, facecolor='none', lw=1.5))

    if show_fits and dye_has_fit:
        for is_coloc, (_, row) in zip(colocalized_dye, dye_df.iterrows()):
            color = 'lime' if is_coloc else 'white'
            radius_px = 4 * max(row['sigx_fit'], row['sigy_fit']) / pix_size_um
            ax_d.add_patch(Circle((row['x_pix'], row['y_pix']), radius_px, edgecolor=color, facecolor='none', lw=1.5))

    ax_u.set_title(f"UCNP: {ucnp_file.name}")
    ax_d.set_title(f"Dye: {dye_file.name}")

    #st.pyplot(fig)
    st.write(f"UCNP DF shape: {ucnp_df.shape}, dye DF shape: {dye_df.shape}")
    st.write(f"Colocalization radius: {colocalization_radius}")
    st.write(f"UCNP sample coords: {ucnp_df[['x_pix', 'y_pix']].head()}")
    st.write(f"Dye sample coords: {dye_df[['x_pix', 'y_pix']].head()}")
    return pd.DataFrame(matched_records)




def extract_subregion(image, x0, y0, radius_pix):
    x_start = int(max(x0 - radius_pix, 0))
    x_end = int(min(x0 + radius_pix + 1, image.shape[1]))
    y_start = int(max(y0 - radius_pix, 0))
    y_end = int(min(y0 + radius_pix + 1, image.shape[0]))
    return image[y_start:y_end, x_start:x_end], x_start, y_start

def gaussian2d(xy, amp, x0, sigma_x, y0, sigma_y, offset):
    x, y = xy
    return (amp * np.exp(-((x - x0)**2)/(2*sigma_x**2)) *
                 np.exp(-((y - y0)**2)/(2*sigma_y**2)) + offset).ravel()


def plot_all_sifs(sif_files, df_dict, colocalization_radius=2, show_fits=True, normalization=None, save_format = 'SVG', univ_minmax=False, cmap = 'grey'):
    required_cols = ['x_pix', 'y_pix', 'sigx_fit', 'sigy_fit', 'brightness_integrated']
    all_matched_pairs = []

    n_files = len(sif_files)
    n_cols = min(4, max(1, n_files))   # between 1 and 4, never more than files
    n_rows = int(np.ceil(n_files / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]    
    all_vals = []
    if univ_minmax and normalization is None:
        all_vals = []
        for sif_file in sif_files:
            sif_name = sif_file.name
            if sif_name in df_dict:
                all_vals.append(df_dict[sif_name]["image"])
        if all_vals:
            stacked = np.stack(all_vals)
            global_min = stacked.min()
            global_max = stacked.max()
            normalization = Normalize(vmin=global_min, vmax=global_max)
    for i, sif_file in enumerate(sif_files):
        ax = axes[i]
        sif_name = sif_file.name  
        if sif_name not in df_dict:
            st.warning(f"Warning: Data for {sif_name} not found in df_dict. Skipping.")
            continue

        df = df_dict[sif_name]["df"]
        img = df_dict[sif_name]["image"]
        has_fit = all(col in df.columns for col in required_cols)

        colocalized = np.zeros(len(df), dtype=bool) if has_fit else None

        # Colocalization
        if show_fits and has_fit:
            for idx, row in df.iterrows():
                x, y = row['x_pix'], row['y_pix']
                distances = np.sqrt((df['x_pix'] - x) ** 2 + (df['y_pix'] - y) ** 2)
                distances[idx] = np.inf
                if np.any(distances <= colocalization_radius):
                    colocalized[idx] = True
                    closest_idx = distances.idxmin()
                    all_matched_pairs.append({
                        'x': x, 'y': y, 'brightness': row['brightness_integrated'],
                        'closest_x': df.at[closest_idx, 'x_pix'],
                        'closest_y': df.at[closest_idx, 'y_pix'],
                        'closest_brightness': df.at[closest_idx, 'brightness_integrated'],
                        'distance': distances[closest_idx]
                    })

        im = ax.imshow(img + 1, cmap=cmap, origin='lower', norm=normalization)
        # Only show colorbar on the last subplot in the first row (column n_cols-1)
        if not univ_minmax:
            plt.colorbar(im, ax=ax, label='pps', fraction=0.046, pad=0.04)


        basename = os.path.basename(sif_name)
        match = re.search(r'(\d+)\.sif$', basename)
        file_number = match.group(1) if match else '?'

        # Overlay fits
        if show_fits and has_fit:
            for is_coloc, (_, row) in zip(colocalized, df.iterrows()):
                color = 'lime' if is_coloc else 'white'
                radius_px = 4 * max(row['sigx_fit'], row['sigy_fit']) / 0.1
                circle = Circle((row['x_pix'], row['y_pix']), radius_px, color=color, fill=False, linewidth=1, alpha=0.7)
                ax.add_patch(circle)
                ax.text(row['x_pix'] + 7.5, row['y_pix'] + 7.5,
                        f"{row['brightness_integrated']/1000:.1f} kpps",
                        color=color, fontsize=7, ha='center', va='center')

        wrapped_basename = "\n".join(textwrap.wrap(basename, width=25))
        ax.set_title(f"Sif {file_number}\n{wrapped_basename}", fontsize = 10)
        ax.axis('off')

    # Turn off unused axes
    for ax in axes[n_files:]:
        ax.axis('off')
    if univ_minmax:
        colorbar_ax = axes[min(n_cols - 1, n_files - 1)]
        im = colorbar_ax.images[0]  # Get the image from that subplot
        plt.colorbar(im, ax=colorbar_ax, label='pps', fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Show the figure
    st.pyplot(fig)

    

    # Download button
    buf = io.BytesIO()
    
    # Save the figure to the binary buffer
    # Use bbox_inches='tight' for better layouts
    fig.savefig(buf, format=save_format, bbox_inches='tight')
    
    # Get the value from the binary buffer
    plot_data = buf.getvalue()
    buf.close()

    # Define the mime type based on the format
    if save_format.lower() == 'svg':
        mime_type = "image/svg+xml"
    elif save_format.lower() == 'tiff':
        mime_type = "image/tiff"
    elif save_format.lower() == 'png':
        mime_type = "image/png"
    elif save_format.lower() == 'jpeg' or save_format.lower() == 'jpg':
        mime_type = "image/jpeg"
    else:
        mime_type = "application/octet-stream" # Default for unknown formats

    today = date.today().strftime('%Y%m%d')
    download_name = f"sif_grid_{today}.{save_format}"
    
    st.download_button(
        label=f"Download all plots as {save_format.upper()}",
        data=plot_data,
        file_name=download_name,
        mime=mime_type
    )

    # Return colocalization results
    if all_matched_pairs:
        return pd.DataFrame(all_matched_pairs)
    return None


