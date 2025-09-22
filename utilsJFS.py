import sif_parser
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.feature import blob_log

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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

from utils import HWT_aesthetic, extract_subregion


def integrate_sif_Center(sif, threshold=1, region='all', signal='UCNP', pix_size_um = 0.1, sig_threshold = 0.3):
    image_data, metadata = sif_parser.np_open(sif, ignore_corrupt=True)
    image_data = image_data[0]  # (H, W)

    gainDAC = metadata['GainDAC']
    if gainDAC == 0:
        gainDAC =1 #account for gain turned off
    exposure_time = metadata['ExposureTime']
    accumulate_cycles = metadata['AccumulatedCycles']

    # Normalize counts → photons
    image_data_cps = image_data * (5.0 / gainDAC) / exposure_time / accumulate_cycles

    radius_um_fine = 0.3
    radius_pix_fine = int(radius_um_fine / pix_size_um)

    # --- Crop image if region specified ---
    region = str(region)
    crop = 64
    if region == '3':
        image_data_cps = image_data_cps[crop:256-crop, crop:256-crop]
    elif region == '4':
        image_data_cps = image_data_cps[crop:256-crop, 256 + crop:512 - crop]
    elif region == '1':
        image_data_cps = image_data_cps[256 + crop:512-crop, 0 + crop:256-crop]
    elif region == '2':
        image_data_cps = image_data_cps[256 + crop:512-crop, 256 + crop:512-crop]
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

            lb = [1, x0_guess - 1, 0.0, y0_guess - 1, 0.0, offset_guess * 0.5]
            ub = [2 * amp_guess, x0_guess + 1, 0.175, y0_guess + 1, 0.175, offset_guess * 1.2]

            # Perform fit
            res = least_squares(residuals, p0, args=(x_flat, y_flat, z_flat), bounds=(lb, ub))
            popt = res.x
            amp_fit, x0_fit, sigx_fit, y0_fit, sigy_fit, offset_fit = popt
            brightness_fit = 2 * np.pi * amp_fit * sigx_fit * sigy_fit / pix_size_um**2
            brightness_integrated = np.sum(sub_img_fine) - sub_img_fine.size * offset_fit

            if brightness_fit > 1e9 or brightness_fit < 50:
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



def gaussian(x, amp, mu, sigma):
  return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))
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

    brightness_vals = df['brightness_fit'].values

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
        popt = p0
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
