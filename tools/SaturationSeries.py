import streamlit as st
import os
import io
from utilsJFS import plot_histogram
from utils import integrate_sif, plot_brightness
from tools.process_files import process_files
from matplotlib.colors import LogNorm
import numpy as np
from scipy.ndimage import gaussian_filter
import pandas as pd
import matplotlib.pyplot as plt

def convertToPowerDensity60x(current, sigma = 0.388): # Current in mA As of JUNE 27th 2023
    powerOut = (0.29187857*current -17.90535715)/1000
    sigmaPostObj = 3.3*sigma/150 #3.3 mm is the 
    radius = 2*sigmaPostObj/10 #cm
    area = np.pi*radius*radius
    return powerOut/(area) #W/cm^2
    
def build_brightness_heatmap(processed_data, weight_col="brightness_integrated", shape_hint=None):
    """
    Aggregates brightness by pixel location across all processed files.
    - Tries to auto-detect coordinate columns from common names.
    - Returns a 2D numpy array heatmap with summed brightness.
    """
    # Candidate column names for x/y in pixels
    x_candidates = ["x", "x_px", "col", "column", "x_pix", "x_idx"]
    y_candidates = ["y", "y_px", "row", "line", "y_pix", "y_idx"]

    # Derive a shape from the first image if possible
    if shape_hint is not None:
        img_h, img_w = shape_hint
    else:
        first_img = None
        for v in processed_data.values():
            if "image" in v and isinstance(v["image"], np.ndarray):
                first_img = v["image"]
                break
        if first_img is None:
            raise ValueError("No image arrays found to infer heatmap shape.")
        img_h, img_w = first_img.shape

    heatmap = np.zeros((img_h, img_w), dtype=np.float64)

    for item in processed_data.values():
        df = item.get("df", None)
        if df is None or df.empty:
            continue

        # Find coordinate columns
        x_col = next((c for c in x_candidates if c in df.columns), None)
        y_col = next((c for c in y_candidates if c in df.columns), None)
        if x_col is None or y_col is None:
            # Skip this file if coords are missing
            continue

        if weight_col not in df.columns:
            # Skip if brightness column missing
            continue

        xs = df[x_col].to_numpy()
        ys = df[y_col].to_numpy()
        ws = df[weight_col].to_numpy()

        # Round to nearest pixel and clamp into image bounds
        xi = np.clip(np.rint(xs).astype(int), 0, img_w - 1)
        yi = np.clip(np.rint(ys).astype(int), 0, img_h - 1)

        # Accumulate brightness at pixel locations
        np.add.at(heatmap, (yi, xi), ws)

    return heatmap 
def plot_all_quadrant_brightness_vs_current(combined_df):
    """
    Calculates and plots the mean brightness vs. current for all four quadrants
    on a single plot, with error bars showing the standard deviation of image means.
    Written by Hephaestus, a Gemini Gem tweaked by JFS
    """
    if combined_df is None or combined_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data available to plot.", ha='center', va='center')
        return fig

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define colors for each quadrant
    colors = {"1": "blue", "2": "green", "3": "red", "4": "black"}

    # Group by quadrant and process each one
    for quadrant, quad_df in combined_df.groupby('quadrant'):
        # Set the cut parameter for outlier removal.
        cut = 2.96
        # Step 1: Calculate the mean and std for each current group using transform.
        # This creates new columns aligned with the original quad_df.
        group_stats = quad_df.groupby('current')['brightness_integrated']
        group_mean = group_stats.transform('mean')
        group_std = group_stats.transform('std')
        # Step 2: Define the cutoff for each particle based on its group's statistics.
        cutoff_value = group_mean + cut * group_std
        # Step 3: Filter the DataFrame, keeping only particles below the cutoff.
        # We also handle cases where std is 0 by keeping those points.
        is_outlier = (quad_df['brightness_integrated'] > cutoff_value) & (group_std > 0)
        df_filtered = quad_df[~is_outlier]
        # Step 4: Now, calculate the final aggregate stats on the cleaned data.
        agg_data = df_filtered.groupby('current')['brightness_integrated'].agg(['mean', 'std']).reset_index()
        # Sort by current for proper line plotting.
        agg_data = agg_data.sort_values('current')
        # If a group has only one particle, its std will be NaN. Set this to 0.
        agg_data['std'] = agg_data['std'].fillna(0)

        # Step 4: Plot this quadrant's data
        ax.errorbar(
            convertToPowerDensity60x(agg_data['current']),
            agg_data['mean'],
            yerr=agg_data['std'],
            fmt='o-',
            capsize=5,
            label=f'Quadrant {quadrant}',
            color=colors.get(str(quadrant), 'gray') # Use gray for unexpected quadrants
        )

    ax.set_yscale('log')
    ax.set_xlabel("Power Density (W/cm2)")
    ax.set_ylabel("Brightness (pps)")
    ax.set_title("Mean Particle Brightness vs. Power Density by Channel")
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    return fig

@st.cache_data
def plot_quadrant_histograms_for_max_current(combined_df): # Changed arguments
    """
    Takes a combined DataFrame for all quadrants and plots a 2x2 grid of 
    brightness histograms for the single highest current found in the data.
    Written by Hephaestus, a Gemini Gem tweaked by JFS
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.flatten()

    if combined_df is None or combined_df.empty:
        fig.text(0.5, 0.5, "No data found in any quadrant.", ha='center')
        return fig

    # This part is now simplified as the df is passed in
    # This line will now execute successfully
    if 'current' not in combined_df.columns:
        combined_df['current'] = combined_df['filename'].str.extract(r'^(\d+)').astype(int)
    
    max_current = combined_df['current'].max()

    fig.suptitle(f"Brightness Histograms for Max Current: {max_current} mA", fontsize=16)

    for i in range(4):
        ax = axes[i]
        quadrant = str(i + 1)
        
        # Filter data for the current quadrant and the max current
        quad_data = combined_df[(combined_df['quadrant'] == quadrant) & (combined_df['current'] == max_current)]
        
        if not quad_data.empty:
            brightness_data = quad_data['brightness_integrated']
            ax.hist(brightness_data, bins=50, color='skyblue', edgecolor='black')
            ax.set_title(f"Quadrant {quadrant}")
            ax.set_xlabel("Brightness (pps)")
            ax.set_ylabel("Counts")
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        else:
            ax.text(0.5, 0.5, "No data", ha='center')
            ax.set_title(f"Quadrant {quadrant}")

    return fig
@st.cache_data
def process_all_quadrants(_uploaded_files, threshold, signal):
    """Processes files for all 4 quadrants and returns a single combined DataFrame."""
    all_dfs = []
    for i in range(1, 5):
        quadrant = str(i)
        processed_data_quad, _ = process_files(list(_uploaded_files), quadrant, threshold=threshold, signal=signal)
        
        for filename, data in processed_data_quad.items():
            df = data.get("df")
            
            # Check if the initial dataframe is valid
            if df is not None and not df.empty:
                # Define the boundary size
                bound = 64
                
                # Apply the filter to keep only data within the central square
                # The coordinates must be between 64 and (256 - 64) = 192
                df_filtered = df[
                    (df['x_pix'] >= bound) & (df['x_pix'] <= 256 - bound) &
                    (df['y_pix'] >= bound) & (df['y_pix'] <= 256 - bound)
                ]
        
                # Only proceed if entries remain after filtering
                if not df_filtered.empty:
                    # It's good practice to create a copy to avoid pandas warnings
                    df_with_meta = df_filtered.copy()
                    
                    df_with_meta['filename'] = filename
                    df_with_meta['quadrant'] = quadrant
                    all_dfs.append(df_with_meta)


    if not all_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df['current'] = combined_df['filename'].str.extract(r'^(\d+)').astype(int)
    return combined_df
# --- Keep your build_brightness_heatmap function here ---
# --- Add the new plot_brightness_vs_current function here ---
def run():
    col1, col2 = st.columns([1, 2])
    
    @st.cache_data
    def df_to_csv_bytes(df):
        return df.to_csv(index=False).encode("utf-8")

    with col1:
        st.header("Analyze SIF Files")
        
        uploaded_files = st.file_uploader(
            "Upload .sif files (e.g., 100_1.sif, 120_1.sif)", 
            type=["sif"], 
            accept_multiple_files=True
        )
        
        threshold = st.number_input("Threshold", min_value=0, value=2, help='''
        Stringency of fit, higher value is more selective:  
        -UCNP signal sets absolute peak cut off  
        -Dye signal sets sensitivity of blob detection
        ''')
        diagram = """ Splits sif into quadrants (256x256 px):  
        ┌─┬─┐  
        │ 1 │ 2 │  
        ├─┼─┤  
        │ 3 │ 4 │  
        └─┴─┘
        """
        region = st.selectbox("Region (for individual analysis)", options=["1", "2", "3", "4", "all"], help=diagram)

        signal = st.selectbox("Signal", options=["UCNP", "dye"], help='''Changes detection method:  
                                 - UCNP for high SNR (sklearn peakfinder)  
                                 - dye for low SNR (sklearn blob detection)''')
        cmap = st.selectbox("Colormap", options=["magma", 'viridis', 'plasma', 'hot', 'gray', 'hsv'])

    with col2:
        if "analyze_clicked" not in st.session_state:
            st.session_state.analyze_clicked = False

        if st.button("Analyze"):
            st.session_state.analyze_clicked = True
            if 'processed_data' in st.session_state:
                del st.session_state.processed_data
            if 'combined_df' in st.session_state:
                del st.session_state.combined_df

        if st.session_state.analyze_clicked and uploaded_files:
            try:
                # Process files for the individually selected region for the first tab
                processed_data, _ = process_files(uploaded_files, region, threshold=threshold, signal=signal)
                st.session_state.processed_data = processed_data

            except Exception as e:
                st.error(f"Error processing files: {e}")
                st.session_state.analyze_clicked = False

        if 'processed_data' in st.session_state:
            processed_data = st.session_state.processed_data

            # Define the new tab structure with the merged summary tab
            tab_analysis, tab_summary = st.tabs(["Image Analysis", "Quadrant Summary"])

            # --- TAB 1: INDIVIDUAL IMAGE ANALYSIS (Unchanged) ---
            with tab_analysis:
                file_options = list(processed_data.keys())
                selected_file = st.selectbox("Select SIF to display:", options=file_options, key="file_select")
                
                if selected_file:
                    selected_file_base = os.path.splitext(selected_file)[0]
                    plot_col1, plot_col2 = st.columns(2)
                    data_for_file = processed_data[selected_file]
                    unfiltered_df = data_for_file.get("df")
                
                    # Create a default empty dataframe
                    df_for_file = pd.DataFrame() 
                    
                    # Check if the unfiltered dataframe is valid before filtering
                    if unfiltered_df is not None and not unfiltered_df.empty:
                        bound = 64
                        # Apply the central square filter
                        df_for_file = unfiltered_df[
                            (unfiltered_df['x_pix'] >= bound) & (unfiltered_df['x_pix'] <= 256 - bound) &
                            (unfiltered_df['y_pix'] >= bound) & (unfiltered_df['y_pix'] <= 256 - bound)
                        ].copy() # Use .copy() to prevent pandas warnings

                    with plot_col1:
                        st.markdown("#### Image Display")
                        show_fits = st.checkbox("Show fits", key="fits_check")
                        normalization = st.checkbox("Log Image Scaling", key="log_check")
                        normalization_to_use = LogNorm() if normalization else None

                        fig_image = plot_brightness(
                            data_for_file["image"], df_for_file,
                            show_fits=show_fits, normalization=normalization_to_use,
                            pix_size_um=0.1, cmap=cmap
                        )
                        st.pyplot(fig_image)
                        svg_buffer_img = io.StringIO()
                        fig_image.savefig(svg_buffer_img, format='svg')
                        st.download_button(
                            "Download Image (SVG)",
                            svg_buffer_img.getvalue(),
                            f"{selected_file_base}.svg"
                        )

                    with plot_col2:
                        st.markdown("#### Brightness Histogram")
                        if df_for_file is not None and not df_for_file.empty:
                            brightness_vals = df_for_file['brightness_integrated'].values
                            min_val, max_val = st.slider(
                                "Select brightness range (pps):", 
                                float(0), float(np.max(brightness_vals)), 
                                (float(0), float(np.max(brightness_vals))),
                                key="hist_slider"
                            )
                            num_bins = st.number_input("# Bins:", value=50, key="hist_bins")
                            
                            fig_hist, _, _ = plot_histogram(df_for_file, min_val=min_val, max_val=max_val, num_bins=num_bins)
                            st.pyplot(fig_hist)
                            
                            svg_buffer_hist = io.StringIO()
                            fig_hist.savefig(svg_buffer_hist, format='svg')
                            st.download_button(
                                "Download Histogram (SVG)",
                                svg_buffer_hist.getvalue(),
                                f"{selected_file_base}_histogram.svg"
                            )

                            csv_bytes = df_to_csv_bytes(df_for_file)
                            st.download_button(
                                "Download Data (CSV)",
                                csv_bytes,
                                f"{selected_file_base}_data.csv"
                            )
                        else:
                            st.info(f"No particles were detected in '{selected_file}'.")

            # --- TAB 2: NEW MERGED QUADRANT SUMMARY TAB ---
            with tab_summary:
                st.markdown("### Comprehensive Quadrant Analysis")
                # Process data for all quadrants once and cache the result
                summary_df = process_all_quadrants(tuple(uploaded_files), threshold, signal)

                if not summary_df.empty:
                    # Plot 1: Brightness vs. Current for all Quads
                    st.markdown("#### Brightness vs. Current by Quadrant")
                    fig_currents = plot_all_quadrant_brightness_vs_current(summary_df)
                    st.pyplot(fig_currents)
                    svg_buffer_currents = io.StringIO()
                    fig_currents.savefig(svg_buffer_currents, format='svg')
                    st.download_button("Download Current Plot (SVG)", svg_buffer_currents.getvalue(), "all_quad_brightness_vs_current.svg", key="dl_current")

                    st.markdown("---") # Visual separator

                    # Plot 2: Histograms at Max Current
                    st.markdown("#### Brightness Histograms for Highest Current")
                    fig_quad_hist = plot_quadrant_histograms_for_max_current(summary_df)
                    st.pyplot(fig_quad_hist)
                    svg_buffer_quad = io.StringIO()
                    fig_quad_hist.savefig(svg_buffer_quad, format='svg')
                    st.download_button("Download Quadrant Plot (SVG)", svg_buffer_quad.getvalue(), "quadrant_histogram.svg", key="dl_quad")
                else:
                    st.warning("No particle data found across any quadrant. Cannot generate summary plots.") 
