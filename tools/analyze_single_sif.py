import streamlit as st
import os
import io
import pandas as pd
from utils import integrate_sif, plot_brightness, plot_histogram
from tools.process_files import process_files
from matplotlib.colors import LogNorm
import numpy as np
from scipy.ndimage import gaussian_filter
import plotly.express as px
import plotly.graph_objects as go

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


def _deduplicate_psf_dataframe(df: pd.DataFrame, precision: int = 3) -> pd.DataFrame:
    if df is None:
        return df
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if df.empty or not {"x_pix", "y_pix"}.issubset(df.columns):
        return df

    working = df.copy()
    working["_x_round"] = working["x_pix"].round(precision)
    working["_y_round"] = working["y_pix"].round(precision)
    if "brightness_integrated" in working.columns:
        working = working.sort_values("brightness_integrated", ascending=False)

    deduped = working.drop_duplicates(subset=["_x_round", "_y_round"], keep="first")
    deduped = deduped.drop(columns=["_x_round", "_y_round"], errors="ignore")
    deduped.reset_index(drop=True, inplace=True)
    return deduped







def run():
    @st.cache_data
    def df_to_csv_bytes(df):
        return df.to_csv(index=False).encode("utf-8")

    if "analyze_clicked" not in st.session_state:
        st.session_state.analyze_clicked = False

    processed_data = None
    image_data_cps = None

    with st.sidebar:
        st.header("Analyze SIF Files")
        uploaded_files = st.file_uploader("Upload .sif file", type=["sif"], accept_multiple_files=True)
        threshold = st.number_input("Peak Finding Threshold", min_value=0, value=5, help='''
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
        sig_threshold = st.number_input("Sigma Threshold", min_value = 0.1, value = 0.20, help= 'Set sigma stringency in pixels')
        region = st.selectbox("Region", options=["1", "2", "3", "4", "all"], help=diagram)
        min_fit_separation_px = st.number_input(
            "Min fit separation (px)",
            min_value=0.0,
            value=3.0,
            step=0.5,
            help="Minimum allowed centre-to-centre distance between accepted PSF fits.",
        )
        min_r2 = st.slider(
            "Min fit R²",
            min_value=0.0,
            max_value=1.0,
            value=0.85,
            step=0.01,
            help="Discard Gaussian fits whose coefficient of determination falls below this threshold.",
        )

        signal = st.selectbox("Signal", options=["UCNP", "dye"], help='''Changes detection method:
                                                                - UCNP for high SNR (sklearn peakfinder)
                                                                - dye for low SNR (sklearn blob detection)''')
        min_fit_separation_px = st.number_input(
            "Min fit separation (px)",
            min_value=0.0,
            value=3.0,
            step=0.5,
            help=(
                "Minimum centre-to-centre distance between accepted PSFs. "
                "Set to 0 to disable the separation filter."
            ),
        )
        min_r2_input = st.number_input(
            "Min fit R²",
            min_value=0.0,
            max_value=1.0,
            value=0.85,
            step=0.01,
            help=(
                "Minimum Gaussian fit quality (coefficient of determination). "
                "Set to 0 to disable this filter."
            ),
        )
        min_r2 = min_r2_input if min_r2_input > 0.0 else None
        cmap = st.selectbox("Colormap", options=['plasma', 'gray', "magma", 'viridis', 'hot', 'hsv'])
        show_fits = st.checkbox("Show fits", value = True)
        normalization = st.checkbox("Log Image Scaling")
        save_format = st.selectbox("Download format", options=["svg", "png", "jpeg"]).lower()
        show_heatmap = st.toggle(
            "Show heatmap (all SIFs)",
            value=False,
            help="Aggregates brightness across all detections from all uploaded .sif files.",
        )

    if st.button("Analyze"):
        st.session_state.analyze_clicked = True

    brightness_col, hist_col = st.columns([3, 1])
    mime_map = {"svg": "image/svg+xml", "png": "image/png", "jpeg": "image/jpeg"}

    if st.session_state.analyze_clicked and uploaded_files:
        try:
            processed_data, combined_df = process_files(

                uploaded_files,
                region,
                threshold=threshold,
                signal=signal,
                sig_threshold=sig_threshold,
                min_fit_separation_px=min_fit_separation_px,
                min_r2=min_r2,
            )

            deduped_frames = []
            for file_name, entry in processed_data.items():
                df = entry.get("df")
                deduped_df = _deduplicate_psf_dataframe(df)
                processed_data[file_name]["df"] = deduped_df
                if isinstance(deduped_df, pd.DataFrame) and not deduped_df.empty:
                    deduped_frames.append(deduped_df)

            combined_df = (
                pd.concat(deduped_frames, ignore_index=True)
                if deduped_frames
                else pd.DataFrame()
            )


            if len(uploaded_files) > 1:
                file_options = [f.name for f in uploaded_files]
                selected_file_name = st.selectbox("Select sif to display:", options=file_options)
            else:
                selected_file_name = uploaded_files[0].name

            if selected_file_name in processed_data:
                selected_file_base = os.path.splitext(selected_file_name)[0]
                data_to_plot = processed_data[selected_file_name]
                df_selected = data_to_plot["df"]
                image_data_cps = data_to_plot["image"]

                normalization_to_use = LogNorm() if normalization else None
                fig_image = plot_brightness(
                    image_data_cps,
                    df_selected,
                    show_fits=show_fits,
                    normalization=normalization_to_use,
                    pix_size_um=0.1,
                    cmap=cmap,
                    interactive=True,
                )

                with brightness_col:
                    if hasattr(fig_image, "savefig"):
                        fig_image.set_size_inches(8, 8)
                        st.pyplot(fig_image, use_container_width=True)
                        buffer = io.BytesIO()
                        fig_image.savefig(buffer, format=save_format)
                        st.download_button(
                            label=f"Download PSFs ({save_format})",
                            data=buffer.getvalue(),
                            file_name=f"{selected_file_base}.{save_format}",
                            mime=mime_map[save_format],
                        )
                    else:
                        fig_image.update_layout(height=640)
                        fmt = save_format.lower()
                        if fmt not in {"png", "jpeg", "jpg", "svg", "webp"}:
                            fmt = "png"
                        st.plotly_chart(
                            fig_image,
                            use_container_width=True,
                            config={
                                "displaylogo": False,
                                "modeBarButtonsToRemove": ["select2d", "lasso2d", "toggleSpikelines"],
                                "toImageButtonOptions": {"format": fmt},
                            },
                        )
                        html_bytes = fig_image.to_html().encode("utf-8")
                        st.download_button(
                            label="Download PSFs (HTML)",
                            data=html_bytes,
                            file_name=f"{selected_file_base}.html",
                            mime="text/html",
                        )

                    if combined_df is not None and not combined_df.empty:
                        csv_bytes = df_to_csv_bytes(combined_df)
                        st.download_button(
                            label="Download as CSV",
                            data=csv_bytes,
                            file_name=f"{os.path.splitext(selected_file_name)[0]}_compiled.csv",
                            mime="text/csv",
                        )
                    else:
                        st.info("No compiled data available to download yet.")

                with hist_col:
                    if not combined_df.empty:
                        brightness_vals = combined_df['brightness_integrated'].values
                        default_min_val = float(np.min(brightness_vals))
                        default_max_val = float(np.max(brightness_vals))

                        user_min_val_str = st.text_input("Min Brightness (pps)", value=f"{default_min_val:.2e}")
                        user_max_val_str = st.text_input("Max Brightness (pps)", value=f"{default_max_val:.2e}")

                        try:
                            user_min = float(user_min_val_str)
                            user_max = float(user_max_val_str)
                        except ValueError:
                            st.warning("Please enter valid numbers (you can use scientific notation like 1e6).")
                            return

                        num_bins = st.number_input("# Bins:", value=50)

                        if user_min < user_max:
                            fig_hist, _, _ = plot_histogram(
                                combined_df,
                                min_val=user_min,
                                max_val=user_max,
                                num_bins=num_bins,
                            )
                            if hasattr(fig_hist, "savefig"):
                                st.pyplot(fig_hist, use_container_width=True)
                                hist_buffer = io.BytesIO()
                                fig_hist.savefig(hist_buffer, format=save_format)
                                st.download_button(
                                    label=f"Download histogram ({save_format})",
                                    data=hist_buffer.getvalue(),
                                    file_name=f"{selected_file_base}_histogram.{save_format}",
                                    mime=mime_map[save_format],
                                )
                            else:
                                fig_hist.update_layout(height=400)
                                fmt_h = save_format.lower()
                                if fmt_h not in {"png", "jpeg", "jpg", "svg", "webp"}:
                                    fmt_h = "png"
                                st.plotly_chart(
                                    fig_hist,
                                    use_container_width=True,
                                    config={
                                        "displaylogo": False,
                                        "modeBarButtonsToRemove": ["select2d", "lasso2d", "toggleSpikelines"],
                                        "toImageButtonOptions": {"format": fmt_h},
                                    },
                                )
                                st.download_button(
                                    label="Download histogram (HTML)",
                                    data=fig_hist.to_html().encode("utf-8"),
                                    file_name=f"{selected_file_base}_histogram.html",
                                    mime="text/html",
                                )
                        else:
                            st.warning("Min greater than max.")
            else:
                st.error(f"Data for file '{selected_file_name}' not found.")

        except Exception as e:
            st.error(f"Error processing files: {e}")
            st.session_state.analyze_clicked = False

    # --- Global Brightness Heatmap (across all SIFs) ---
    if show_heatmap:
        with hist_col:
            if processed_data:
                smooth_sigma = st.slider(
                    "Smoothing (σ, px)",
                    min_value=0.0,
                    max_value=8.0,
                    value=2.0,
                    step=0.5,
                    help="Apply Gaussian smoothing to reduce patchy coverage. Set to 0 for no smoothing.",
                )
                heat_cmap = st.selectbox(
                    "Heatmap colormap",
                    options=["hot", "magma", "inferno", "plasma", "viridis", "cividis"],
                    index=0,
                )

                try:
                    shape_hint = image_data_cps.shape if isinstance(image_data_cps, np.ndarray) else None
                    heatmap = build_brightness_heatmap(processed_data, weight_col="brightness_integrated", shape_hint=shape_hint)

                    if smooth_sigma > 0:
                        if gaussian_filter is not None:
                            heatmap = gaussian_filter(heatmap, sigma=smooth_sigma, mode="nearest")
                        else:
                            k = int(max(1, round(smooth_sigma * 3)))
                            kernel = np.ones((k, k), dtype=np.float64)
                            kernel /= kernel.sum()
                            from numpy.lib.stride_tricks import sliding_window_view
                            if heatmap.shape[0] >= k and heatmap.shape[1] >= k:
                                windows = sliding_window_view(
                                    np.pad(heatmap, ((k//2, k-1-k//2), (k//2, k-1-k//2)), mode="edge"),
                                    (k, k)
                                )
                                heatmap = (windows * kernel).sum(axis=(-1, -2))

                    import matplotlib.pyplot as plt
                    fig_hm, ax_hm = plt.subplots()
                    im = ax_hm.imshow(heatmap, origin="lower", cmap=heat_cmap, norm=None)
                    ax_hm.set_title("Brightness Heatmap (All SIFs)")
                    ax_hm.set_xlabel("X (px)")
                    ax_hm.set_ylabel("Y (px)")
                    cbar = fig_hm.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
                    cbar.set_label("Summed brightness (pps)")

                    st.pyplot(fig_hm)

                    hm_svg_buf = io.StringIO()
                    fig_hm.savefig(hm_svg_buf, format="svg")
                    heatmap_base = (
                        os.path.splitext(selected_file_name)[0]
                        if "selected_file_name" in locals() and selected_file_name
                        else "brightness_heatmap"
                    )
                    st.download_button(
                        label="Download heatmap",
                        data=hm_svg_buf.getvalue(),
                        file_name=f"{heatmap_base}_heatmap.svg",
                        mime="image/svg+xml",
                    )

                except Exception as e_hm:
                    st.warning(f"Couldn't build heatmap: {e_hm}")
            else:
                st.info("Run analysis to build heatmap.")
