# tools/monomers.py
import streamlit as st
import os, io, tempfile, hashlib, re
from contextlib import contextmanager

import pandas as pd
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
from matplotlib.lines import Line2D
import tools.process_files as process_files_module

from utils import integrate_sif, plot_histogram, HWT_aesthetic

CATEGORY_ORDER = ["Monomers", "Dimers", "Trimers", "Multimers"]
CATEGORY_COLORS = {
    "Monomers":  "#029E73",  # green
    "Dimers":    "#0173B2",  # blue
    "Trimers":   "#DE8F05",  # orange
    "Multimers": "#D55E00",  # red
}

CACHE_SESSION_KEY = "monomers_file_cache"


def thresholds_from_single_brightness(single_ucnp_brightness: float):
    """
    Return brightness thresholds [t1, t2, t3] in pps
    that split Monomers < 2x, 2x<=Dimers<3x, 3x<=Trimers<4x, >=4x Multimers.
    """
    t1 = 2.0 * single_ucnp_brightness
    t2 = 3.0 * single_ucnp_brightness
    t3 = 4.0 * single_ucnp_brightness
    return [t1, t2, t3]

def _hash_file(uploaded_file):
    uploaded_file.seek(0)
    h = hashlib.md5(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return h


def _hash_file_path(path: str, chunk_size: int = 1 << 20) -> str:
    """Compute an md5 hash for an on-disk file path."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_size), b""):
                if not chunk:
                    break
                h.update(chunk)
    except FileNotFoundError:
        return ""
    return h.hexdigest()


def _normalize_saved_values(values_iterable):
    """Return a list of (display_name, temp_path, file_hash) tuples."""
    normalized = []
    for v in values_iterable:
        name = path = file_hash = None
        if isinstance(v, (tuple, list)):
            if len(v) >= 3:
                name, path, file_hash = v[:3]
            elif len(v) == 2:
                name, path = v
                file_hash = _hash_file_path(path)
        elif isinstance(v, str):
            path = v
            name = os.path.basename(v)
            file_hash = _hash_file_path(v)

        if name is None or path is None:
            continue
        if not file_hash:
            file_hash = _hash_file_path(path)
        normalized.append((str(name), str(path), str(file_hash)))

    return normalized


def _ensure_triplet_saved_files(saved_files: dict) -> None:
    """Mutate saved_files so every entry is a (name, path, hash) triple."""
    for key, value in list(saved_files.items()):
        name = path = file_hash = None
        if isinstance(value, str):
            path = value
            name = os.path.basename(path)
        elif isinstance(value, (tuple, list)):
            if len(value) >= 3:
                name, path, file_hash = value[:3]
            elif len(value) == 2:
                name, path = value

        if name is None or path is None:
            saved_files.pop(key, None)
            continue

        if not file_hash:
            file_hash = _hash_file_path(path)

        saved_files[key] = (str(name), str(path), str(file_hash))


def _prune_cached_results(valid_hashes):
    cache = st.session_state.get(CACHE_SESSION_KEY)
    if not cache:
        return

    for cache_key in list(cache.keys()):
        if cache_key[0] not in valid_hashes:
            cache.pop(cache_key, None)


@contextmanager
def _patched_dataframe_constructor():
    original_ctor = pd.DataFrame

    def _dataframe_with_callable_copy(data=None, *args, **kwargs):
        if isinstance(data, (list, tuple)):
            fixed = []
            changed = False
            for item in data:
                if callable(item) and getattr(item, "__name__", "") == "copy":
                    owner = getattr(item, "__self__", None)
                    if isinstance(owner, dict):
                        fixed.append(item())
                        changed = True
                        continue
                fixed.append(item)
            if changed:
                data = fixed
        return original_ctor(data, *args, **kwargs)

    pd.DataFrame = _dataframe_with_callable_copy
    try:
        yield
    finally:
        pd.DataFrame = original_ctor


def _deduplicate_psf_dataframe(df) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if df.empty or not {"x_pix", "y_pix"}.issubset(df.columns):
        return df

    working = df.copy()
    working["_x_round"] = working["x_pix"].round(3)
    working["_y_round"] = working["y_pix"].round(3)
    if "brightness_integrated" in working.columns:
        working = working.sort_values("brightness_integrated", ascending=False)

    deduped = working.drop_duplicates(subset=["_x_round", "_y_round"], keep="first")
    deduped = deduped.drop(columns=["_x_round", "_y_round"], errors="ignore")
    deduped = deduped.reset_index(drop=True)
    return deduped


def _run_integrate_sif(
    path,
    *,
    threshold,
    region,
    signal,
    pix_size_um=0.1,
    sig_threshold=0.2,
    min_fit_separation_px=3.0,
    min_r2=0.85,
):
    with _patched_dataframe_constructor():
        df, image_data_cps = integrate_sif(
            path,
            threshold=threshold,
            region=region,
            signal=signal,
            pix_size_um=pix_size_um,
            sig_threshold=sig_threshold,
            min_fit_separation_px=min_fit_separation_px,
            min_r2=min_r2,
        )

    df = _deduplicate_psf_dataframe(df)
    return df, image_data_cps


def _process_files_cached(
    saved_records,
    region,
    threshold,
    signal,
    pix_size_um=0.1,
    sig_threshold=0.3,
    min_fit_separation_px=3.0,
    min_r2=0.85,
):

    cache = st.session_state.setdefault(CACHE_SESSION_KEY, {})
    processed_data = {}
    combined_frames = []

    for display_name, path, file_hash in saved_records:
        cache_key = (
            file_hash,
            str(region),
            float(threshold),
            str(signal),
            float(pix_size_um),
            float(sig_threshold),
            None if min_fit_separation_px is None else float(min_fit_separation_px),
            float(min_fit_separation_px),
            None if min_r2 is None else float(min_r2),
        )
        cached_entry = cache.get(cache_key)
        if cached_entry is not None:
            df_cached = cached_entry["df"].copy(deep=True)
            image_data = cached_entry["image"]
        else:
            df_cached, image_data = _run_integrate_sif(
                path,
                threshold=threshold,
                region=region,
                signal=signal,
                pix_size_um=pix_size_um,
                sig_threshold=sig_threshold,
                min_fit_separation_px=min_fit_separation_px,
                min_r2=min_r2,
            )
            cache[cache_key] = {
                "df": df_cached.copy(deep=True),
                "image": image_data.copy() if isinstance(image_data, np.ndarray) else image_data,
            }
            df_cached = df_cached.copy(deep=True)

        processed_data[display_name] = {"df": df_cached, "image": image_data}
        if not df_cached.empty:
            combined_frames.append(df_cached)

    combined_df = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame()
    return processed_data, combined_df


def plot_monomer_brightness(
    image_data_cps,
    df,
    show_fits=True,
    plot_brightness_histogram=False,
    normalization=False,
    pix_size_um=0.1,
    cmap='magma',
    single_ucnp_brightness=None,
    *,
    interactive=False,
    dragmode='zoom'
):
    """
    Plot brightness map and overlay Gaussian-fit circles colored by brightness category.
    If interactive=True returns a Plotly figure; otherwise Matplotlib.
    """
    if not interactive:
        fig_width, fig_height = 5, 5
        scale = fig_width / 5
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        norm = LogNorm() if normalization else None
        im = ax.imshow(image_data_cps + 1, cmap=cmap, norm=norm, origin='lower')
        ax.tick_params(axis='both', length=0,
                       labelleft=False, labelright=False,
                       labeltop=False, labelbottom=False)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10 * scale)
        cbar.set_label('pps', fontsize=10 * scale)

        if single_ucnp_brightness is None:
            single_ucnp_brightness = float(np.mean(image_data_cps))

        t1, t2, t3 = thresholds_from_single_brightness(single_ucnp_brightness)

        if show_fits:
            for _, row in df.iterrows():
                x_px = row['x_pix']
                y_px = row['y_pix']
                brightness_pps = row['brightness_integrated']
                brightness_kpps = brightness_pps / 1000.0

                radius_px = 3 * max(row['sigx_fit'], row['sigy_fit']) / pix_size_um

                if brightness_pps < t1:
                    cat = "Monomers"
                elif brightness_pps < t2:
                    cat = "Dimers"
                elif brightness_pps < t3:
                    cat = "Trimers"
                else:
                    cat = "Multimers"

                circle_color = CATEGORY_COLORS[cat]
                circle = Circle((x_px, y_px), radius_px,
                                color=circle_color, fill=False,
                                linewidth=1.25 * scale, alpha=0.95)
                ax.add_patch(circle)

                ax.text(x_px + 7.5, y_px + 7.5,
                        f"{brightness_kpps:.1f} kpps",
                        color='white', fontsize=7 * scale,
                        ha='center', va='center')

            legend_elements = [
                Line2D([0], [0], color=CATEGORY_COLORS["Monomers"], lw=2, label="Monomers"),
                Line2D([0], [0], color=CATEGORY_COLORS["Dimers"],   lw=2, label="Dimers"),
                Line2D([0], [0], color=CATEGORY_COLORS["Trimers"],  lw=2, label="Trimers"),
                Line2D([0], [0], color=CATEGORY_COLORS["Multimers"],lw=2, label="Multimers"),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=8, frameon=False, labelcolor='white')

        plt.tight_layout()
        HWT_aesthetic()
        return fig

    # --- Interactive Plotly path ---
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np

    if single_ucnp_brightness is None:
        single_ucnp_brightness = float(np.mean(image_data_cps))
    t1, t2, t3 = thresholds_from_single_brightness(single_ucnp_brightness)

    cmap_map = {
        "magma": "Magma", "viridis": "Viridis", "plasma": "Plasma",
        "hot": "Hot", "gray": "Gray", "hsv": "HSV", "cividis": "Cividis", "inferno": "Inferno"
    }
    plotly_scale = cmap_map.get(cmap, "Magma")

    img = image_data_cps.astype(float)
    if normalization:
        eps = max(float(np.percentile(img, 0.01)), 1e-9)
        img_display = np.log10(np.clip(img + 1.0, eps, None))
    else:
        img_display = img

    fig = px.imshow(img_display, origin="lower", aspect="equal", color_continuous_scale=plotly_scale)
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

    if df is not None and not df.empty:
        xs = df['x_pix'].to_numpy()
        ys = df['y_pix'].to_numpy()
        rs = (3 * np.maximum(df['sigx_fit'].to_numpy(), df['sigy_fit'].to_numpy()) / pix_size_um).astype(float)
        br = df['brightness_integrated'].to_numpy()
        br_k = (br / 1000.0).astype(float)
        cats = np.where(br < t1, 'Monomers', np.where(br < t2, 'Dimers', np.where(br < t3, 'Trimers', 'Multimers')))
        colors = [CATEGORY_COLORS[c] for c in cats]

        custom = np.stack([br_k, cats], axis=1)
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='markers',
            marker=dict(size=1, opacity=0),
            name='Fits',
            customdata=custom,
            hovertemplate="x=%{x:.2f}px<br>y=%{y:.2f}px<br>brightness=%{customdata[0]:.1f} kpps<br>%{customdata[1]}<extra></extra>",
            showlegend=False,
        ))

        if show_fits:
            shapes=[]
            for x,y,r,c in zip(xs,ys,rs,colors):
                shapes.append(dict(type='circle', xref='x', yref='y', x0=x-r, x1=x+r, y0=y-r, y1=y+r,
                                   line=dict(width=1.5, color=c), fillcolor='rgba(0,0,0,0)', layer='above'))
            fig.update_layout(shapes=shapes)

            fig.add_trace(go.Scatter(
                x=xs + 7.5,
                y=ys + 7.5,
                mode='text',
                text=[f"{v:.1f} kpps" for v in br_k],
                textfont=dict(color='white', size=10),
                textposition='middle center',
                showlegend=False,
                hoverinfo='skip',
            ))

            for cat in CATEGORY_ORDER:
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                         marker=dict(color=CATEGORY_COLORS[cat], size=10),
                                         name=cat))

    h, w = img.shape
    fig.update_xaxes(range=[-0.5, w - 0.5], constrain='domain', showgrid=False, zeroline=False)
    fig.update_yaxes(range=[-0.5, h - 0.5], scaleanchor='x', scaleratio=1, showgrid=False, zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def _process_files_cached(
    saved_records,
    region,
    threshold,
    signal,
    pix_size_um=0.1,
    sig_threshold=0.3,
    min_fit_separation_px=3.0,
    min_r2=0.85,
):

    class _FakeUpload:
        def __init__(self, name, path):
            self.name = name
            self._path = path
        def getbuffer(self):
            with open(self._path, "rb") as f:
                return memoryview(f.read())

    uploads = [_FakeUpload(name, path) for name, path, *_ in saved_records]
    pf = getattr(process_files_module.process_files, "__wrapped__", None)
    if pf is None:
        raise RuntimeError("process_files.__wrapped__ not found; cannot bypass Streamlit cache.")

    # FORWARD the parameters to process_files.__wrapped__
    processed_data, combined_df = pf(
        uploads,
        region=region,
        threshold=threshold,
        signal=signal,
        pix_size_um=pix_size_um,
        sig_threshold=sig_threshold,
        min_fit_separation_px=min_fit_separation_px,
        min_r2=min_r2,
    )

    deduped_data = {}
    deduped_frames = []
    for name, entry in processed_data.items():
        df = entry.get("df")
        deduped_df = _deduplicate_psf_dataframe(df)
        deduped_data[name] = {"df": deduped_df, "image": entry.get("image")}
        if isinstance(deduped_df, pd.DataFrame) and not deduped_df.empty:
            deduped_frames.append(deduped_df)

    combined_df = (
        pd.concat(deduped_frames, ignore_index=True)
        if deduped_frames
        else pd.DataFrame()
    )

    return deduped_data, combined_df
def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

def run():
    col1, col2 = st.columns([1, 2])

    # Persistent state
    if "saved_files" not in st.session_state:
        # key -> (display_name, temp_path)  (legacy may be plain path str)
        st.session_state.saved_files = {}
    if "processed" not in st.session_state:
        st.session_state.processed = None  # (processed_data, combined_df)
    if "selected_file_name" not in st.session_state:
        st.session_state.selected_file_name = None
    if CACHE_SESSION_KEY not in st.session_state:
        st.session_state[CACHE_SESSION_KEY] = {}

    with col1:
        uploaded_files = st.file_uploader(
            "Upload .sif file", type=["sif"], accept_multiple_files=True
        )

        # --- SYNC PHASE: make session match current uploader selection ---
        prev_keys = set(st.session_state.saved_files.keys())
        changed = False

        # 1) Add new uploads
        current_keys = set()
        if uploaded_files:
            for f in uploaded_files:
                file_hash = _hash_file(f)
                key = f"{f.name}:{file_hash}"
                current_keys.add(key)
                if key not in st.session_state.saved_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".sif") as tmp:
                        tmp.write(f.getbuffer())
                        st.session_state.saved_files[key] = (f.name, tmp.name, file_hash)
                    changed = True

        # 2) Remove files no longer present in the uploader
        #    (also clean up their temp files)
        stale_keys = [k for k in st.session_state.saved_files.keys() if k not in current_keys]
        for k in stale_keys:
            val = st.session_state.saved_files[k]
            if isinstance(val, (tuple, list)) and len(val) >= 2:
                path = val[1]
            else:
                path = str(val)
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
            del st.session_state.saved_files[k]
            changed = True

        _ensure_triplet_saved_files(st.session_state.saved_files)

        # 3) If the set of saved files changed (added/removed), invalidate results
        if changed or (set(st.session_state.saved_files.keys()) != prev_keys):
            st.session_state.processed = None
            # If selected file no longer exists, clear selection
            current_names = [v[0] for v in st.session_state.saved_files.values()]
            if st.session_state.selected_file_name not in current_names:
                st.session_state.selected_file_name = None


        # --- UI to select file & params (based on synced saved_files) ---
        current_values = list(st.session_state.saved_files.values())
        normalized_records = _normalize_saved_values(current_values)
        valid_hashes = {record[2] for record in normalized_records}
        _prune_cached_results(valid_hashes)

        file_options = [display for (display, _, _) in normalized_records]

        if file_options:
            default_index = 0
            if st.session_state.selected_file_name in file_options:
                default_index = file_options.index(st.session_state.selected_file_name)
            selected_file_name = st.selectbox(
                "Select sif to display:", options=file_options, index=default_index
            )
            st.session_state.selected_file_name = selected_file_name

            # Parameters (kept to preserve existing UI)
            threshold = st.number_input(
                                        "Threshold", min_value=1, value=1,
                                        help=("Stringency of fit, higher value is more selective:\n"
                                              "- UCNP signal sets absolute peak cut off\n"
                                              "- Dye signal sets sensitivity of blob detection")
                                        )

            signal = st.selectbox(
                                    "Signal", options=["UCNP", "dye"],
                                    help=("Changes detection method:\n"
                                          "- UCNP for high SNR (sklearn peakfinder)\n"
                                          "- dye for low SNR (sklearn blob detection)")
                                    )
            min_fit_separation_px = st.number_input(
                "Min fit separation (px)",
                min_value=0.0,
                value=3.0,
                step=0.5,
                help=(
                    "Minimum distance between accepted PSFs. "
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

            diagram = """ Splits sif into quadrants (256x256 px):
                                ┌─┬─┐
                                │ 1 │ 2 │
                                ├─┼─┤
                                │ 3 │ 4 │
                                └─┴─┘
                                """
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

            cmap = st.selectbox("Colormap", options=["plasma", "viridis", "magma", "hot", "gray", "hsv"])
            st.session_state["monomers_cmap"] = cmap

            # PROCESS
            if st.button("Process uploaded files"):
                with st.spinner("Processing…"):
                    saved_records = tuple(normalized_records)
                    processed_data, combined_df = _process_files_cached(
                        saved_records,
                        region=region,
                        threshold=threshold,
                        signal=signal,
                        min_fit_separation_px=min_fit_separation_px,
                        min_r2=min_r2,

                    st.session_state.processed = (processed_data, combined_df)

    # DISPLAY
    if st.session_state.get("processed"):
        processed_data, combined_df = st.session_state.processed

        plot_col1, plot_col2 = col2.columns(2)
        mime_map = {"svg": "image/svg+xml", "png": "image/png", "jpeg": "image/jpeg"}

        with plot_col1:
            show_fits = st.checkbox("Show fits", value=True)
            normalization = st.checkbox("Log Image Scaling", value = True)
            save_format = st.selectbox("Download format", ["svg", "png", "jpeg"]).lower()

            selected_file_name = st.session_state.get("selected_file_name")
            if not selected_file_name and processed_data:
                selected_file_name = next(iter(processed_data.keys()))
                st.session_state.selected_file_name = selected_file_name

            if selected_file_name in processed_data:
                data_to_plot = processed_data[selected_file_name]
                df_selected = data_to_plot["df"]
                image_data_cps = data_to_plot["image"]
                fig_image = plot_monomer_brightness(
                    image_data_cps,
                    df_selected,
                    show_fits=show_fits,
                    normalization=normalization,
                    pix_size_um=0.1,
                    cmap=st.session_state.get("monomers_cmap", "magma"),
                    single_ucnp_brightness = st.session_state.get("single_ucnp_brightness"),
                    interactive=True,
                )
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
                    file_name=f"{os.path.splitext(selected_file_name)[0]}.html",
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
            else:
                st.error(f"Data for file '{selected_file_name}' not found.")

        with plot_col2:
            if not combined_df.empty:
                # Get defaults
                brightness_vals = combined_df['brightness_integrated'].values
                default_min_val = float(np.min(brightness_vals))
                default_max_val = float(np.max(brightness_vals))
                
                user_min_val_str = st.text_input("Min Brightness (pps)", value=f"{default_min_val:.2e}")
                user_max_val_str = st.text_input("Max Brightness (pps)", value=f"{default_max_val:.2e}")
                
                try:
                    user_min_val = float(user_min_val_str); user_max_val = float(user_max_val_str)
                except ValueError:
                    st.warning("Please enter valid numbers (you can use scientific notation like 1e6).")
                    st.stop()
                
                if user_min_val >= user_max_val:
                    st.warning("Min brightness must be less than max brightness.")
                else:
                    num_bins = st.number_input("# Bins:", value=50)
                
                    # Single-particle brightness drives thresholds everywhere
                    # default to previous session value or to mean image intensity if missing
                    default_spb = st.session_state.get("single_ucnp_brightness", float(np.mean(brightness_vals)))
                    single_ucnp_brightness = st.number_input(
                        "Single Particle Brightness (pps)",
                        min_value=user_min_val, max_value=user_max_val, value=float(default_spb)
                    )
                    st.session_state["single_ucnp_brightness"] = float(single_ucnp_brightness)
                
                    thresholds = thresholds_from_single_brightness(single_ucnp_brightness)
                
                    # Plot the histogram with unified thresholds
                    fig_hist_final, _, _ = plot_histogram(
                        combined_df,
                        min_val=user_min_val,
                        max_val=user_max_val,
                        num_bins=num_bins,
                        thresholds=thresholds
                    )
                    st.pyplot(fig_hist_final)


                    with plot_col1:
                        bins_for_pie = [user_min_val] + [t for t in thresholds if user_min_val < t < user_max_val] + [user_max_val]
                        bins_for_pie = sorted(bins_for_pie)
                        num_bins_pie = len(bins_for_pie) - 1
                        labels_for_pie = CATEGORY_ORDER[:num_bins_pie]

                        if len(labels_for_pie) != num_bins_pie:
                            st.warning(f"Label/bin mismatch: {len(labels_for_pie)} labels for {num_bins_pie} bins.")
                        else:
                            categories = pd.cut(
                                combined_df['brightness_integrated'],
                                bins=bins_for_pie,
                                right=False,
                                include_lowest=True,
                                labels=labels_for_pie
                            )
                            category_counts = categories.value_counts().reset_index()
                            category_counts.columns = ['Category', 'Count']
                    
                            # Force consistent color order by label
                            plotly_colors = [mcolors.to_hex(CATEGORY_COLORS[label]) for label in labels_for_pie]
                    
                            fig_pie = px.pie(
                                category_counts,
                                values='Count',
                                names='Category',
                                color='Category',
                                color_discrete_map=CATEGORY_COLORS
                            )
                            # Larger fonts for readability
                            fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=18)
                            fig_pie.update_layout(
                                font=dict(size=18),
                                legend=dict(font=dict(size=16)),
                                margin=dict(l=0, r=0, t=0, b=0)
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)

        # Summary PSF count bar plot in left column (col1)
        with col1:
            if st.session_state.get("processed"):
                processed = st.session_state.processed[0]
                psf_counts = {os.path.basename(name): len(processed[name]["df"]) for name in processed.keys()}

                def extract_sif_number(filename):
                    m = re.search(r'_([0-9]+)\.sif$', filename)
                    return m.group(1) if m else filename

                file_names = [extract_sif_number(n) for n in psf_counts.keys()]
                counts = list(psf_counts.values())
                mean_count = np.mean(counts) if counts else 0

                fig_count, ax_count = plt.subplots(figsize=(5, 3))
                ax_count.bar(file_names, counts)
                ax_count.axhline(mean_count, color=CATEGORY_COLORS["Multimers"], linestyle='--',
                 label=f'Avg = {mean_count:.1f}', linewidth=0.8)                
                ax_count.set_ylabel("# Fit PSFs", fontsize=10)
                ax_count.set_xlabel("SIF #", fontsize=10)
                ax_count.legend(fontsize=10)
                ax_count.tick_params(axis='x', labelsize=8)
                ax_count.tick_params(axis='y', labelsize=8)
                HWT_aesthetic()
                st.pyplot(fig_count)
