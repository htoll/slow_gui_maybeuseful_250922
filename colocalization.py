
# streamlit entry module for "UNDER CONSTRUCTION Colocalization Set"
# Minimal UI: show matched UCNP↔Dye SIFs in two columns with selectable overlays.

import os
import re
import io
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

import streamlit as st
if not hasattr(st, "experimental_rerun"):
    if hasattr(st, "rerun"):
        st.experimental_rerun = st.rerun  # type: ignore

import warnings
try:
    from scipy.optimize import OptimizeWarning  # type: ignore
except Exception:
    class OptimizeWarning(Warning):
        pass
warnings.filterwarnings("ignore", message="Adding colorbar to a different Figure")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", category=OptimizeWarning)

import sys
if "/mnt/data" not in sys.path:
    sys.path.append("/mnt/data")
import utils

PIX_SIZE_UM = 0.1  # fixed

try:
    from tools.process_files import process_files as _process_files_external  # type: ignore
except Exception:
    _process_files_external = None

# --- Helpers ---
def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def _process_files_fallback(uploaded_files, region, threshold=1, signal="UCNP", pix_size_um=PIX_SIZE_UM, sig_threshold=0.3):
    processed_data: Dict[str, Dict[str, object]] = {}
    all_dfs = []
    temp_dir = Path(tempfile.gettempdir()) / "coloc_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for uf in uploaded_files:
        file_path = temp_dir / uf.name
        with open(file_path, "wb") as f:
            f.write(uf.getbuffer())
        try:
            df, image_data_cps = utils.integrate_sif(
                str(file_path),
                region=region,
                threshold=threshold,
                signal=signal,
                pix_size_um=pix_size_um,
                sig_threshold=sig_threshold,
            )
            processed_data[uf.name] = {"df": df, "image": image_data_cps}
            all_dfs.append(df)
        except Exception as e:
            st.error(f"Error processing {uf.name}: {e}")
    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return processed_data, combined_df

def _process_files(uploaded_files, region, threshold, signal):
    if _process_files_external is not None:
        return _process_files_external(uploaded_files, region, threshold=threshold, signal=signal, pix_size_um=PIX_SIZE_UM)
    return _process_files_fallback(uploaded_files, region, threshold=threshold, signal=signal, pix_size_um=PIX_SIZE_UM)

def _split_ucnp_dye(files):
    # Prefer utils sorter if available
    if hasattr(utils, "sort_UCNP_dye_sifs"):
        return utils.sort_UCNP_dye_sifs(files)
    # Fallback by guessing from name
    ucnp, dye = [], []
    for f in files:
        name = f.name.lower()
        if "976" in name and "638" not in name:
            ucnp.append(f)
        elif "638" in name and "976" not in name:
            dye.append(f)
    return ucnp, dye

def _match_ucnp_dye_files(ucnps: List[str], dyes: List[str]) -> List[Tuple[str, str]]:
    # Deterministic sort
    u_sorted = sorted(ucnps, key=natural_sort_key)
    d_sorted = sorted(dyes, key=natural_sort_key)
    # Build a lookup from trailing index -> full name
    d_map = {}
    for fn in d_sorted:
        m = re.search(r"(\d+)\.sif$", fn)
        if m:
            d_map[int(m.group(1))] = fn
    pairs = []
    used = set()
    for uf in u_sorted:
        m = re.search(r"(\d+)\.sif$", uf)
        if not m:
            continue
        uidx = int(m.group(1))
        cand = []
        if (uidx + 1) in d_map and (uidx + 1) not in used:
            cand.append(uidx + 1)
        if (uidx - 1) in d_map and (uidx - 1) not in used:
            cand.append(uidx - 1)
        if cand:
            choice = cand[0]
            pairs.append((uf, d_map[choice]))
            used.add(choice)
    return pairs

def _overlay_circles(ax, df: pd.DataFrame, color: str, alpha: float, label: bool = False):
    from matplotlib.patches import Circle
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    required = {"x_pix","y_pix","sigx_fit","sigy_fit","brightness_fit"}
    if not required.issubset(df.columns):
        return
    for _, row in df.iterrows():
        try:
            rad_px = 4 * float(max(row["sigx_fit"], row["sigy_fit"])) / PIX_SIZE_UM  # assumes sig in µm
        except Exception:
            rad_px = 6.0  # fallback
        ax.add_patch(Circle((row["x_pix"], row["y_pix"]), radius=rad_px, color=color, fill=False, linewidth=1.2, alpha=alpha))
        if label:
            ax.text(row["x_pix"] + 8, row["y_pix"] + 8, f"{row['brightness_fit']/1000:.1f} kpps",
                    color=color, fontsize=8, ha="center", va="center")

def _compute_coloc_mask(df_u: pd.DataFrame, df_d: pd.DataFrame, radius_px: int):
    if df_u is None or df_d is None or df_u.empty or df_d.empty:
        return None, None, []
    if not {"x_pix","y_pix"}.issubset(df_u.columns) or not {"x_pix","y_pix"}.issubset(df_d.columns):
        return None, None, []
    u_mask = np.zeros(len(df_u), dtype=bool)
    d_mask = np.zeros(len(df_d), dtype=bool)
    pairs = []
    used_d = set()
    for i_u, row_u in df_u.iterrows():
        du = df_d.loc[~df_d.index.isin(used_d)]
        if du.empty:
            continue
        dx = du["x_pix"].values - row_u["x_pix"]
        dy = du["y_pix"].values - row_u["y_pix"]
        dist = np.hypot(dx, dy)
        j = dist.argmin()
        if dist[j] <= radius_px:
            d_idx = du.index[j]
            u_mask[df_u.index.get_loc(i_u)] = True
            d_mask[df_d.index.get_loc(d_idx)] = True
            pairs.append((i_u, d_idx, float(dist[j])))
            used_d.add(d_idx)
    return u_mask, d_mask, pairs

# --- App ---
def run():
    st.title("Colocalization Set (UNDER CONSTRUCTION)")
    st.caption("Matched UCNP↔Dye SIFs in two columns with overlays. Uses your existing fitting pipeline.")

    with st.sidebar:
        st.header("Inputs")
        sif_files = st.file_uploader("SIF files (UCNP + Dye)", type=["sif"], accept_multiple_files=True)
        csv_help = "Optional: upload one combined CSV with a 'sif_name'/'file' column, or per-image CSVs."
        fit_csvs = st.file_uploader("Fit CSVs (optional)", type=["csv"], accept_multiple_files=True, help=csv_help)

        st.divider()
        st.header("Fitting")
        threshold = st.number_input("Threshold", min_value=0, value=2)
        region = st.selectbox("Region", options=["1","2","3","4","all"], index=4)
        radius_px = st.number_input("Colocalization radius (pixels)", min_value=1, value=2)

        st.header("Overlays")
        show_all_fits = st.checkbox("Show ALL fits", value=True)
        show_coloc_fits = st.checkbox("Show COLOCALIZED fits", value=True)

        st.header("Display")
        cmap = st.selectbox("Colormap", options=["magma","viridis","plasma","hot","gray","hsv"], index=0)
        use_lognorm = st.checkbox("Log image scaling", value=True)

    if not sif_files:
        st.info("Upload SIF files to begin.")
        return

    # Optional: load fit CSVs
    fit_map: Dict[str, pd.DataFrame] = {}
    if fit_csvs:
        if len(fit_csvs) == 1:
            df = pd.read_csv(fit_csvs[0])
            name_col = next((c for c in df.columns if c.lower() in ("sif_name","filename","file","image","sif")), None)
            if name_col:
                for name, sub in df.groupby(name_col):
                    base = os.path.basename(str(name))
                    fit_map[os.path.splitext(base)[0]] = sub.reset_index(drop=True)
        if not fit_map:
            for f in fit_csvs:
                base_noext = os.path.splitext(os.path.basename(f.name))[0]
                try:
                    fit_map[base_noext] = pd.read_csv(f)
                except Exception as e:
                    st.warning(f"Could not read CSV '{f.name}': {e}")

    # Split & process
    ucnp_files, dye_files = _split_ucnp_dye(sif_files)
    u_data, _ = _process_files(ucnp_files, region=region, threshold=threshold, signal="UCNP") if ucnp_files else ({}, pd.DataFrame())
    d_data, _ = _process_files(dye_files,  region=region, threshold=threshold, signal="dye")   if dye_files  else ({}, pd.DataFrame())

    # Build simple name lists for matching (just the .name keys we used in dicts)
    u_names = sorted(list(u_data.keys()), key=natural_sort_key)
    d_names = sorted(list(d_data.keys()), key=natural_sort_key)
    pairs = _match_ucnp_dye_files(u_names, d_names)
    st.caption(f"Matched {len(pairs)} UCNP↔Dye pairs.")

    # Prepare Matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # Accumulate matched-peak rows
            tab_pairs, tab_plots = st.tabs(["Pairs", "Plots"])

        with tab_pairs:
matched_rows = []

    # Render each pair as a row with 2 columns
    for u_name, d_name in pairs:
        u_bundle = u_data.get(u_name, {})
        d_bundle = d_data.get(d_name, {})
        u_df = u_bundle.get("df", pd.DataFrame())
        d_df = d_bundle.get("df", pd.DataFrame())
        # Prefer CSV fits if present
        base_u = os.path.splitext(os.path.basename(u_name))[0]
        base_d = os.path.splitext(os.path.basename(d_name))[0]
        if base_u in fit_map: u_df = fit_map[base_u]
        if base_d in fit_map: d_df = fit_map[base_d]
        u_img = u_bundle.get("image", None)
        d_img = d_bundle.get("image", None)

        colL, colR = st.columns(2)
        with colL:
            st.markdown(f"**UCNP:** {u_name}")
            if isinstance(u_img, np.ndarray):
                fig_u, ax_u = plt.subplots(figsize=(5,5))
                ax_u.set_xticks([]); ax_u.set_yticks([])
                norm = LogNorm() if use_lognorm else None
                ax_u.imshow(u_img + 1, cmap=cmap, norm=norm, origin="lower")
            else:
                fig_u, ax_u = plt.subplots(figsize=(5,5))
                ax_u.text(0.5,0.5,"No image", ha="center", va="center"); ax_u.axis("off")

        with colR:
            st.markdown(f"**Dye:** {d_name}")
            if isinstance(d_img, np.ndarray):
                fig_d, ax_d = plt.subplots(figsize=(5,5))
                ax_d.set_xticks([]); ax_d.set_yticks([])
                norm = LogNorm() if use_lognorm else None
                ax_d.imshow(d_img + 1, cmap=cmap, norm=norm, origin="lower")
            else:
                fig_d, ax_d = plt.subplots(figsize=(5,5))
                ax_d.text(0.5,0.5,"No image", ha="center", va="center"); ax_d.axis("off")

        # Compute coloc mask & overlay
        u_mask, d_mask, pair_idx = _compute_coloc_mask(u_df, d_df, radius_px=radius_px)

        # Overlays
        if show_all_fits:
            _overlay_circles(ax_u, u_df, color="white", alpha=0.7, label=False)
            _overlay_circles(ax_d, d_df, color="white", alpha=0.7, label=False)
        if show_coloc_fits and u_mask is not None and d_mask is not None:
            _overlay_circles(ax_u, u_df[u_mask], color="lime", alpha=0.9, label=False)
            _overlay_circles(ax_d, d_df[d_mask], color="lime", alpha=0.9, label=False)

        # Collect matched pairs data rows (if any)
        if pair_idx:
            for i_u, i_d, dist in pair_idx:
                row_u = u_df.loc[i_u] if i_u in u_df.index else None
                row_d = d_df.loc[i_d] if i_d in d_df.index else None
                if row_u is not None and row_d is not None:
                    matched_rows.append({
                        "ucnp_sif": u_name,
                        "dye_sif": d_name,
                        "ucnp_x_pix": row_u.get("x_pix", np.nan),
                        "ucnp_y_pix": row_u.get("y_pix", np.nan),
                        "ucnp_brightness": row_u.get("brightness_fit", np.nan),
                        "dye_x_pix": row_d.get("x_pix", np.nan),
                        "dye_y_pix": row_d.get("y_pix", np.nan),
                        "dye_brightness": row_d.get("brightness_fit", np.nan),
                        "distance_px": dist,
                    })

        with colL: st.pyplot(fig_u)
        with colR: st.pyplot(fig_d)

    # Download matched results CSV
    if matched_rows:
        matched_df = pd.DataFrame(matched_rows)
            st.session_state['coloc_matched_df'] = matched_df

        st.download_button(
            "Download colocalized pairs (CSV)",
            data=matched_df.to_csv(index=False).encode("utf-8"),
            file_name="colocalized_pairs.csv",
            mime="text/csv",
        )



        with tab_plots:
            st.subheader("Plots")
            matched_df = st.session_state.get("coloc_matched_df", pd.DataFrame())
            if matched_df is None or matched_df.empty:
                st.info("No matched peaks yet — open the Pairs tab first.")
            else:
                mode = st.radio("Single-particle brightness", ["Automatic (Gaussian μ)", "Manual (enter pps)"], index=0, help="Automatic assumes most PSFs are single particles. If this is not true, use Manual.")
                st.warning("Automatic mode assumes the majority of PSFs are single particles.")

                if mode.startswith("Automatic"):
                    try:
                        from scipy.stats import norm
                        uvals = matched_df["ucnp_brightness"].astype(float).to_numpy()
                        dvals = matched_df["dye_brightness"].astype(float).to_numpy()
                        import numpy as _np
                        uvals = uvals[_np.isfinite(uvals) & (uvals > 0)]
                        dvals = dvals[_np.isfinite(dvals) & (dvals > 0)]
                        mu_ucnp, _ = norm.fit(uvals) if uvals.size else (_np.nan, _np.nan)
                        mu_dye,  _ = norm.fit(dvals) if dvals.size else (_np.nan, _np.nan)
                        single_ucnp_brightness = float(mu_ucnp) if _np.isfinite(mu_ucnp) else 1.0
                        single_dye_brightness  = float(mu_dye)  if _np.isfinite(mu_dye)  else 1.0
                        st.caption(f"Estimated single UCNP = {single_ucnp_brightness:.3e} pps, single Dye = {single_dye_brightness:.3e} pps")
                    except Exception as e:
                        st.error(f"Gaussian fit failed: {e}")
                        single_ucnp_brightness = 1.0
                        single_dye_brightness = 1.0
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        single_ucnp_brightness = st.number_input("Single UCNP brightness (pps)", min_value=0.0, value=1e5, format="%.3e")
                    with c2:
                        single_dye_brightness  = st.number_input("Single Dye brightness (pps)", min_value=0.0, value=1e5, format="%.3e")

                md = matched_df.copy()
                md["num_ucnps"] = md["ucnp_brightness"].astype(float) / max(single_ucnp_brightness, 1e-12)
                md["num_dyes"]  = md["dye_brightness"].astype(float)  / max(single_dye_brightness,  1e-12)

                thresh_factor = st.number_input("Exclude UCNPs below factor × single UCNP", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
                thresholded_df = md[md["ucnp_brightness"] >= thresh_factor * single_ucnp_brightness].copy()

                import matplotlib.pyplot as plt
                fig_sc2, ax_sc2 = plt.subplots(figsize=(6,5))
                ax_sc2.scatter(md["num_ucnps"].to_numpy(), md["num_dyes"].to_numpy(), alpha=0.6)
                ax_sc2.set_xlabel("Number of UCNPs per PSF")
                ax_sc2.set_ylabel("Number of Dyes per PSF")
                ax_sc2.set_title("Matched UCNPs")
                ax_sc2.set_xlim(0, 5)
                st.pyplot(fig_sc2)

                msk = (thresholded_df["num_ucnps"] >= 0) & (thresholded_df["num_ucnps"] <= 2)
                y_subset = thresholded_df.loc[msk, "num_dyes"].dropna().to_numpy()
                fig_h2, ax_h2 = plt.subplots(figsize=(6,5))
                if y_subset.size:
                    import numpy as _np
                    mean_val = float(_np.mean(y_subset))
                    ax_h2.hist(y_subset, bins=15, edgecolor="black")
                    ax_h2.set_title(f"Single UCNPs: Mean = {mean_val:.1f}")
                else:
                    ax_h2.hist([], bins=15, edgecolor="black")
                    ax_h2.set_title("Single UCNPs: no data in [0, 2] after threshold")
                ax_h2.set_xlabel("Number of Dyes per Single UCNP")
                ax_h2.set_ylabel("Count")
                st.pyplot(fig_h2)

                st.download_button(
                    "Download thresholded results (CSV)",
                    data=thresholded_df.to_csv(index=False).encode("utf-8"),
                    file_name="thresholded_results.csv",
                    mime="text/csv",
                )
if __name__ == "__main__":
    run()
