
import io
import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st





def run():
  
  st.set_page_config(page_title="CSV Plotter", layout="wide")
  
  st.title("CSV Plotter — Flexible, Tidy or Paired XY")
  
  with st.sidebar:
      st.header("1) Load Data")
      uploaded = st.file_uploader("Upload a CSV", type=["csv"])
      use_example = st.checkbox("Use example (if no file)", value=not uploaded)
      paired_mode = st.checkbox("Auto-detect paired XY format", value=True,
                                help="For files where columns come in pairs like [Wavelength, Abs] with the first row describing units.")
  
      st.header("2) Plot Options")
      plot_type = st.selectbox("Plot type", ["Line", "Scatter"])
      marker_mode = st.checkbox("Show markers (for lines)")
      width = st.slider("Line width", 1, 8, 2)
      opacity = st.slider("Opacity", 0.1, 1.0, 0.95, step=0.05)
      point_size = st.slider("Point size (scatter or line markers)", 2, 20, 6)
  
      st.header("3) Axes & Scales")
      x_log = st.checkbox("Log scale X", value=False)
      y_log = st.checkbox("Log scale Y", value=False)
      x_title_override = st.text_input("X-axis title override", value="")
      y_title_override = st.text_input("Y-axis title override", value="")
  
      st.header("4) Styling")
      palette_name = st.selectbox(
          "Color palette",
          [
              "plotly", "viridis", "cividis", "plasma",
              "inferno", "magma", "Turbo", "aggrnyl", "agsunset"
          ],
          help="A selection of Plotly palettes."
      )
      custom_colors = st.text_input("Custom colors (comma-separated HEX or names)", value="",
                                    help="Example: #1f77b4, #ff7f0e, #2ca02c")
  
      st.header("5) Advanced")
      smooth = st.checkbox("Apply rolling mean smoothing")
      smooth_window = st.slider("Rolling window size", 3, 101, 9, step=2)
      group_agg = st.selectbox("If multiple rows per group, aggregate Y using:", ["None", "mean", "median"])
      download_cleaned = st.checkbox("Enable download of cleaned/reshaped data", value=True)

  if uploaded is not None:
      selected_file_name = uploaded.name
  else:
      selected_file_name = "example_data.csv" if use_example else "plotted_data.csv"
  selected_file_stem = os.path.splitext(selected_file_name)[0]
  
  # Helper palettes
  PLOTLY_PALETTES = {
      "plotly": px.colors.qualitative.Plotly,
      "viridis": px.colors.sequential.Viridis,
      "cividis": px.colors.sequential.Cividis,
      "plasma": px.colors.sequential.Plasma,
      "inferno": px.colors.sequential.Inferno,
      "magma": px.colors.sequential.Magma,
      "Turbo": px.colors.sequential.Turbo,
      "aggrnyl": px.colors.sequential.Aggrnyl,
      "agsunset": px.colors.sequential.Agsunset,
  }
  
  def parse_custom_colors(s: str):
      if not s.strip():
          return None
      parts = [p.strip() for p in s.split(",") if p.strip()]
      return parts if parts else None
  
  def try_load_example():
      # Minimal synthetic example if none supplied; you can replace with a real example later.
      x = np.linspace(0, 10, 101)
      df = pd.DataFrame({
          "time": x,
          "cond": np.where(x < 5, "A", "B"),
          "signal1": np.sin(x),
          "signal2": np.cos(x) * 0.5 + 0.2
      })
      return df
  
  def read_csv(file) -> pd.DataFrame:
      try:
          return pd.read_csv(file)
      except Exception as e:
          st.error(f"Failed to read CSV: {e}")
          return None
  
  def is_paired_xy(df: pd.DataFrame) -> bool:
      if df is None or df.empty:
          return False
      # Heuristic: first row has text like 'Wavelength (nm)' and 'Abs', and columns come in pairs
      first_row = df.iloc[0].astype(str).str.lower()
      paired_keywords = {"wavelength", "abs", "nm"}
      hits = sum(any(k in cell for k in paired_keywords) for cell in first_row)
      return hits >= 2 and (df.shape[1] % 2 == 0)
  
  def reshape_paired_xy(df: pd.DataFrame):
      # Transform a 'paired' format like:
      #   [SampleA_x, SampleA_y, SampleB_x, SampleB_y, ...]
      # where row 0 contains subheaders like 'Wavelength (nm)' / 'Abs'
      # into a tidy DataFrame with columns: x, y, series
      df_local = df.copy()
  
      # If the first row contains labels (e.g., 'Wavelength (nm)' / 'Abs'), drop it after use
      subheader = df_local.iloc[0].astype(str).tolist()
      df_local = df_local.iloc[1:].reset_index(drop=True)
  
      # Convert all numeric-like strings to numbers where possible
      df_local = df_local.apply(pd.to_numeric, errors="ignore")
  
      pairs = []
      cols = list(df_local.columns)
      i = 0
      while i < len(cols) - 1:
          left = cols[i]
          right = cols[i + 1]
  
          # Use the left column name as the series label
          series_name = str(left)
          x_vals = pd.to_numeric(df_local[left], errors="coerce")
          y_vals = pd.to_numeric(df_local[right], errors="coerce")
          pair_df = pd.DataFrame({
              "x": x_vals,
              "y": y_vals,
              "series": series_name
          })
          pairs.append(pair_df)
          i += 2
  
      tidy = pd.concat(pairs, axis=0, ignore_index=True) if pairs else pd.DataFrame(columns=["x","y","series"])
      tidy = tidy.dropna(subset=["x", "y"])
      return tidy
  
  def build_generic_controls(df: pd.DataFrame):
      st.subheader("Column Selection")
      numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
      all_cols = list(df.columns)
  
      default_x = numeric_cols[0] if numeric_cols else (all_cols[0] if all_cols else None)
      default_y = numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if numeric_cols else None)
  
      col1, col2, col3 = st.columns(3)
      with col1:
          x_col = st.selectbox("X column", all_cols, index=all_cols.index(default_x) if default_x in all_cols else 0)
      with col2:
        # allow multiple Y cols
        y_cols = st.multiselect(
            "Y column(s)", 
            options=df.columns.tolist(), 
            default=[df.columns[1]]  # default to one column
        )
     
        with col3:
          color_col = st.selectbox("Color / condition (optional)", ["(none)"] + all_cols)
  
      filters = {}
      if color_col != "(none)":
          uniques = df[color_col].dropna().unique().tolist()
          selected_vals = st.multiselect(f"Filter {color_col} values (optional)", uniques, default=uniques)
          filters[color_col] = selected_vals
  
      return x_col, y_cols, (None if color_col == "(none)" else color_col), filters
  
  def apply_filters(df: pd.DataFrame, filters: dict):
      out = df.copy()
      for col, allowed in filters.items():
          if allowed:
              out = out[out[col].isin(allowed)]
      return out
  
  def maybe_smooth(series: pd.Series, window: int, enabled: bool):
      if enabled and window and window > 1:
          try:
              return series.rolling(window, min_periods=1, center=True).mean()
          except Exception:
              return series
      return series
  
  def draw_generic_plot(df: pd.DataFrame, x_col: str, y_cols: list, color_col: str | None):
      if not x_col or not y_cols:
          st.warning("Please select an X column and at least one Y column.")
          return
  
      base_df = df.copy()
      if group_agg != "None":
          aggfunc = {"mean": "mean", "median": "median"}[group_agg]
          gb_cols = [x_col] + ([color_col] if color_col else [])
          base_df = base_df.groupby(gb_cols)[y_cols].agg(aggfunc).reset_index()
  
      # Long form: one row per (x, series, y)
      long_df = base_df.melt(
          id_vars=[c for c in [x_col, color_col] if c],
          value_vars=y_cols,
          var_name="series",
          value_name="y"
      )
  
      # Try to coerce X to numeric for proper ordering
      x_is_numeric = False
      try:
          coerced = pd.to_numeric(long_df[x_col], errors="coerce")
          if coerced.notna().any():
              long_df[x_col] = coerced
              x_is_numeric = True
      except Exception:
          pass
  
      if x_is_numeric:
          # Drop rows with non-numeric X
          long_df = long_df.dropna(subset=[x_col])
          # Sort by series then X to keep each trace monotonic
          long_df = long_df.sort_values(["series", x_col], kind="mergesort")
          category_orders = None
      else:
          # For string/categorical X, enforce a deterministic order
          # Preserve first-seen order within each series, then union
          order = []
          for _, g in long_df.groupby("series", sort=False):
              seen = list(dict.fromkeys(g[x_col].tolist()))
              # merge while preserving previously-added order
              for v in seen:
                  if v not in order:
                      order.append(v)
          category_orders = {x_col: order}
          long_df = long_df.sort_values(["series"], kind="mergesort")
  
      # Optional smoothing within each series (+ color if provided)
      if color_col:
          long_df["y"] = long_df.groupby(["series", color_col], sort=False)["y"].transform(
              lambda s: maybe_smooth(s, smooth_window, smooth)
          )
      else:
          long_df["y"] = long_df.groupby(["series"], sort=False)["y"].transform(
              lambda s: maybe_smooth(s, smooth_window, smooth)
          )
  
      # Palette
      palette = parse_custom_colors(custom_colors) or PLOTLY_PALETTES.get(palette_name)
  
      # Build figure
      common_kwargs = dict(
          x=x_col, y="y",
          color=(color_col if color_col else "series"),
          color_discrete_sequence=palette,
      )
      if not x_is_numeric and category_orders:
          common_kwargs["category_orders"] = category_orders
  
      if plot_type == "Line":
          fig = px.line(long_df, **common_kwargs)
          if marker_mode:
              fig.update_traces(mode="lines+markers", marker=dict(size=point_size), opacity=opacity, connectgaps=False)
          else:
              fig.update_traces(mode="lines", opacity=opacity, connectgaps=False)
          fig.update_traces(line=dict(width=width))
      else:
          fig = px.scatter(long_df, **common_kwargs, opacity=opacity)
          fig.update_traces(marker=dict(size=point_size))
  
      # Axes
      fig.update_xaxes(type="log" if x_log else "linear", title=(x_title_override or x_col))
      fig.update_yaxes(type="log" if y_log else "linear", title=(y_title_override or "value"))
  
      st.plotly_chart(fig, use_container_width=True)
  
      if download_cleaned:
          csv = long_df.to_csv(index=False).encode("utf-8")
          st.download_button(
              "⬇️ Download plotted data (CSV)",
              data=csv,
              file_name=f"{selected_file_stem}_plotted_data.csv",
              mime="text/csv",
          )
  
  def draw_paired_plot(tidy: pd.DataFrame):
      if tidy.empty:
          st.warning("No data detected after reshaping paired XY format.")
          return
  
      st.subheader("Series Selection (Paired XY)")
      available_series = tidy["series"].unique().tolist()
      selected_series = st.multiselect("Select series to plot", options=available_series, default=available_series)
  
      plot_df = tidy[tidy["series"].isin(selected_series)].copy()
      plot_df = plot_df.sort_values(by=["series", "x"])
  
      plot_df["y"] = plot_df.groupby("series")["y"].transform(
          lambda s: maybe_smooth(s, smooth_window, smooth)
      )
  
      palette = parse_custom_colors(custom_colors) or PLOTLY_PALETTES.get(palette_name)
  
      if plot_type == "Line":
          fig = px.line(plot_df, x="x", y="y", color="series", color_discrete_sequence=palette)
          if marker_mode:
              fig.update_traces(mode="lines+markers", marker=dict(size=point_size), opacity=opacity)
          else:
              fig.update_traces(mode="lines", opacity=opacity)
          fig.update_traces(line=dict(width=width))
      else:
          fig = px.scatter(plot_df, x="x", y="y", color="series", color_discrete_sequence=palette, opacity=opacity)
          fig.update_traces(marker=dict(size=point_size))
  
      fig.update_xaxes(type="log" if x_log else "linear", title=(x_title_override or "x"))
      fig.update_yaxes(type="log" if y_log else "linear", title=(y_title_override or "y"))
  
      st.plotly_chart(fig, use_container_width=True)
  
      if download_cleaned:
          csv = plot_df.to_csv(index=False).encode("utf-8")
          st.download_button(
              "⬇️ Download plotted data (CSV)",
              data=csv,
              file_name=f"{selected_file_stem}_plotted_data_paired.csv",
              mime="text/csv",
          )
  
  # Load data
  if uploaded is not None:
      df = read_csv(uploaded)
  elif use_example:
      df = try_load_example()
  else:
      df = None
  
  if df is None:
      st.info("Upload a CSV to begin.")
      st.stop()
  
  st.subheader("Raw Data Preview")
  st.dataframe(df.head(30), use_container_width=True)
  
  # Main logic: paired vs generic
  if paired_mode and is_paired_xy(df):
      st.success("Paired XY format detected.")
      tidy = reshape_paired_xy(df)
      st.caption(f"Tidy view: {len(tidy)} rows, {tidy['series'].nunique()} series.")
      st.dataframe(tidy.head(30), use_container_width=True)
      draw_paired_plot(tidy)
  else:
      st.info("Using generic tidy CSV mode.")
      x_col, y_cols, color_col, filters = build_generic_controls(df)
      filtered = apply_filters(df, filters) if filters else df
      draw_generic_plot(filtered, x_col, y_cols, color_col)
  
  st.markdown("---")
  st.caption("Tip: Add your own palettes or extend paired-XY detection in the code if your data has unique structure.")

