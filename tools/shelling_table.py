import math
import numpy as np
import pandas as pd
import streamlit as st

def highlight_cells(val, row_name):
    if row_name == "% Volume Injected" and float(val) > 10:
        return "background-color: lightcoral"
    return ""

def run():
  
  # --- Page config ---
  st.set_page_config(
      page_title="Shelling Injection Table")
  
  
  with st.form("inputs"):
      col1, col2 = st.columns(2)
      with col1:
          delta = st.number_input(
              "Δ thickness per injection (nm)",
              min_value=0.01,
              value=0.25,
              step=0.01,
              help=(
                  "Desired increase in shell thickness per injection. "
                  "Example: 8 nm core grows to 8.5 nm after a single injection → Δ = 0.5"
              ),
              format="%0.2f",
          )
          initial_radius = st.number_input(
              "Initial core radius (nm)",
              min_value=0.0,
              value=13.7/2,
              step=0.05,
              format="%0.2f",
          )
          final_radius = st.number_input(
              "Target final radius (nm)",
              min_value=0.0,
              value=19/2,
              step=0.05,
              format="%0.2f",
          )
      with col2:
          nm_per_mL = st.number_input(
              "nm³ of shell per mL LnOA",
              min_value=1,
              value=200,
              step=1,
              help=(
                  "Estimated nanoparticle volume (nm³) grown per mL of YAc precursor added. Experimentally, this can vary from ~150-400."
              ),
          )
          injection_time = st.number_input(
              "Delay between injections (min)",
              min_value=1,
              value=20,
              step=1,
          )
          initial_rxn_vol = st.number_input(
              "Initial reaction volume (mL)",
              min_value=0.1,
              value=10.0,
              step=0.1,
          )
  
      submitted = st.form_submit_button("Calculate")
  
  
  def compute_injection_plan(delta: float,
                              initial_radius: float,
                              final_radius: float,
                              nm_per_mL: float = 200,
                              injection_time: int = 20,
                              initial_rxn_vol: float = 10.0):
      warnings = []
  
      if delta <= 0:
          raise ValueError("Δ thickness per injection must be > 0")
      if final_radius <= initial_radius:
          raise ValueError("Final radius must be greater than initial radius")
  
      num_injections = math.ceil((final_radius - initial_radius) / delta)
  
      inj_numbers = np.arange(1, num_injections + 1)
      est_radius = np.array([initial_radius + i * delta for i in range(num_injections)])
      inj_times = np.array([injection_time * (i + 1) for i in range(num_injections)], dtype=float)
  
      volume_added = np.zeros(num_injections - 1)
      for v in range(num_injections - 1):
          r1, r2 = est_radius[v], est_radius[v + 1]
          volume_added[v] = (4.0 / 3.0) * math.pi * (r2 ** 3 - r1 ** 3)
  
      yac_added = np.zeros(num_injections)
      for y in range(num_injections - 1):
          yac_added[y] = round(volume_added[y] / nm_per_mL, 2)
  
      tfa_added = np.zeros(num_injections)
      for t in range(num_injections - 1):
          tfa_added[t + 1] = round(yac_added[t] / 2.0, 2)
  
      total_vol = np.zeros(num_injections)
      pct_injected = np.zeros(num_injections)
      prev_vol = float(initial_rxn_vol)
      for q in range(num_injections):
          this_add = yac_added[q] + tfa_added[q]
          total_vol[q] = round(prev_vol + this_add, 2)
          pct = (total_vol[q] - prev_vol) / prev_vol * 100.0 if prev_vol > 0 else 0.0
          pct_injected[q] = round(pct, 2)
          if pct > 10.0:
              warnings.append(
                  f"Injection {q+1}: risks temperature fluctuation \n"
              )
          prev_vol = total_vol[q]
  
      df = pd.DataFrame({
          "Injection": inj_numbers,
          "Time (min)": inj_times,
          "Estimated radius (nm)": np.round(est_radius, 3),
          "NaTFA (mL)": tfa_added,
          "YAc (mL)": yac_added,
          "Total Rxn Volume (mL)": total_vol,
          "% Volume Injected": pct_injected,
      })
  
      # Transpose so each injection is a column
      df_t = df.set_index("Injection").T
      df_t.index.name = None  # hide row index label
      df_t.columns = [f"Injection {int(c)}" for c in df_t.columns]  # prettier column headers
      return df_t, warnings
  
  if submitted:
    try:
        df_t, warnings = compute_injection_plan(
            delta=delta,
            initial_radius=initial_radius,
            final_radius=final_radius,
            nm_per_mL=nm_per_mL,
            injection_time=injection_time,
            initial_rxn_vol=initial_rxn_vol,
        )
        numeric_rows = [
            "Estimated radius (nm)",
            "NaTFA (mL)",
            "YAc (mL)",
            "Total Rxn Volume (mL)",
            "% Volume Injected",
        ]
        df_t.loc[numeric_rows] = (
            df_t.loc[numeric_rows]
              .apply(pd.to_numeric, errors="coerce")
              .round(2)
        )
        df_t.loc["Time (min)"] = (
            pd.to_numeric(df_t.loc["Time (min)"], errors="coerce")
              .round(0).astype("Int64")   # keeps blanks as <NA> if any
        )
        
        # ---- Styling: color the % row cells >10 ----
        styled = (
            df_t.style
                .format(precision=2, na_rep="")  # <-- forces 2-decimal display for floats
                .applymap(
                    lambda v: "background-color: lightcoral" if pd.notna(v) and float(v) > 10 else "",
                    subset=pd.IndexSlice["% Volume Injected", :]
                )
        )

        st.subheader("Injection Table")

        st.dataframe(styled, use_container_width=True)

        csv = df_t.to_csv().encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="shelling_injection_table.csv",
            mime="text/csv",
        )

        if warnings:
            st.warning("\n".join(warnings))

    except Exception as e:
        st.error(str(e))

  
