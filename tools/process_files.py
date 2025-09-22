import streamlit as st
import os
import pandas as pd
from matplotlib.colors import LogNorm
import io
from utils import integrate_sif, plot_brightness, plot_histogram 


@st.cache_data
def process_files(uploaded_files, region, threshold=1,
                  signal="UCNP", 
                  pix_size_um=0.1, 
                  sig_threshold=0.3,
                  min_fit_separation_px=3,
                  min_r2=0.85):
    """
    Processes all uploaded .sif files and returns a dictionary of dataframes
    and images, plus a single combined dataframe for the histogram.
    """
    processed_data = {}
    all_dfs = []
    
    # Use a temporary directory to store files
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            df, image_data_cps = integrate_sif(
                file_path,
                region=region,
                threshold=threshold,
                signal=signal,
                pix_size_um=pix_size_um,
                sig_threshold=sig_threshold,
                min_fit_separation_px=min_fit_separation_px,
                min_r2=min_r2,
            )
            processed_data[uploaded_file.name] = {
                "df": df,
                "image": image_data_cps,
            }
            all_dfs.append(df)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
            
    # Combine all dataframes into a single one for the histogram
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
    else:
        combined_df = pd.DataFrame()
        
    return processed_data, combined_df
