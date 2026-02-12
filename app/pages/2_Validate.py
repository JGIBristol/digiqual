import streamlit as st
import pandas as pd
from digiqual.core import SimulationStudy
from utils import load_basic_theme

st.set_page_config(page_title="Validate", layout="wide")
load_basic_theme()

with st.sidebar:
    st.title("DigiQual")
    st.caption("v1.0.0 | Simulation Reliability")
    st.divider()

st.title(":material/fact_check: Validation & Diagnostics")

# --- DATA LOADING ---
uploaded_files = st.file_uploader(
    "Upload Simulation CSVs",
    accept_multiple_files=True,
    type="csv"
)

if uploaded_files:
    df_list = [pd.read_csv(f) for f in uploaded_files]
    st.session_state['shared_df'] = pd.concat(df_list, ignore_index=True)
    st.success(f"Loaded {len(st.session_state['shared_df'])} rows.")

# --- DIAGNOSTICS ---
if 'shared_df' in st.session_state:
    df = st.session_state['shared_df']

    col_in, col_out = st.columns([1, 2])

    with col_in:
        with st.container(border=True):
            st.subheader("Settings")
            all_cols = df.columns.tolist()
            inputs = st.multiselect("Inputs", all_cols, default=all_cols[:-1])
            outcome = st.selectbox("Outcome", all_cols, index=len(all_cols)-1)

            run_diag = st.button("Run Check", type="primary", use_container_width=True)

    with col_out:
        if run_diag:
            study = SimulationStudy(input_cols=inputs, outcome_col=outcome)
            study.add_data(df)
            report = study.diagnose()

            st.subheader("Diagnostic Results")
            st.dataframe(report, use_container_width=True)

            if report['Pass'].all():
                st.success("Dataset is stable, continue to Analyse.")

            if not report['Pass'].all():
                st.warning("Data refinement suggested.")
                n_new = st.number_input("New samples", value=10)
                new_data = study.refine(n_points=n_new)
                st.download_button("Download New Samples", new_data.to_csv(index=False), "refinement.csv")
else:
    st.info("Please upload data to begin.")
