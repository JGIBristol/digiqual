import streamlit as st
import matplotlib
matplotlib.use('Agg')
from digiqual.core import SimulationStudy
from utils import load_basic_theme

st.set_page_config(page_title="Analyze", layout="wide")
load_basic_theme()

with st.sidebar:
    st.title("DigiQual")
    st.caption("v1.0.0 | Simulation Reliability")
    st.divider()

st.title(":material/analytics: PoD Analysis")

if 'shared_df' not in st.session_state:
    st.warning("No data found. Please upload data in the Validate tab.")
    st.stop()

df = st.session_state['shared_df']

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    all_cols = df.columns.tolist()

    with c1:
        poi = st.selectbox("X-Axis (POI)", all_cols)
        outcome = st.selectbox("Y-Axis (Signal)", all_cols, index=len(all_cols)-1)
    with c2:
        threshold = st.number_input("Threshold", value=10.0)
    with c3:
        run = st.button("Calculate", type="primary", use_container_width=True)

if run:
    study = SimulationStudy(input_cols=[c for c in all_cols if c != outcome], outcome_col=outcome)
    study.add_data(df)
    study.pod(poi_col=poi, threshold=threshold)
    study.visualise(show=False)

    col_left, col_right = st.columns(2)
    with col_left:
        st.pyplot(study.plots['signal_model'].get_figure())
    with col_right:
        st.pyplot(study.plots['pod_curve'].get_figure())
