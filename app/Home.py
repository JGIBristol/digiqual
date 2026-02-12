import streamlit as st
from utils import load_basic_theme

st.set_page_config(page_title="DigiQual", layout="wide")
load_basic_theme()

with st.sidebar:
    st.title("DigiQual")
    st.caption("v1.0.0 | Simulation Reliability")
    st.divider()

st.title("DigiQual Reliability Toolkit")
st.markdown("Welcome to the **Simulation Reliability Assistant**. Use the sidebar to navigate through the workflow.")



# Navigation via standard UI elements
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.subheader(":material/grid_on: Design")
        st.write("Generate space-filling LHS designs for your simulations.")
        if st.button("Open Design Tool", use_container_width=True):
            st.switch_page("pages/1_Design.py")

with col2:
    with st.container(border=True):
        st.subheader(":material/fact_check: Validate")
        st.write("Check data sufficiency and get refinement suggestions.")
        if st.button("Open Validation Tool", use_container_width=True):
            st.switch_page("pages/2_Validate.py")

with col3:
    with st.container(border=True):
        st.subheader(":material/analytics: Analyse")
        st.write("Calculate PoD curves and reliability metrics.")
        if st.button("Open Analysis Tool", use_container_width=True):
            st.switch_page("pages/3_Analyse.py")
