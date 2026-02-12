import streamlit as st
import pandas as pd
from digiqual.sampling import generate_lhs
from utils import load_basic_theme

# Page configuration for a clean, wide layout
st.set_page_config(page_title="Design Tool", layout="wide")
load_basic_theme()

with st.sidebar:
    st.title("DigiQual")
    st.caption("v1.0.0 | Simulation Reliability")
    st.divider()

st.title(":material/grid_on: Sample Space Generator")
st.markdown("""
Use this tool to create a space-filling **Latin Hypercube Sampling (LHS)** design.
This ensures your simulation covers the entire parameter space efficiently.
""")

st.divider()

col_config, col_preview = st.columns([1, 2])

with col_config:
    with st.container(border=True):
        st.subheader("1. Define Variables")

        # We use st.data_editor to allow users to add/remove variables easily
        default_vars = pd.DataFrame([
            {"Name": "Length", "Min": 0.0, "Max": 10.0},
            {"Name": "Angle", "Min": -45.0, "Max": 45.0}
        ])

        st.write("Specify the ranges for your simulation parameters:")
        var_df = st.data_editor(
            default_vars,
            num_rows="dynamic",
            column_config={
                "Min": st.column_config.NumberColumn(format="%.2f"),
                "Max": st.column_config.NumberColumn(format="%.2f")
            },
            hide_index=True,
            use_container_width=True
        )

        st.divider()

        st.subheader("2. Settings")
        n_samples = st.number_input("Number of Samples (N)", min_value=10, value=50, step=10)

        generate_btn = st.button("Generate Design", type="primary", use_container_width=True)

with col_preview:
    if generate_btn:
        with st.spinner("Generating LHS design..."):
            try:
                # Call your library function
                design = generate_lhs(n=n_samples, ranges=var_df)

                # Store in session state so it persists during download
                st.session_state['generated_design'] = design
                st.success(f"Successfully generated {n_samples} points.")

            except Exception as e:
                st.error(f"Generation failed: {e}")

    # Display results if they exist in the session
    if 'generated_design' in st.session_state:
        with st.container(border=True):
            st.subheader("Design Preview")
            df = st.session_state['generated_design']

            st.dataframe(df, use_container_width=True, height=450)

            # Standard Download Button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Design CSV",
                data=csv,
                file_name="lhs_design.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        with st.container(border=True):
            st.info("Configure your variables and click 'Generate Design' to see the preview.")
