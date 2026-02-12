import streamlit as st

def load_basic_theme():
    """
    Standard professional look.
    """
    st.markdown("""
        <style>
            html, body, [class*="css"] {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .stButton > button {
                border-radius: 4px;
            }
        </style>
    """, unsafe_allow_html=True)
