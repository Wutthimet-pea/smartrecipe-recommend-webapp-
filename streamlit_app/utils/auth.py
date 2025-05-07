import streamlit as st
from time import sleep

def hide_pages(pages_to_hide):
    st.markdown(
        f"""
        <style>
        {''.join([f'[data-testid="stSidebarNav"] a[href$="{page}"] {{ display: none; }}' for page in pages_to_hide])}
        </style>
        """,
        unsafe_allow_html=True
    )

def log_in(target_page):
    st.session_state["logged_in"] = True
    st.success("Logged in!")
    sleep(0.5)
    st.switch_page(target_page)

def log_out():
    st.session_state["logged_in"] = False
    st.success("Logged out!")
    sleep(0.5)
    st.switch_page("main.py")
