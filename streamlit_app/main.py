import streamlit as st
import pandas as pd
from utils.auth import log_in, log_out, hide_pages
from pathlib import Path

# Config path
PATH_DATA = Path("data")
PATH_IMAGE = Path("image")
df_reviews = pd.read_parquet(PATH_DATA / "reviews.parquet", engine="auto")
list_id = df_reviews["AuthorId"].drop_duplicates().tolist()



st.set_page_config(
    page_title="Your App",
    layout="wide",  # หรือ "centered"
    initial_sidebar_state="collapsed"  # <<<< ปิด sidebar ตอนเริ่มต้น
)

# st.logo(PATH_IMAGE / "shopping_basket.png")

# หน้า login
st.title("🔐 Welcome to Recipe Recommender")

if not st.session_state.get("logged_in", False):
    hide_pages(["shopping.py"])
    with st.form("login_form"):
        st.text_input("User ID", key="email")
        submitted = st.form_submit_button("Login")

        if submitted:
            user_input = st.session_state["email"]
            if user_input.isdigit() and int(user_input) in list_id:
                st.session_state["user_id"] = int(user_input)
                log_in("pages/shopping.py")
            else:
                st.error("❌ Invalid ID. Please try again.")
else:
    st.write("✅ Logged in!")
    st.button("Log out", on_click=log_out)
