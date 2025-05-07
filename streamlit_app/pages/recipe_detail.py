import streamlit as st
import pandas as pd
from pathlib import Path
from utils.auth import log_in, log_out, hide_pages
from time import sleep

# Config path
PATH_DATA = Path("data")
df_reviews = pd.read_csv(PATH_DATA / "reviews.csv")
list_id = df_reviews["AuthorId"].drop_duplicates().tolist()

df_reciped = pd.read_csv(PATH_DATA / "recipes.csv")
df_reciped = df_reciped[["RecipeId", "Name", "RecipeCategory", "RecipeIngredientParts", "RecipeIngredientQuantities", "RecipeInstructions", "Images"]].copy()
df_reciped["RecipeId"] = df_reciped["RecipeId"].astype("int")
df_reciped["Images"] = df_reciped["Images"].str[0]

df_reciped_cosine = pd.read_parquet(PATH_DATA / "cosine_similarity_result", engine="auto")
df_reciped_cosine = df_reciped_cosine[["recipeid", "firstsuggestid", "firstsuggestname"]]
df_reciped_cosine["recipeid"] = df_reciped_cosine["recipeid"].astype("int")
df_reciped_cosine["firstsuggestid"] = df_reciped_cosine["firstsuggestid"].astype("int")
df_reciped_cosine = df_reciped_cosine.merge(
    df_reciped[["RecipeId", "Images", "RecipeCategory", "RecipeIngredientParts", "RecipeIngredientQuantities", "RecipeInstructions"]],
    how="inner",
    left_on=['firstsuggestid'],
    right_on=['RecipeId']
)

df_als_recommend_result = pd.read_parquet(PATH_DATA / "ALS_recommend_result", engine="auto")
df_als_recommend_result = df_als_recommend_result[["authorid", "recipeid1", "name1"]]
df_als_recommend_result["authorid"] = df_als_recommend_result["authorid"].astype("int")
df_als_recommend_result["recipeid1"] = df_als_recommend_result["recipeid1"].astype("int")
df_als_recommend_result = df_als_recommend_result.merge(
    df_reciped[["RecipeId", "Images", "RecipeCategory", "RecipeIngredientParts", "RecipeIngredientQuantities", "RecipeInstructions"]],
    how="inner",
    left_on=['recipeid1'],
    right_on=['RecipeId']
)

st.set_page_config(page_title="Your App", layout="wide", initial_sidebar_state="collapsed")

if not st.session_state.get("logged_in", False):
    st.warning("🔒 Access denied. Redirecting to login...")
    st.switch_page("main.py")

selected_id = st.session_state.get("selected_recipe_id")
user_id = st.session_state.get("user_id")

def display_image_block(recipe, title, name_key="Name", id_key="RecipeId"):
    st.markdown(f"## {title}")
    st.markdown(f"### {recipe[name_key]}")
    st.markdown(f"**Category**: {recipe['RecipeCategory']}")
    st.markdown(f"**Recipe ID**: {recipe[id_key]}")
    
    if isinstance(recipe["Images"], str) and recipe["Images"].startswith("http"):
        st.image(recipe["Images"], width=400)
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)  # กันพื้นที่ให้สมดุล
    else:
        st.warning("📷 No image available for this recipe.")
        st.markdown("<div style='height: 320px'></div>", unsafe_allow_html=True)

def display_ingedient_block(recipe, title, name_key="Name", id_key="RecipeId"):
    st.markdown("#### Ingredient")
    ingredients = recipe["RecipeIngredientParts"].tolist()
    quantities = recipe["RecipeIngredientQuantities"].tolist()

    if isinstance(ingredients, list) and isinstance(quantities, list):
        max_len = max(len(ingredients), len(quantities))
        
        # เติม list ให้มีความยาวเท่ากัน
        ingredients += ["not have information"] * (max_len - len(ingredients))
        quantities += ["not have information"] * (max_len - len(quantities))

        combined_df = pd.DataFrame({
            "Ingredient": ingredients,
            "Quantity": quantities
        })
        st.dataframe(combined_df)
    else:
        st.warning("❗ Ingredients and quantities are not properly formatted.")

def display_instruction_block(recipe, title, name_key="Name", id_key="RecipeId"):
    st.markdown("#### Instruction")
    st.dataframe(recipe["RecipeInstructions"])
    st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if selected_id is not None:
        recipe = df_reciped[df_reciped["RecipeId"] == selected_id].iloc[0]
        display_image_block(recipe, "Recipe Selected")
    else:
        st.warning("⚠️ No recipe selected. Please go back to homepage.")
        st.markdown("<div style='height: 600px'></div>", unsafe_allow_html=True)

with col2:
    if selected_id is not None:
        recipe = df_reciped_cosine[df_reciped_cosine["recipeid"] == selected_id].iloc[0]
        display_image_block(recipe, "Recipe Recommend #1", name_key="firstsuggestname", id_key="firstsuggestid")
    else:
        st.warning("⚠️ No recipe selected. Please go back to homepage.")
        st.markdown("<div style='height: 600px'></div>", unsafe_allow_html=True)

with col3:
    if user_id is not None:
        user_id = int(user_id)
        if not df_als_recommend_result[df_als_recommend_result["authorid"] == user_id].empty:
            recipe = df_als_recommend_result[df_als_recommend_result["authorid"] == user_id].iloc[0]
            display_image_block(recipe, "Recipe Recommend #2", name_key="name1", id_key="recipeid1")
        else:
            st.warning("❌ No ALS recommendation available for this user.")
            st.markdown("<div style='height: 600px'></div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ No user ID found in session.")
        st.markdown("<div style='height: 600px'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if selected_id is not None:
        recipe = df_reciped[df_reciped["RecipeId"] == selected_id].iloc[0]
        display_ingedient_block(recipe, "Recipe Selected")
    else:
        st.warning("⚠️ No recipe selected. Please go back to homepage.")
        st.markdown("<div style='height: 600px'></div>", unsafe_allow_html=True)

with col2:
    if selected_id is not None:
        recipe = df_reciped_cosine[df_reciped_cosine["recipeid"] == selected_id].iloc[0]
        display_ingedient_block(recipe, "Recipe Recommend #1", name_key="firstsuggestname", id_key="firstsuggestid")
    else:
        st.warning("⚠️ No recipe selected. Please go back to homepage.")
        st.markdown("<div style='height: 600px'></div>", unsafe_allow_html=True)

with col3:
    if user_id is not None:
        user_id = int(user_id)
        if not df_als_recommend_result[df_als_recommend_result["authorid"] == user_id].empty:
            recipe = df_als_recommend_result[df_als_recommend_result["authorid"] == user_id].iloc[0]
            display_ingedient_block(recipe, "Recipe Recommend #2", name_key="name1", id_key="recipeid1")
        else:
            st.warning("❌ No ALS recommendation available for this user.")
            st.markdown("<div style='height: 600px'></div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ No user ID found in session.")
        st.markdown("<div style='height: 600px'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if selected_id is not None:
        recipe = df_reciped[df_reciped["RecipeId"] == selected_id].iloc[0]
        display_instruction_block(recipe, "Recipe Selected")
    else:
        st.warning("⚠️ No recipe selected. Please go back to homepage.")
        st.markdown("<div style='height: 600px'></div>", unsafe_allow_html=True)

with col2:
    if selected_id is not None:
        recipe = df_reciped_cosine[df_reciped_cosine["recipeid"] == selected_id].iloc[0]
        display_instruction_block(recipe, "Recipe Recommend #1", name_key="firstsuggestname", id_key="firstsuggestid")
    else:
        st.warning("⚠️ No recipe selected. Please go back to homepage.")
        st.markdown("<div style='height: 600px'></div>", unsafe_allow_html=True)

with col3:
    if user_id is not None:
        user_id = int(user_id)
        if not df_als_recommend_result[df_als_recommend_result["authorid"] == user_id].empty:
            recipe = df_als_recommend_result[df_als_recommend_result["authorid"] == user_id].iloc[0]
            display_instruction_block(recipe, "Recipe Recommend #2", name_key="name1", id_key="recipeid1")
        else:
            st.warning("❌ No ALS recommendation available for this user.")
            st.markdown("<div style='height: 600px'></div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ No user ID found in session.")
        st.markdown("<div style='height: 600px'></div>", unsafe_allow_html=True)

