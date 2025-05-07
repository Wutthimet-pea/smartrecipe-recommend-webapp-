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
df_reciped = df_reciped[["RecipeId", "Name", "RecipeCategory", "Images"]].copy()
df_reciped["RecipeId"] = df_reciped["RecipeId"].astype("int")
df_reciped["Images"] = df_reciped["Images"].str[0]  # extract first image

df_reciped_cosine = pd.read_parquet(PATH_DATA / "cosine_similarity_result", engine="auto")
df_reciped_cosine["recipeid"] = df_reciped_cosine["recipeid"].astype("int")

df_reciped = df_reciped.merge(df_reciped_cosine[['recipeid']], how="inner", left_on=['RecipeId'],right_on=['recipeid']).drop("recipeid", axis = 1)
df_reciped = df_reciped.drop_duplicates()

df_count_top5_cat_review = df_reviews.merge(df_reciped, on = ["RecipeId"], how = "left")
df_count_top5_cat_review = df_count_top5_cat_review.groupby(["RecipeCategory"]).agg(count=pd.NamedAgg(column="AuthorId", aggfunc="count")).sort_values("count", ascending= False).reset_index().head(5)

df_count_top5_menu_by_cat = df_reviews.merge(df_reciped, on = ["RecipeId"], how = "left")
df_count_top5_menu_by_cat = df_count_top5_menu_by_cat.groupby(["RecipeId", "RecipeCategory"]).agg(count=pd.NamedAgg(column="AuthorId", aggfunc="count")).sort_values(["count"], ascending= False)
df_count_top5_menu_by_cat["n"] = df_count_top5_menu_by_cat.groupby(["RecipeCategory"]).cumcount() + 1
df_count_top5_menu_by_cat = df_count_top5_menu_by_cat[df_count_top5_menu_by_cat["n"] <= 5]
df_count_top5_menu_by_cat = df_count_top5_menu_by_cat.merge(df_reciped, on = ["RecipeId"], how = "left")
df_count_top5_menu_by_cat = df_count_top5_menu_by_cat.merge(df_count_top5_cat_review[["RecipeCategory"]], on = ["RecipeCategory"], how = "inner").reset_index(drop = True)

st.set_page_config(page_title="Your App", layout="wide", initial_sidebar_state="collapsed")


if not st.session_state.get("logged_in", False):
    st.warning("🔒 Access denied. Redirecting to login...")
    st.switch_page("main.py")


# 🌟 Main Header
st.markdown("""
# 🍴 Welcome to **Reciped Shopping Mall**
""", unsafe_allow_html=True)

st.markdown("""
### 👇 *Choose the recipe you want to explore, cook, and enjoy!*
""", unsafe_allow_html=True)

# 🔎 Filter by RecipeCategory
category_options = df_reciped["RecipeCategory"].dropna().unique().tolist()
selected_category = st.selectbox("📂 Filter by Category", ["All"] + category_options)

# 🔍 Search by Name
search_query = st.text_input("🔎 Search by Name (e.g., Thai, Rice, etc.):")

# 🧠 Apply filter and search
filtered_df = df_reciped.copy()
if selected_category != "All":
    filtered_df = filtered_df[filtered_df["RecipeCategory"] == selected_category]
if search_query.strip():
    filtered_df = filtered_df[filtered_df["Name"].str.contains(search_query.strip(), case=False, na=False)]

# Add selection column
filtered_df["Select"] = False
edited_df = st.data_editor(
    filtered_df,
    column_config={
        "Images": st.column_config.ImageColumn("Image", width="1000px"),
        "Select": st.column_config.CheckboxColumn("Select One", disabled=False),
    },
    use_container_width=True,
    hide_index=True,
    num_rows="fixed"
)

# Get selected row
selected_rows = edited_df[edited_df["Select"] == True]


# Enforce single selection
if len(selected_rows) > 1:
    st.warning("⚠️ Please select only one recipe.")
elif len(selected_rows) == 1:
    selected_id = selected_rows.iloc[0]["RecipeId"]
    st.session_state["selected_recipe_id"] = selected_id
    if st.button("✅ Confirm Selection"):
        st.success(f"You selected recipe ID: {selected_id}")
        st.switch_page("pages/recipe_detail.py")
else:
    st.info("🔍 Please select a recipe from the table.")

st.markdown("""---""")
st.write("""
## Top 5 Category and Reciped
""")
categories = df_count_top5_menu_by_cat['RecipeCategory'].drop_duplicates().tolist()
cols = st.columns(len(categories))

for i, cat in enumerate(categories):
    with cols[i]:
        st.markdown(f"### {cat}")

        df_top5 = df_count_top5_menu_by_cat[df_count_top5_menu_by_cat['RecipeCategory'] == cat].sort_values("n", ascending=False).head(5)

        for _, row in df_top5.iterrows():
            recipe_id = int(row["RecipeId"])
            name = row["Name"]
            image_url = row["Images"]
            button_key = f"button_{cat}_{recipe_id}"

            # Trigger button (behind-the-scenes)
            clicked = st.button(" ", key=button_key)

            # Display image with title overlay
            st.markdown(
                f"""
                <style>
                    .recipe-card {{
                        position: relative;
                        width: 100%;
                        height: 200px;
                        border-radius: 10px;
                        overflow: hidden;
                        margin-bottom: 20px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.4);
                        cursor: pointer;
                    }}

                    .recipe-card img {{
                        width: 100%;
                        height: 100%;
                        object-fit: cover;
                        border-radius: 10px;
                        display: block;
                    }}

                    .recipe-title-overlay {{
                        position: absolute;
                        bottom: 0;
                        width: 100%;
                        background: rgba(0, 0, 0, 0.6);
                        color: #fff;
                        text-align: center;
                        padding: 8px 6px;
                        font-size: 14px;
                        font-weight: 500;
                    }}
                </style>

                <div class="recipe-card" onclick="document.querySelector('button[data-testid=stButton][key={button_key}]').click()">
                    <img src="{image_url}" onerror="this.style.display='none'" />
                    <div class="recipe-title-overlay">{name}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if clicked:
                st.session_state["selected_recipe_id"] = recipe_id
                st.switch_page("pages/recipe_detail.py")

st.markdown("""---""")
st.write("""
## Just Random Recipe
""")

# ปุ่มสุ่มใหม่
if st.button("🔄 Re-random"):
    st.session_state.random_5 = df_reciped.sample(5)

# เริ่มต้นถ้ายังไม่เคยสุ่ม
if "random_5" not in st.session_state:
    st.session_state.random_5 = df_reciped.sample(5)

# แสดงใน columns แบบ responsive
cols = st.columns(5)  # สร้าง 5 คอลัมน์แนวนอน
for i, (_, row) in enumerate(st.session_state.random_5.iterrows()):
    with cols[i]:
        
        # ปุ่มไปหน้า detail
        if st.button(f"{row['Name']}", key=row["RecipeId"]):
            st.session_state["selected_recipe_id"] = row["RecipeId"]
            st.switch_page("pages/recipe_detail.py")

for i, (_, row) in enumerate(st.session_state.random_5.iterrows()):
    with cols[i]:
        if isinstance(row["Images"], str) and row["Images"].startswith("http"):
            st.markdown(
                f"<img src='{row['Images']}' style='width:100%; height:200px; object-fit:cover; border-radius:8px;'>",
                unsafe_allow_html=True
            )
        else:
            st.warning("📷 No image available")
#### คิดไม่ออกกินไรดี

#### สุ่มเมนู




