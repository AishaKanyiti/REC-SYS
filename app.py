# --- app.py : Recommender System Streamlit App ---

import streamlit as st
import pickle
import scipy.sparse as sp
import os

# --- Setup Paths ---
BASE_PATH = os.getcwd()

# --- Load Data & Model ---
@st.cache_resource
def load_data():
    UI_csr = sp.load_npz(os.path.join(BASE_PATH, "UI_csr.npz"))

    with open(os.path.join(BASE_PATH, "user_mappings.pkl"), "rb") as f:
        user_id_to_idx, idx_to_user_id = pickle.load(f)

    with open(os.path.join(BASE_PATH, "item_mappings.pkl"), "rb") as f:
        item_id_to_idx, idx_to_item_id = pickle.load(f)

    return UI_csr, user_id_to_idx, idx_to_user_id, item_id_to_idx, idx_to_item_id


@st.cache_resource
def load_model():
    with open(os.path.join(BASE_PATH, "als_model.pkl"), "rb") as f:
        als = pickle.load(f)
    return als


UI_csr, user_id_to_idx, idx_to_user_id, item_id_to_idx, idx_to_item_id = load_data()
als = load_model()

st.success("✅ Model, data, and mappings loaded successfully!")


# --- Get Active Users & Items ---
active_user_idxs = list(set(UI_csr.nonzero()[0]))  # users with at least 1 interaction
active_user_ids = [idx_to_user_id[i] for i in active_user_idxs]

active_item_idxs = list(set(UI_csr.nonzero()[1]))  # items that appear in interactions
active_item_ids = [idx_to_item_id[i] for i in active_item_idxs]


# --- Recommend Function ---
def recommend_cf(user_raw_id, N=10):
    """Generate top-N recommendations for a given user ID (active items only)."""
    if user_raw_id not in user_id_to_idx:
        st.warning(f"⚠️ User {user_raw_id} not found in mappings.")
        return []

    uidx = user_id_to_idx[user_raw_id]

    try:
        ids, _ = als.recommend(uidx, UI_csr[uidx], N=N, recalculate_user=True)
    except IndexError:
        st.warning("⚠️ No recommendations found (IndexError).")
        return []

    # Convert to real item IDs, but keep only active ones
    return [idx_to_item_id[i] for i in ids if idx_to_item_id[i] in active_item_ids]


# --- Streamlit UI ---
st.title("🎯 Recommendation System")

# ✅ Only active users in dropdown
user_input = st.selectbox(
    "Select a User ID:",
    options=sorted(active_user_ids)
)

# ✅ Slider to choose number of recommendations
num_recs = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)

if st.button("Get Recommendations"):
    recs = recommend_cf(user_input, N=num_recs)
    if recs:
        st.write(f"✅ Top {len(recs)} Recommendations for User {user_input}:")
        for i, item in enumerate(recs, start=1):
            st.write(f"{i}. {item}")
    else:
        st.write("No recommendations found for this user.")
