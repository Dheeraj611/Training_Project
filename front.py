import streamlit as st
import requests


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Caterer AI",
    page_icon="🍽️",
    layout="wide"
)


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://127.0.0.1:8000"


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🍽️ Caterer AI Search")
st.write(
    "Find caterers based on location, budget, specialization, "
    "guest capacity, rating, and other requirements."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Search Settings")

    top_k = st.slider(
        "Number of results",
        min_value=1,
        max_value=10,
        value=3
    )

    st.divider()

    st.info(
        "Powered by FastAPI + FAISS + "
        "Sentence Transformers + Groq"
    )


# ============================================================
# QUERY INPUT
# ============================================================

query = st.text_input(
    "🔎 What are you looking for?",
    placeholder=(
        "Example: Find wedding caterers in Karnal "
        "for 500 guests with low budget"
    )
)


# ============================================================
# SEARCH BUTTON
# ============================================================

if st.button(
    "🔍 Search Caterers",
    type="primary",
    use_container_width=True
):

    if not query.strip():

        st.warning(
            "Please enter a search query."
        )

    else:

        with st.spinner(
            "Finding the best caterers..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/search",
                    json={
                        "query": query,
                        "top_k": top_k
                    },
                    timeout=120
                )

                # ------------------------------------------------
                # Check API response
                # ------------------------------------------------

                if response.status_code != 200:

                    try:
                        error = response.json()

                        st.error(
                            error.get(
                                "detail",
                                "API request failed."
                            )
                        )

                    except Exception:

                        st.error(
                            f"API Error: "
                            f"{response.status_code}"
                        )

                else:

                    data = response.json()

                    # ==================================================
                    # SUMMARY
                    # ==================================================

                    st.subheader("📋 Summary")

                    st.write(
                        data.get(
                            "summary",
                            "No summary available."
                        )
                    )

                    # ==================================================
                    # RESULTS COUNT
                    # ==================================================

                    total = data.get(
                        "total_found",
                        0
                    )

                    st.success(
                        f"Found {total} caterer(s)"
                    )

                    # ==================================================
                    # CATERER RESULTS
                    # ==================================================

                    caterers = data.get(
                        "caterers",
                        []
                    )

                    if not caterers:

                        st.warning(
                            "No caterers found for your query."
                        )

                    else:

                        for i, caterer in enumerate(
                            caterers,
                            start=1
                        ):

                            with st.container(
                                border=True
                            ):

                                st.subheader(
                                    f"{i}. "
                                    f"{caterer.get('name', 'Unknown Caterer')}"
                                )

                                # ----------------------------------
                                # Main information
                                # ----------------------------------

                                col1, col2, col3 = st.columns(3)

                                with col1:

                                    st.write(
                                        "📍 **Location**"
                                    )

                                    st.write(
                                        caterer.get(
                                            "location",
                                            "N/A"
                                        )
                                    )

                                with col2:

                                    st.write(
                                        "⭐ **Rating**"
                                    )

                                    st.write(
                                        caterer.get(
                                            "rating",
                                            "N/A"
                                        )
                                    )

                                with col3:

                                    st.write(
                                        "💰 **Budget**"
                                    )

                                    st.write(
                                        caterer.get(
                                            "budget_tier",
                                            "N/A"
                                        )
                                    )

                                # ----------------------------------
                                # Additional information
                                # ----------------------------------

                                col1, col2, col3 = st.columns(3)

                                with col1:

                                    st.write(
                                        "🍽️ **Specialization**"
                                    )

                                    st.write(
                                        caterer.get(
                                            "specialization",
                                            "N/A"
                                        )
                                    )

                                with col2:

                                    st.write(
                                        "👥 **Guest Capacity**"
                                    )

                                    st.write(
                                        caterer.get(
                                            "guest_capacity",
                                            "N/A"
                                        )
                                    )

                                with col3:

                                    st.write(
                                        "🎯 **Similarity**"
                                    )

                                    score = caterer.get(
                                        "similarity_score",
                                        0
                                    )

                                    st.write(
                                        f"{score:.4f}"
                                        if isinstance(
                                            score,
                                            (int, float)
                                        )
                                        else score
                                    )

                                # ----------------------------------
                                # Contact
                                # ----------------------------------

                                if caterer.get(
                                    "contact_phone"
                                ):

                                    st.write(
                                        "📞 **Phone:** "
                                        + str(
                                            caterer[
                                                "contact_phone"
                                            ]
                                        )
                                    )

                                if caterer.get(
                                    "website"
                                ):

                                    st.write(
                                        "🌐 **Website:** "
                                    )

                                    st.link_button(
                                        "Visit Website",
                                        caterer[
                                            "website"
                                        ]
                                    )

                                # ----------------------------------
                                # Match reason
                                # ----------------------------------

                                if caterer.get(
                                    "match_reason"
                                ):

                                    st.info(
                                        "💡 "
                                        + str(
                                            caterer[
                                                "match_reason"
                                            ]
                                        )
                                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Cannot connect to FastAPI.\n\n"
                    "Make sure your FastAPI server is running."
                )

                st.code(
                    "uvicorn api:app --reload"
                )

            except requests.exceptions.Timeout:

                st.error(
                    "⏳ The request took too long. "
                    "Please try again."
                )

            except Exception as e:

                st.error(
                    f"❌ Unexpected error: {e}"
                )