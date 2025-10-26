import streamlit as st

st.set_page_config(
    page_title="Churn Prediction App",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("Welcome to the Churn Prediction Dashboard!")
st.write("Please select a page from the sidebar to get started.")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    This dashboard helps you predict customer churn and gain valuable insights.

    **👈 Select a page from the sidebar** to get started.

    ### What's inside?
    - **Predict:** Upload your customer data to get churn predictions.
    - **Insights:** Explore visualizations and key performance indicators related to churn.
    - **Business Recommendations:** Get actionable recommendations based on churn risk.
    """
)