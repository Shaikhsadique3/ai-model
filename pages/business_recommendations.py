import streamlit as st
import pandas as pd

st.set_page_config(page_title="Business Recommendations", page_icon="💡", layout="wide")

st.title("💡 Business Recommendations")

if 'processed_df' in st.session_state:
    processed_df = st.session_state['processed_df']

    st.subheader("Recommendations Based on Churn Risk")

    # Define churn risk thresholds
    high_risk_threshold = 0.7
    medium_risk_threshold = 0.3

    # Filter customers by churn risk
    high_risk_customers = processed_df[processed_df['churn_prediction_xgb'] >= high_risk_threshold]
    medium_risk_customers = processed_df[(processed_df['churn_prediction_xgb'] >= medium_risk_threshold) & (processed_df['churn_prediction_xgb'] < high_risk_threshold)]
    low_risk_customers = processed_df[processed_df['churn_prediction_xgb'] < medium_risk_threshold]

    st.write(f"Total customers analyzed: {len(processed_df)}")
    st.write(f"High-risk customers: {len(high_risk_customers)}")
    st.write(f"Medium-risk customers: {len(medium_risk_customers)}")
    st.write(f"Low-risk customers: {len(low_risk_customers)}")

    with st.expander("High-Risk Customers Recommendations"):
        st.write(f"**Number of High-Risk Customers:** {len(high_risk_customers)}")
        st.markdown("""
        - **Proactive Outreach:** Immediately engage with these customers through personalized calls or dedicated support.
        - **Exclusive Offers:** Provide highly attractive, personalized incentives (e.g., significant discounts, premium feature upgrades, loyalty bonuses) to retain them.
        - **Issue Resolution:** Prioritize and resolve any outstanding issues or complaints they might have.
        - **Feedback Collection:** Conduct in-depth interviews to understand their pain points and reasons for potential churn.
        - **Dedicated Account Management:** Assign a dedicated account manager to build a stronger relationship.
        """)

    with st.expander("Medium-Risk Customers Recommendations"):
        st.write(f"**Number of Medium-Risk Customers:** {len(medium_risk_customers)}")
        st.markdown("""
        - **Targeted Campaigns:** Send personalized email campaigns highlighting new features, success stories, or relevant use cases.
        - **Value Reinforcement:** Showcase the value they are getting from the product through usage reports or personalized dashboards.
        - **Engagement Programs:** Offer webinars, tutorials, or workshops to improve their product utilization.
        - **Feedback Surveys:** Deploy targeted surveys to gather feedback and identify areas for improvement.
        - **Tiered Incentives:** Offer moderate incentives for continued engagement or upgrades.
        """)

    with st.expander("Low-Risk Customers Recommendations"):
        st.write(f"**Number of Low-Risk Customers:** {len(low_risk_customers)}")
        st.markdown("""
        - **Customer Loyalty Programs:** Reward their loyalty with exclusive content, early access to features, or community recognition.
        - **Upselling/Cross-selling:** Identify opportunities to introduce them to higher-tier plans or complementary products.
        - **Referral Programs:** Encourage them to refer new customers with attractive referral bonuses.
        - **Regular Communication:** Maintain consistent, valuable communication through newsletters and product updates.
        - **Advocacy Programs:** Encourage them to become brand advocates through testimonials or case studies.
        """)
else:
    st.warning("Please run the prediction on the 'Predict' page first to generate customer data.")