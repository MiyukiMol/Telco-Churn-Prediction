import streamlit as st

# Set the title of application
st.title("🚀 Telco Churn Prediction: Docker Test")

# show a simple message
st.write("Congratulation! Your Docker container is running Streamlit successfully.")

# サイドバーにステータスを表示
# show a status at sidebar
st.sidebar.success("Environment: Docker + Streamlit")
st.sidebar.info("Next Step: Integrate CatBoost Model")