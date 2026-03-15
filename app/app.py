import streamlit as st
import pandas as pd
import joblib
import os

# --- 【超重要】モデルが探している関数をここで定義する ---
# --- IMPORTANT: Custom function definition for model loading ---
def add_custom_features(data):
    data = data.copy()
    data['charge_per_tenure'] = data['MonthlyCharges'] / (data['tenure'] + 1)
    return data

# ページ設定
# Page Configuration
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

# モデルの読み込み
# Path to the saved model file
model_path = "models/final_churn_model.joblib"

@st.cache_resource
def load_model():
    """Load the trained machine learning pipeline from a joblib file."""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

pipeline = load_model()

if pipeline is None:
    st.error(f"Model file not found at: {model_path}")
    st.stop()

# セッション状態で履歴を保持する
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame()

st.title("📱 Telco Churn Prediction (AI App)")
st.markdown("This application uses a trained CatBoost Pipeline to predict customer churn risk.")

# --- 2. 入力フォーム (サイドバー) ---
# --- 2. Input Form (Sidebar) ---
st.sidebar.header("Customer Information")
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 0.0, 150.0, 50.0)
total_charges = st.sidebar.number_input("Total Charges ($)", value=1000.0)
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

# --- 3. 予測実行 ---
# --- 3. Prediction Logic ---
st.subheader("Analysis Result")

input_df = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Contract': [contract],
    'PaymentMethod': [payment_method],
    'InternetService': [internet_service]
})

if st.button("Run Prediction"):
    # 予測の実行
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    # 履歴保存
    new_record = input_df.copy()
    new_record['Probability'] = f"{probability:.1%}"
    new_record['Prediction'] = "Churn" if prediction == 1 else "Stay"
    st.session_state.history = pd.concat([new_record, st.session_state.history], ignore_index=True)

    st.divider()

    # --- 🆕 判定結果に基づいてグラフの色を決める ---
    # --- 🆕 Determine chart color based on prediction results ---
    # Churn(1)なら赤系、Stay(0)なら青系に設定
    # Set Red for Churn (1), Blue for Stay (0)
    chart_color = "#EF5A5A" if prediction == 1 else "#29B5E8"

    # 特徴量の重要度を取得
    # Get feature importance from CatBoost
    raw_importances = pipeline.named_steps['classifier'].get_feature_importance()
    feature_names = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaymentMethod', 'InternetService', 'charge_per_tenure']
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': raw_importances
    }).sort_values(by='Importance', ascending=True)

    # レイアウトを2カラムに分割
    col1, col2 = st.columns([1, 1])

    with col1:
        # 予測結果の表示
        if prediction == 1:
            st.error(f"### ⚠️ High Churn Risk (Probability: {probability:.1%})")
            st.progress(probability)
            st.write("This customer is likely to churn. Immediate follow-up is recommended.")
        else:
            st.success(f"### ✅ Low Churn Risk (Probability: {probability:.1%})")
            st.progress(probability)
            st.write("This customer is likely to remain stable.")

    with col2:
        # 判断根拠のグラフ表示（動的な色を適用）
        # Display Factor Importance with dynamic color
        st.write("**🧐 Why did the AI make this decision?**")
        st.bar_chart(
            importance_df.set_index('Feature'), 
            color=chart_color, # ここで色を指定！
            height=300
        )
        st.caption("The color represents the current risk level.")

# --- 4. 予測履歴の表示 ---
if not st.session_state.history.empty:
    st.divider()
    st.subheader("📊 Prediction History")
    st.dataframe(st.session_state.history, use_container_width=True)
    
    if st.button("Clear History"):
        st.session_state.history = pd.DataFrame()
        st.rerun()

# デバッグ用
with st.expander("Show input data details"):
    st.write(input_df)