import streamlit as st
import pandas as pd
import joblib
import os
import requests  # バックエンドと通信するために必要 / Required for backend communication

# --- 1. モデル読み込み用のカスタム関数 ---
# --- 1. Custom function for model loading ---
def add_custom_features(data):
    # Pipelineが期待する特徴量を計算します
    # Calculates features expected by the Pipeline
    data = data.copy()
    data['charge_per_tenure'] = data['MonthlyCharges'] / (data['tenure'] + 1)
    return data

# ページ設定
# Page Configuration
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

# モデルの読み込み設定
# Model loading configuration
model_path = "models/final_churn_model.joblib"

@st.cache_resource
def load_model():
    """学習済みモデルをロードします / Loads the pre-trained model"""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

pipeline = load_model()

if pipeline is None:
    st.error(f"Model file not found at: {model_path}")
    st.stop()

# セッション状態で履歴を保持
# Maintain prediction history in session state
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame()

st.title("📱 Telco Churn Prediction (AI App)")
st.markdown("This application uses a trained CatBoost Pipeline and Gemini AI for analysis.")

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
# --- 3. Prediction Execution ---
st.subheader("Analysis Result")

# バックエンドに送るためのデータ構造
# Data structure to send to the backend
input_data = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Contract': contract,
    'PaymentMethod': payment_method,
    'InternetService': internet_service
}

if st.button("Run Prediction"):
    # バックエンド(FastAPI)を呼び出して、予測とGeminiのアドバイスを取得
    # Call backend (FastAPI) to get prediction and Gemini's advice
    with st.spinner('AI Advisor is thinking...'):
        try:
            # Dockerネットワーク内のURLを使用
            # Use the URL within the Docker network
            response = requests.post("http://backend:8000/predict", json=input_data)
            result_json = response.json()
            
            prediction = result_json['prediction']
            probability = result_json['probability']
            gemini_advice = result_json.get('advice', "No advice available.")
        except Exception as e:
            st.error(f"Backend connection error: {e}")
            # エラー時のフォールバック: ローカルで予測のみ実行
            # Fallback: Execute local prediction only on error
            input_df = pd.DataFrame([input_data])
            prediction = pipeline.predict(input_df)[0]
            probability = pipeline.predict_proba(input_df)[0][1]
            gemini_advice = "Could not connect to AI Advisor."

    # 履歴を保存
    # Save to history
    new_record = pd.DataFrame([input_data])
    new_record['Probability'] = f"{probability:.1%}"
    new_record['Prediction'] = "Churn" if prediction == 1 else "Stay"
    st.session_state.history = pd.concat([new_record, st.session_state.history], ignore_index=True)

    st.divider()

    # 判定結果に基づいて色を設定
    # Set color based on the result
    chart_color = "#EF5A5A" if prediction == 1 else "#29B5E8"

    # レイアウト作成
    # Create layout
    col1, col2 = st.columns([1, 1])

    with col1:
        # 予測結果の表示
        # Display prediction results
        if prediction == 1:
            st.error(f"### ⚠️ High Churn Risk (Probability: {probability:.1%})")
            st.progress(probability)
        else:
            st.success(f"### ✅ Low Churn Risk (Probability: {probability:.1%})")
            st.progress(probability)

        # --- Gemini のアドバイスを表示 ---
        # --- Display Gemini's Advice ---
        st.markdown("---")
        st.subheader("🤖 AI Advisor (Gemini)")
        st.write(gemini_advice)

    with col2:
        # 判断根拠のグラフ表示
        # Display feature importance chart
        st.write("**🧐 Why did the AI make this decision?**")
        raw_importances = pipeline.named_steps['classifier'].get_feature_importance()
        feature_names = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaymentMethod', 'InternetService', 'charge_per_tenure']
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': raw_importances
        }).sort_values(by='Importance', ascending=True)

        st.bar_chart(
            importance_df.set_index('Feature'), 
            color=chart_color,
            height=300
        )

# --- 4. 予測履歴の表示 ---
# --- 4. Prediction History Display ---
if not st.session_state.history.empty:
    st.divider()
    st.subheader("📊 Prediction History")
    # 警告を避けるため width="stretch" を使用
    # Use width="stretch" to avoid warnings
    st.dataframe(st.session_state.history, width="stretch")
    
    if st.button("Clear History"):
        st.session_state.history = pd.DataFrame()
        st.rerun()