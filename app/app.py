import streamlit as st
import pandas as pd
import joblib
import os

# --- 【超重要】モデルが探している関数をここで定義する ---
# --- IMPORTANT: Custom function definition for model loading ---
# joblib.load requires the custom functions used in the Pipeline 
# to be defined in the main namespace of the loading script.
def add_custom_features(data):
    data = data.copy()
    # 学習時と同じ計算を行う
    # Perform the same calculation as during the training phase
    data['charge_per_tenure'] = data['MonthlyCharges'] / (data['tenure'] + 1)
    return data

# ページ設定
# Page Configuration
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

# モデルの読み込み（この中で joblib.load が動くときに、上の関数が必要になります）
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
# Initialize session state for history to keep data between reruns
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame()

st.title("📱 Telco Churn Prediction (AI App)")
st.markdown("This application uses a trained CatBoost Pipeline to predict customer churn risk.")

# --- 2. 入力フォーム (サイドバー) ---
# --- 2. Input Form (Sidebar) ---
st.sidebar.header("Customer Information")

# 数値データの入力
# Numerical data inputs
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 0.0, 150.0, 50.0)
total_charges = st.sidebar.number_input("Total Charges ($)", value=1000.0)

# カテゴリデータの選択（学習コードの変数を反映）
# Categorical data inputs (mapped to training feature names)
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

# --- 3. 予測実行 ---
# --- 3. Prediction Logic ---
st.subheader("Analysis Result")

# 学習時と同じ DataFrame 構造を作成
# Create a DataFrame with the exact structure used during training
input_df = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Contract': [contract],
    'PaymentMethod': [payment_method],
    'InternetService': [internet_service]
})

if st.button("Run Prediction"):
    # Pipelineが 'engineering' (charge_per_tenureの追加) も自動で行います
    # The Pipeline automatically runs the 'engineering' step (add_custom_features)
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    # 履歴保存用のデータを作成
    # Create a result summary for history tracking
    new_record = input_df.copy()
    new_record['Probability'] = f"{probability:.1%}"
    new_record['Prediction'] = "Churn" if prediction == 1 else "Stay"
    
    # 履歴を更新（新しい順に上に表示）
    # Update history (concatenate new record to the top)
    st.session_state.history = pd.concat([new_record, st.session_state.history], ignore_index=True)

    st.divider()
    
    # 予測結果の表示
    # Display the prediction results
    if prediction == 1:
        st.error(f"### ⚠️ High Churn Risk (Probability: {probability:.1%})")
        st.progress(probability)
        st.write("This customer is likely to churn. Immediate follow-up or retention offers are recommended.")
    else:
        st.success(f"### ✅ Low Churn Risk (Probability: {probability:.1%})")
        st.progress(probability)
        st.write("This customer is likely to remain stable.")

# --- 4. 予測履歴の表示 ---
# --- 4. Prediction History Display ---
if not st.session_state.history.empty:
    st.divider()
    st.subheader("📊 Prediction History")
    st.dataframe(st.session_state.history, use_container_width=True)
    
    # 履歴のリセットボタン
    if st.button("Clear History"):
        st.session_state.history = pd.DataFrame()
        st.rerun()

# デバッグ用：入力データの確認
# Debugging: Show the raw input data provided to the model
with st.expander("Show input data details"):
    st.write(input_df)