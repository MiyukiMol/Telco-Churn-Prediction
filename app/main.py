from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from google import genai  # 最新のGoogle GenAI SDK / Latest Google GenAI SDK
import sys
import app.main as main_module

# --- 1. モデル読み込み用のカスタム関数 ---
# --- 1. Custom function for model loading ---
def add_custom_features(data):
    # Pipelineが期待する特徴量エンジニアリングを定義します
    # Define the feature engineering expected by the Pipeline
    data = data.copy()
    data['charge_per_tenure'] = data['MonthlyCharges'] / (data['tenure'] + 1)
    return data

# FastAPIの初期化
# Initialize FastAPI
app = FastAPI(title="Telco Churn Prediction API")

# --- 2. モデルとGeminiの設定 ---
# --- 2. Model and Gemini Configuration ---
MODEL_PATH = "models/final_churn_model.joblib"

# 環境変数からAPIキーを取得
# Retrieve API key from environment variables
api_key = os.getenv("GEMINI_API_KEY")
client = None

if api_key:
    # 最新のGoogle GenAIクライアントを初期化
    # Initialize the latest Google GenAI client
    client = genai.Client(api_key=api_key)
    print("✅ Gemini Client initialized.")
else:
    print("⚠️ WARNING: GEMINI_API_KEY environment variable is not set.")

try:
    # 保存された学習済みモデルをロード
    # Load the pre-trained model pipeline
    sys.modules['__main__'] = main_module
    pipeline = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    pipeline = None

# --- 3. データ形式の定義 (Pydantic) ---
# --- 3. Data Schema Definition (Pydantic) ---
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    PaymentMethod: str
    InternetService: str

# --- 4. 予測 & AIアドバイスのエンドポイント ---
# --- 4. Prediction & AI Advice Endpoint ---
@app.post("/predict")
async def predict_churn(customer: CustomerData):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 入力データをDataFrameに変換
    # Convert input data to DataFrame
    input_dict = customer.dict()
    input_df = pd.DataFrame([input_dict])
    
    # 1. 解約予測の実行
    # 1. Execute Churn Prediction
    prediction = int(pipeline.predict(input_df)[0])
    probability = float(pipeline.predict_proba(input_df)[0][1])

    # 2. Gemini による具体的な対策案の生成
    # 2. Generate specific retention strategies using Gemini
    advice = "No advice generated."
    
    # 解約リスクが高い場合(1)、かつクライアントが準備できている場合に実行
    # Execute only if prediction is 1 (High Risk) and client is initialized
    if prediction == 1 and client: 
        try:
            # プロンプトの定義
            # Define the prompt
            prompt = f"""
            You are a customer retention expert for a telecom company.
            A customer has been identified with a {probability:.1%} churn risk.
            
            Customer Profile:
            - Tenure: {customer.tenure} months
            - Monthly Charges: ${customer.MonthlyCharges}
            - Contract: {customer.Contract}
            - Internet Service: {customer.InternetService}

            Please provide 3 specific, professional, and actionable suggestions in English to prevent this customer from leaving.
            Keep the tone professional and the advice practical.
            """
            
            # Gemini 2.0 Flash を使用して回答を生成
            # Generate response using Gemini 2.0 Flash
            response = client.models.generate_content(
                model='gemini-2.0-flash', 
                contents=prompt
            )
            advice = response.text
        except Exception as e:
            advice = f"AI Advice is currently unavailable: {str(e)}"
    elif not client:
        advice = "Gemini API key is missing. Please set GEMINI_API_KEY."

    # 結果を返す
    # Return results
    return {
        "prediction": prediction,
        "probability": probability,
        "advice": advice
    }

@app.get("/")
def home():
    return {"message": "Telco Churn API is running!"}