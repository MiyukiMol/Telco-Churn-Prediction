from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import skops.io as sio  # 2026年の標準：セキュリティ重視のSkops
from google import genai
import sys

# Custom feature engineering for model pipeline, required during Skops loading.
def add_custom_features(data):
    """
    Custom feature engineering function required for model loading.
    This calculates 'charge_per_tenure' and is invoked by the Skops/Pipeline.
    
    Args:
        data (pd.DataFrame): Input customer data.
    Returns:
        pd.DataFrame: Data with additional features.
    """
    data = data.copy()
    # Add +1 to tenure to avoid division by zero
    data['charge_per_tenure'] = data['MonthlyCharges'] / (data['tenure'] + 1)
    return data

# インポートしたこのファイル自身を「メイン」として認識させる
sys.modules['__main__'].add_custom_features = add_custom_features

# FastAPIの初期化
app = FastAPI(
    title="MiyukiMol AI Platform: Churn Prediction API",
    description="CatBoostによる予測とGemini 3.1による戦略提案を統合したAPI",
    version="2.0.0"
)

# --- 1. モデルとGeminiの設定 ---
MODEL_PATH = "models/final_churn_model.skops"
api_key = os.getenv("GEMINI_API_KEY")



# Gemini クライアントの初期化
client = genai.Client(api_key=api_key) if api_key else None

# モデルのロード (Skops によるセキュアロード)
pipeline = None
if os.path.exists(MODEL_PATH):
    try:
        # 信頼できる型を自動取得してロード
        unknown_types = sio.get_untrusted_types(file=MODEL_PATH)
        pipeline = sio.load(MODEL_PATH, trusted=unknown_types)
        print(f"✅ Securely loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model with Skops: {e}")
else:
    print(f"⚠️ Model file not found at {MODEL_PATH}")

# --- 2. データ形式の定義 (Pydantic) ---
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    PaymentMethod: str
    InternetService: str

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 1,
                "MonthlyCharges": 70.0,
                "TotalCharges": 70.0,
                "Contract": "Month-to-month",
                "PaymentMethod": "Electronic check",
                "InternetService": "Fiber optic"
            }
        }

# --- 3. エンドポイントの実装 ---

@app.post("/predict")
async def predict_churn(customer: CustomerData):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Prediction engine is offline.")

    # 1. データ変換と予測
    input_df = pd.DataFrame([customer.dict()])
    
    # しきい値 0.45 を適用して判定
    probability = float(pipeline.predict_proba(input_df)[0][1])
    prediction = 1 if probability > 0.45 else 0

    # 2. Gemini による戦略アドバイスの生成
    advice = "No advice requested for low-risk customers."
    
    if prediction == 1:
        if not client:
            advice = "Strategic advice is unavailable (API Key missing)."
        else:
            try:
                prompt = f"""
                You are a senior customer retention consultant. 
                A customer has a {probability:.1%} churn risk. 
                Profile: {customer.dict()}
                
                Provide 3 high-impact, actionable retention strategies in English. 
                Focus on the 'Contract' and 'MonthlyCharges' aspects.
                Keep it concise and professional.
                """
                response = client.models.generate_content(
                    model='gemini-3.1-flash-lite-preview', 
                    contents=prompt
                )
                advice = response.text
            except Exception as e:
                advice = f"Consultant AI is busy: {str(e)}"

    return {
        "prediction": "High Risk" if prediction == 1 else "Low Risk",
        "probability": round(probability, 4),
        "advice": advice
    }

@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_loaded": pipeline is not None,
        "gemini_active": client is not None
    }