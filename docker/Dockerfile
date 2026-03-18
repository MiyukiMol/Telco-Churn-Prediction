# 1. 軽量なPython 3.9イメージを使用
# 1. Use an official lightweight Python 3.9 image
FROM python:3.9-slim

# 2. コンテナ内の作業ディレクトリを設定
# 2. Set the working directory inside the container
WORKDIR /app

# 3. 必要なパッケージリストをコピーしてインストール
# 3. Copy the requirements file and install necessary Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. app フォルダと models フォルダをコンテナにコピー
# 4. Copy the app and models directories into the container
COPY app/ ./app/
COPY models/ ./models/

# 5. Hugging Face用のユーザー設定（書き込み権限エラー回避のため）
# 5. Set up a non-root user for Hugging Face Spaces to avoid permission issues
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 6. ポート 7860 を開放（Hugging Faceの規定）
# 6. Expose port 7860 as required by Hugging Face Spaces
EXPOSE 7860

# 7. Streamlitアプリを起動（ポート 7860 を指定）
# 7. Command to run the Streamlit app on port 7860
CMD ["streamlit", "run", "app/app.py", "--server.port=7860", "--server.address=0.0.0.0"]