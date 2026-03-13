# 1. 軽量なPython 3.9イメージを使用
# 1. Use an official lightweight Python 3.9 image
FROM python:3.9-slim

# 2. コンテナ内の作業ディレクトリを設定
# 2. Set the working directory inside the container
WORKDIR /app

# 3. ルートディレクトリにある requirements.txt をコピー
# 3. Copy the requirements file from the root directory
COPY requirements.txt .

# 4. ライブラリをインストール
# 4. Install the necessary Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 5. app フォルダと models フォルダをコンテナにコピー
# 5. Copy the app and models directories into the container
COPY app/ ./app/
COPY models/ ./models/

# 6. Streamlitのデフォルトポート 8501 を開放
# 6. Expose the default Streamlit port
EXPOSE 8501

# 7. アプリを起動するコマンド
# 7. Command to run the Streamlit app when the container starts
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]