from fastapi import FastAPI
import gradio as gr
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Загрузка необходимых ресурсов NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = FastAPI()

# Загрузка сохраненных компонентов
MODEL_DIR = "fake_news_detector"
model = load_model(MODEL_DIR + "/model.h5")
tfidf_vectorizer = joblib.load(MODEL_DIR + "/tfidf_vectorizer.pkl")
encoder = joblib.load(MODEL_DIR + "/label_encoder.pkl")

def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def predict_news(text):
    try:
        # Предобработка текста
        processed_text = preprocess_text(text)
        
        # Векторизация
        X = tfidf_vectorizer.transform([processed_text])
        
        # Предсказание
        prediction = model.predict(X.toarray())
        predicted_class = np.argmax(prediction, axis=1)
        confidence = np.max(prediction)
        
        # Декодирование метки
        label = encoder.inverse_transform(predicted_class)[0]
        
        result = {
            'FAKE': 'Фейковая новость',
            'REAL': 'Настоящая новость'
        }.get(label, label)
        
        return f"Результат: {result}, Уверенность: {confidence:.2f}"
    
    except Exception as e:
        return f"Ошибка анализа: {str(e)}"

# Gradio интерфейс
iface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=5, placeholder="Введите текст новости здесь..."),
    outputs="text",
    title="Детектор фейковых новостей",
    description="Проверьте, является ли новость фейковой",
    examples=[
        ["Scientists prove that Earth is flat"],
        ["Annual economic forum held in Moscow"]
    ]
)

# Монтируем Gradio
app = gr.mount_gradio_app(app, iface, path="/")

# API endpoint
@app.get("/api/check_news")
async def check_news(text: str):
    return {"result": predict_news(text)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=7860)