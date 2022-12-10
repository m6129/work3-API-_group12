from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel #pydantic, занимается автоматической проверкой формата и типа данных

class Item(BaseModel):# импортируем из pydantic базовый вариант класса для моделей BaseModel и создаем на его основе свою модель:
    text: str
app = FastAPI()
classifier = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning") #подключаем модель с HugginFace

@app.get("/")
def root():
    """Эта функция вызывает классификатор"""
    return {"img":'https://img5.goodfon.ru/original/1920x1080/b/2f/tpvvvr-aprvvr-dublin.jpg'}
@app.post("/predict/")
def predict(item: Item):
    """генерируем описание рисунка"""
    return classifier(item.text)[0]
