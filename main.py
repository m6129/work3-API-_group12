from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel #pydantic, занимается автоматической проверкой формата и типа данных

class Item(BaseModel):# импортируем из pydantic базовый вариант класса для моделей BaseModel и создаем на его основе свою модель:
    text: str
app = FastAPI()
classifier = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning") #подключаем модель с HuggingFace

@app.get("/")
def root():
    """Эта функция вызывает классификатор"""
    return {"message": "Hello World from group 12"}

#пара примеров картинок:
#https://www.sunhome.ru/i/wallpapers/215/starinnii-zamok.1920x1080.jpg
#https://phonoteka.org/uploads/posts/2022-09/1663900455_4-phonoteka-org-p-oboi-nochnogo-goroda-krasivo-4.jpg

@app.post("/predict/")
def predict(item: Item):
    """Эта функция генерирует описание рисунка"""
    return classifier(item.text)[0]
