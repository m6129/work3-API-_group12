from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel #pydantic, занимается автоматической проверкой формата и типа данных

#pydantic использует модели данных для того, чтобы задать,какие именно данные требуются и какого типа.
class Item(BaseModel):# импортируем из pydantic базовый вариант класса для моделей BaseModel и создаем на его основе свою модель:
    text: str
#Класс нашей модели называется Item (элемент) и он содержит единственное поле с названием text строкового типа (str).
# Именно в параметре text мы и будем передавать текст, для которого нужно определить тональность.
app = FastAPI()
classifier = pipeline("sentiment-analysis")

@app.get("/")
def root():
    """Эта функция вызывает классификатор"""
    return {"message": "Run of road

@app.post("/predict/")
def predict(item: Item):
    """Эмоционально окрашенный анализ текста№1"""
    return classifier(item.text )[0]
