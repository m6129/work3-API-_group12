# work3-API-_group12 + Work5
Группа 12:
1) Зайцев Антон
2) Зайцев Александр
3) Чурилов Алексей  

Work№3 + Work5
Данную программу можно запустить через uvicorn.
Программа реализована на FastAPI.
Модель машинного обучения генерирует текст на английском языке по изображению, т.е решает задачу формата Image-to-Text 

Модель принимает на вход изображения типа: https://.....jpg
Модель подлючается через библиотеку transformes, вес модели 900+ mb. 

Короткая памятка по применению:  

1.запустить приложение в терминале питона:  
	uvicorn main:app  
	
2.открыть в браузере:  
	http://127.0.0.1:8000/docs  
3. жмем post -> try it out -> вместо string вводим https:....jpg -> execute

4. запустить тест приложения в терминале питона:
  pytest	


