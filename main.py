# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import os
import uuid
import shutil
from typing import Optional

from c_to_excel_analyzer import analyze_c_file_to_excel

app = FastAPI(title="Requirements Generator")

# Создаем временную директорию для загрузок
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Кастомный системный промпт
CUSTOM_SYSTEM_PROMPT = (
        "Ты — инженер по требованиям и документации. Твоя задача: прочитать переданный C-код и "
        "сформулировать чёткое, тестируемое требование (или набор требований) к функциональности "
)

# Словарь для хранения соответствия file_id и имени результата
file_mapping = {}

@app.get("/")
async def home(request: Request):
    """Главная страница с формой загрузки"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_c_file(file: UploadFile = File(...)):
    """
    Загружает C-файл и запускает анализ
    """
    # Проверяем расширение файла
    if not file.filename.lower().endswith(('.c', '.cpp', '.h', '.hpp')):
        raise HTTPException(
            status_code=400, 
            detail="Поддерживаются только файлы с расширениями .c, .cpp, .h, .hpp"
        )
    
    # Генерируем уникальное имя для файла
    file_id = str(uuid.uuid4())
    upload_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    
    try:
        # Сохраняем загруженный файл
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Проверяем кодировку и пересохраняем если нужно
        try:
            with open(upload_path, 'r', encoding='utf-8') as f:
                f.read()
        except UnicodeDecodeError:
            # Если UTF-8 не работает, пробуем cp1251
            try:
                with open(upload_path, 'r', encoding='cp1251') as f:
                    content_cp1251 = f.read()
                # Пересохраняем в UTF-8 для единообразия
                with open(upload_path, 'w', encoding='utf-8') as f:
                    f.write(content_cp1251)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Ошибка чтения файла: {str(e)}")
        
        # Запускаем анализ с передачей оригинального имени файла
        excel_file_path = analyze_c_file_to_excel(
            c_file_path=upload_path,
            output_dir=RESULTS_DIR,
            original_filename=file.filename,  # Передаем оригинальное имя
            system_prompt=CUSTOM_SYSTEM_PROMPT,
            max_tokens=2024,
            delay_between_requests=1.0
        )
        
        # Сохраняем соответствие file_id и имени результата
        result_filename = os.path.basename(excel_file_path)
        file_mapping[file_id] = result_filename
        
        # Удаляем временный файл
        if os.path.exists(upload_path):
            os.remove(upload_path)
        
        return {
            "status": "success",
            "message": "Анализ завершен успешно",
            "file_id": file_id,
            "original_filename": file.filename,
            "result_filename": result_filename
        }
        
    except Exception as e:
        # Удаляем временные файлы в случае ошибки
        if os.path.exists(upload_path):
            os.remove(upload_path)
        
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе файла: {str(e)}")

@app.get("/download/{file_id}")
async def download_result(file_id: str):
    """
    Скачивание результата анализа
    """
    # Получаем имя файла из mapping
    if file_id not in file_mapping:
        raise HTTPException(status_code=404, detail="Файл не найден")
    
    result_filename = file_mapping[file_id]
    result_path = os.path.join(RESULTS_DIR, result_filename)
    
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Файл не найден")
    
    return FileResponse(
        path=result_path,
        filename=result_filename,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

@app.get("/status/{file_id}")
async def check_status(file_id: str):
    """
    Проверка статуса обработки файла
    """
    if file_id in file_mapping:
        result_filename = file_mapping[file_id]
        result_path = os.path.join(RESULTS_DIR, result_filename)
        if os.path.exists(result_path):
            return {
                "status": "completed", 
                "file_id": file_id, 
                "filename": result_filename
            }
    
    return {"status": "processing", "file_id": file_id}

@app.on_event("shutdown")
async def shutdown_event():
    """Очистка временных файлов при завершении"""
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)