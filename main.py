# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
import os
import uuid
import shutil

from c_to_excel import analyze_c_file_to_excel
from ai_func import generate_with_rag, add_document_to_vector_db, extract_text_from_file, delete_document_from_vector_db, list_documents_in_vector_db

app = FastAPI(title="Requirements Generator")

# Создаем временные директории
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
DOCUMENTS_DIR = "documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

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
    """Главная страница с навигацией"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/analyze")
async def analyze_page(request: Request):
    """Страница анализа C/C++ кода"""
    return templates.TemplateResponse("analyze.html", {"request": request})

@app.get("/rag-chat")
async def rag_chat_page(request: Request):
    """Страница RAG чата"""
    return templates.TemplateResponse("rag_chat.html", {"request": request})

@app.get("/add-document")
async def add_document_page(request: Request):
    """Страница добавления документов"""
    documents = list_documents_in_vector_db()
    return templates.TemplateResponse("add_document.html", {"request": request, "documents": documents})

@app.post("/api/rag/chat")
async def rag_chat(query: str = Form(...)):
    """RAG чат эндпоинт"""
    try:        
        response = generate_with_rag(
            query=query,
            system_prompt="Ты — помощник по разработке требований. Отвечай на вопросы на основе предоставленного контекста.",
            temperature=0.7
        )
        
        return {"status": "success", "response": response}
        
    except Exception as e:
        return {"status": "error", "message": f"Ошибка: {str(e)}"}

@app.post("/api/rag/add-document")
async def add_document(file: UploadFile = File(...), description: str = Form("")):
    """Добавление документа в векторную БД"""
    file_path = os.path.join(DOCUMENTS_DIR, f"{uuid.uuid4()}_{file.filename}")
    
    try:
        # Сохраняем файл
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Извлекаем текст
        file_extension = os.path.splitext(file.filename)[1]
        text_content = extract_text_from_file(file_path, file_extension)
        
        # Добавляем в векторную БД
        metadata = {
            "filename": file.filename,
            "description": description,
            "type": file_extension
        }
        
        success, chunk_count = add_document_to_vector_db(text_content, metadata)
        
        # Удаляем временный файл
        os.remove(file_path)
        
        if success:
            return {
                "status": "success", 
                "message": f"Документ успешно добавлен в базу знаний ({chunk_count} чанков)"
            }
        else:
            return {"status": "error", "message": "Ошибка при добавлении документа"}
            
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return {"status": "error", "message": f"Ошибка обработки документа: {str(e)}"}

@app.delete("/api/rag/delete-document/{filename}")
async def delete_document(filename: str):
    """Удаление документа из векторной БД"""
    try:
        success = delete_document_from_vector_db(filename)
        if success:
            return {"status": "success", "message": f"Документ '{filename}' удален"}
        else:
            return {"status": "error", "message": f"Документ '{filename}' не найден"}
    except Exception as e:
        return {"status": "error", "message": f"Ошибка удаления: {str(e)}"}

@app.get("/api/rag/documents")
async def get_documents():
    """Получение списка документов в векторной БД"""
    try:
        documents = list_documents_in_vector_db()
        return {"status": "success", "documents": documents}
    except Exception as e:
        return {"status": "error", "message": f"Ошибка получения списка: {str(e)}"}

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
    """Скачивание результата анализа"""
    if file_id not in file_mapping:
        raise HTTPException(status_code=404, detail="Файл не найден")
    
    result_filename = file_mapping[file_id]
    result_path = os.path.join(RESULTS_DIR, result_filename)
    
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Результат анализа не найден")
    
    # Извлекаем оригинальное имя файла из имени результата
    # Формат: {original_filename}_requirements_{timestamp}.xlsx
    if result_filename.startswith(file_id):
        # Если имя результата начинается с file_id, используем стандартное имя
        download_filename = f"{file_id}_requirements.xlsx"
    else:
        # Иначе используем имя результата как есть (уже содержит оригинальное имя)
        download_filename = result_filename
    
    return FileResponse(
        result_path,
        filename=download_filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.get("/status/{file_id}")
async def check_status(file_id: str):
    """Проверка статуса обработки файла"""
    if file_id in file_mapping:
        return {"status": "completed", "file_id": file_id}
    else:
        return {"status": "processing", "file_id": file_id}

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    print("Requirements Generator запущен!")
    print("Временные директории созданы")

@app.on_event("shutdown")
async def shutdown_event():
    """Очистка временных файлов при завершении"""
    for directory in [UPLOAD_DIR, RESULTS_DIR, DOCUMENTS_DIR]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    print("🧹 Временные файлы очищены")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
