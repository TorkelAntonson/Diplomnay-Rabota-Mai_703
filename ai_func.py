from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

import chromadb
import hashlib
import PyPDF2
import docx


MODEL_ID = "models/YandexGPT-5-Lite-8B-instruct"

# Настройка устройства
device = "cuda" if torch.cuda.is_available() else "cpu"

# Инициализация токенайзера и модели
print(f"Loading tokenizer and model '{MODEL_ID}' on device {device} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Оптимизированная инициализация модели
if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=None,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to(device)

model.eval()

def build_prompt(code: str, system_prompt: Optional[str] = None) -> str:
    """
    Создаёт промпт для данного куска C-кода.
    Вы можете менять шаблон под свои стандарты.
    """
    base_system = (
        "Ты — инженер по требованиям и документации. Твоя задача: прочитать переданный C-код и "
        "сформулировать чёткое, тестируемое требование (или набор требований) к функциональности "
    )
    if system_prompt:
        base_system = system_prompt

    prompt = (
        f"{base_system}\n\n"
        "Инструкция: на основе следующего кода на языке C сформулируй:\n"
        "Точные требования (пункты), подробно описывающие действия функии.\n"
        "Код:\n```\n" + code.strip() + "\n```\n\n"
        "Ответь развернуто, иногда можешь использовать псевдокод."
    )
    return prompt

# Функция генерации требования к коду на С
def generate_requirement(code: str, system_prompt: Optional[str]=None, max_new_tokens: int = 2024, temperature: float = 0.0) -> str:
    """
    Сгенерировать требование/спецификацию для переданного C-кода.
    """
    prompt = build_prompt(code, system_prompt=system_prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        do_sample=False if temperature == 0.0 else True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
        )

    out = tokenizer.decode(generated[0], skip_special_tokens=True)
    # Вычистим: оставить только текст после исходного промпта, если модель вернула и вход.
    if out.startswith(prompt):
        result = out[len(prompt):].strip()
    else:
        # иногда модель возвращает полный диалог иначе — просто возвращаем всё, но аккуратно
        result = out.strip()
    return result

# Пример использования
if __name__ == "__main__":

    sample_c_code = """..."""
    spec = generate_requirement(sample_c_code, max_new_tokens=2024, temperature=0.0)
    print("=== GENERATED SPEC ===\n")
    print(spec)

# Инициализация ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

def split_text_into_chunks(text: str, max_lines: int = 30) -> List[str]:
    """Разбивает текст на чанки по максимальному количеству строк"""
    lines = text.split('\n')
    chunks = []
    
    for i in range(0, len(lines), max_lines):
        chunk = '\n'.join(lines[i:i + max_lines])
        chunks.append(chunk)
    
    return chunks

def build_prompt_rag(query: str, context_docs: List[str], system_prompt: str = "") -> str:
    """Строит промпт с RAG контекстом"""
    context = "\n\n".join(context_docs)
    
    prompt = f"""{system_prompt}

Контекст для ответа:
{context}

Вопрос: {query}

Ответ:"""
    return prompt

def search_similar_documents(query: str, n_results: int = 3) -> List[str]:
    """Поиск похожих отрезков из всех документов в векторной БД"""
    try:
        # Получаем больше результатов для лучшего выбора
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results * 5, 20)  # Получаем больше результатов для фильтрации
        )
        
        if not results['documents']:
            return []
        
        # Фильтруем и ранжируем чанки по релевантности и размеру
        filtered_docs = []
        
        for i, doc in enumerate(results['documents'][0]):
            lines = doc.split('\n')
            line_count = len(lines)
            
            # Оцениваем качество чанка по нескольким критериям
            score = 0
            
            # 1. Предпочтение чанкам оптимального размера (20-30 строк)
            if 20 <= line_count <= 30:
                score += 3  # Максимальный балл за идеальный размер
            elif 15 <= line_count <= 40:
                score += 2
            elif line_count <= 50:
                score += 1
            
            # 2. Учитываем релевантность (расстояние/схожесть)
            if results['distances'] and i < len(results['distances'][0]):
                similarity = 1 - results['distances'][0][i]  # Конвертируем расстояние в схожесть
                score += similarity * 3  # Больший вес релевантности
            
            # 3. Предпочтение чанкам с законченными предложениями/блоками
            if has_complete_sentences(doc):
                score += 1
            
            # 4. Бонус за чанки, которые содержат ключевые термины запроса
            query_terms = set(query.lower().split())
            doc_terms = set(doc.lower().split())
            common_terms = query_terms.intersection(doc_terms)
            if common_terms:
                score += len(common_terms) * 0.5
            
            filtered_docs.append({
                'text': doc,
                'score': score,
                'line_count': line_count
            })
        
        # Сортируем по убыванию score и берем лучшие
        filtered_docs.sort(key=lambda x: x['score'], reverse=True)
        
        # Возвращаем только тексты лучших чанков
        best_chunks = [doc['text'] for doc in filtered_docs[:n_results]]
        
        # Логирование для отладки
        if filtered_docs:
            print(f"Найдено {len(filtered_docs)} потенциальных чанков, выбрано {len(best_chunks)} лучших")
            print(f"Лучший score: {filtered_docs[0]['score']:.2f}, размер: {filtered_docs[0]['line_count']} строк")
        
        return best_chunks
        
    except Exception as e:
        print(f"Ошибка поиска в ChromaDB: {e}")
        return []

def extract_most_relevant_part(text: str, query: str, max_lines: int = 30) -> Optional[str]:
    """Извлекает наиболее релевантную часть из большого текста"""
    try:
        lines = text.split('\n')
        if len(lines) <= max_lines:
            return text
        
        # Разбиваем на перекрывающиеся окна
        window_size = max_lines
        step_size = max(1, max_lines // 3)  # 33% перекрытие для меньших чанков
        
        best_window = ""
        best_score = 0
        
        query_terms = set(query.lower().split())
        
        for start in range(0, len(lines) - window_size + 1, step_size):
            end = start + window_size
            window = '\n'.join(lines[start:end])
            
            # Простая оценка релевантности по совпадению терминов
            window_terms = set(window.lower().split())
            common_terms = query_terms.intersection(window_terms)
            score = len(common_terms)
            
            # Бонус за окна в начале документа (часто содержат введение/основную мысль)
            if start == 0:
                score += 1
            
            if score > best_score:
                best_score = score
                best_window = window
        
        return best_window if best_score > 0 else '\n'.join(lines[:max_lines])
    
    except Exception as e:
        print(f"Ошибка извлечения релевантной части: {e}")
        return '\n'.join(lines[:max_lines])

def has_complete_sentences(text: str) -> bool:
    """Проверяет, содержит ли текст законченные предложения"""
    # Простая эвристика: проверяем наличие пунктуации в конце
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) >= 2:  # Хотя бы 2 предложения
        return True
    
    # Проверяем наличие завершающих символов
    endings = ['.', ';', '!', '?']
    for ending in endings:
        if text.rstrip().endswith(ending):
            return True
    
    return False


def add_document_to_vector_db(text: str, metadata: Dict = None):
    """Добавление документа в векторную БД с разбивкой на чанки по 30 строк"""
    try:
        # Разбиваем текст на чанки по 30 строк
        chunks = split_text_into_chunks(text, max_lines=30)
        
        # Создаем ID для каждого чанка
        doc_hashes = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_hash = hashlib.md5(f"{text}_{i}".encode()).hexdigest()
            doc_hashes.append(chunk_hash)
            documents.append(chunk)
            
            # Добавляем информацию о чанке в метаданные
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            chunk_metadata["line_count"] = len(chunk.split('\n'))  # Добавляем количество строк
            metadatas.append(chunk_metadata)
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=doc_hashes
        )
        print(f"Добавлено {len(chunks)} чанков по ~30 строк каждый")
        return True, len(chunks)
    except Exception as e:
        print(f"Ошибка добавления в ChromaDB: {e}")
        return False, 0
    
def delete_document_from_vector_db(filename: str) -> bool:
    """Удаление всех чанков документа по имени файла"""
    try:
        # Находим все чанки с указанным именем файла
        results = collection.get(
            where={"filename": filename}
        )
        
        if results and results['ids']:
            # Удаляем все найденные чанки
            collection.delete(ids=results['ids'])
            print(f"Удалено {len(results['ids'])} чанков документа '{filename}'")
            return True
        else:
            print(f"Документ '{filename}' не найден")
            return False
            
    except Exception as e:
        print(f"Ошибка удаления из ChromaDB: {e}")
        return False

def list_documents_in_vector_db() -> List[Dict]:
    """Получение списка всех уникальных документов в векторной БД"""
    try:
        # Получаем все данные из коллекции
        all_data = collection.get()
        
        if not all_data['metadatas']:
            return []
        
        # Группируем по имени файла
        documents_map = {}
        
        for metadata in all_data['metadatas']:
            filename = metadata.get('filename', 'Unknown')
            if filename not in documents_map:
                documents_map[filename] = {
                    'filename': filename,
                    'description': metadata.get('description', ''),
                    'type': metadata.get('type', ''),
                    'chunk_count': 0,
                    'total_chunks': metadata.get('total_chunks', 1)
                }
            documents_map[filename]['chunk_count'] += 1
        
        return list(documents_map.values())
        
    except Exception as e:
        print(f"Ошибка получения списка документов: {e}")
        return []

def extract_text_from_file(file_path: str, file_extension: str) -> str:
    """Извлечение текста из файлов разных форматов"""
    try:
        if file_extension.lower() in ['.txt']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        elif file_extension.lower() in ['.pdf']:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
                
        elif file_extension.lower() in ['.docx']:
            doc = docx.Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
        else:
            # Для других форматов пытаемся читать как текст
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
    except Exception as e:
        raise Exception(f"Ошибка извлечения текста: {str(e)}")

def generate_with_rag(query: str, system_prompt: str = "", 
                     max_new_tokens: int = 4096, temperature: float = 0.7,
                     n_context_docs: int = 3) -> str:
    """Генерация ответа с использованием RAG"""
    
    print(f"Поиск релевантной информации для запроса: '{query}'")
    
    # Поиск похожих документов
    context_docs = search_similar_documents(query, n_context_docs)
    
    if not context_docs:
        print("Релевантные данные не найдены, используется только системный промпт")
    
    # Построение промпта с RAG
    prompt = build_prompt_rag(query, context_docs, system_prompt)
    
    # Генерация ответа
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        do_sample=False if temperature == 0.0 else True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
        )
    
    # Декодируем только сгенерированную часть
    generated_text = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    # Вычистим: оставить только текст после исходного промпта, если модель вернула и вход.
    if generated_text.startswith(prompt):
        result = generated_text[len(prompt):].strip()
    else:
        result = generated_text.strip()
        
    return result
