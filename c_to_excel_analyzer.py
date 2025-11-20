# c_to_excel_analyzer.py
"""
c_to_excel_analyzer.py

Модуль для анализа C-файлов и создания Excel-таблицы с требованиями к функциям.
Использует c_parser.py для извлечения функций и ai_analysis.py для генерации требований.
"""

import os
import pandas as pd
from typing import List
import time
from c_parser import extract_function_data
from ai_analysis import generate_requirement


def analyze_c_file_to_excel(c_file_path: str, output_dir: str = None, 
                          original_filename: str = None,
                          system_prompt: str = None, max_tokens: int = 2024,
                          delay_between_requests: float = 1.0) -> str:
    """
    Анализирует C-файл и создает Excel-таблицу с требованиями к функциям.
    
    Args:
        c_file_path: Путь к C-файлу для анализа
        output_dir: Директория для сохранения Excel-файла (по умолчанию - директория C-файла)
        original_filename: Оригинальное имя файла (без UUID)
        system_prompt: Кастомный системный промпт для AI
        max_tokens: Максимальное количество токенов для генерации
        delay_between_requests: Задержка между запросами к AI (в секундах)
    
    Returns:
        str: Путь к созданному Excel-файлу
    """
    
    # Проверяем существование файла
    if not os.path.exists(c_file_path):
        raise FileNotFoundError(f"Файл {c_file_path} не найден")
    
    # Определяем имя для выходного файла
    if original_filename:
        # Используем переданное оригинальное имя
        c_file_name = os.path.splitext(original_filename)[0]
    else:
        # Используем имя из пути, но убираем UUID если есть
        temp_name = os.path.basename(c_file_path)
        if '_' in temp_name and len(temp_name.split('_')[0]) == 36:  # UUID имеет длину 36 символов
            # Убираем UUID часть
            c_file_name = '_'.join(temp_name.split('_')[1:])
        else:
            c_file_name = temp_name
        c_file_name = os.path.splitext(c_file_name)[0]
    
    if output_dir is None:
        output_dir = os.path.dirname(c_file_path)
    
    # Используем имя входного файла + "_requirements" для выходного файла
    excel_file_path = os.path.join(output_dir, f"{c_file_name}_requirements.xlsx")
    
    # Читаем и анализируем C-файл
    print(f"Чтение файла: {c_file_path}")
    with open(c_file_path, 'r', encoding='cp1251', errors='ignore') as f:
        c_code = f.read()
    
    # Извлекаем данные о функциям
    print("Извлечение данных о функциям...")
    functions_data = extract_function_data(c_code)
    
    # Разделяем основные функции и сегменты
    main_functions = [f for f in functions_data if "name" in f]
    segments = [f for f in functions_data if "segment_index" in f]
    
    print(f"Найдено функций: {len(main_functions)}")
    print(f"Найдено сегментов: {len(segments)}")
    
    # Подготавливаем данные для таблицы
    excel_data = []
    requirement_counter = 1
    
    # Обрабатываем основные функции
    for i, func in enumerate(main_functions, 1):
        func_name = func["name"]
        print(f"Обработка функции {i}/{len(main_functions)}: {func_name}")
        
        # Проверяем, есть ли сегменты для этой функции
        func_segments = [s for s in segments if s['signature'] == func['signature']]
        has_segments = len(func_segments) > 0
        
        # Генерируем общее требование для всей функции ТОЛЬКО если нет сегментов
        function_requirement = ""
        if not has_segments:
            # Формируем полный код функции для анализа
            function_code = f"{func['signature']} {{\n{func['full_body']}\n}}"
            
            try:
                print(f"  Генерация общего требования для функции {func_name}...")
                function_requirement = generate_requirement(
                    function_code, 
                    system_prompt=system_prompt,
                    max_new_tokens=max_tokens
                )
                print(f"  ✓ Общее требование сгенерировано")
            except Exception as e:
                print(f"  ✗ Ошибка генерации общего требования: {e}")
                function_requirement = f"Ошибка генерации требования: {str(e)}"
        else:
            print(f"  Функция разбита на {len(func_segments)} сегментов - пропускаем генерацию общего требования")
        
        # Добавляем информацию о функции
        function_elements = [
            {
                'element': 'Прототип',
                'value': func['signature'],
                'is_requirement': 'нет'
            },
            {
                'element': 'Входной поток',
                'value': ', '.join(func['globals_used']) if func['globals_used'] else 'нет',
                'is_requirement': 'нет'
            },
            {
                'element': 'Выходной поток', 
                'value': ', '.join(func['globals_written']) if func['globals_written'] else 'нет',
                'is_requirement': 'нет'
            },
            {
                'element': 'Используемые константы',
                'value': ', '.join(func['constants']) if func['constants'] else 'нет',
                'is_requirement': 'нет'
            },
            {
                'element': 'Вспомогательные функции',
                'value': ', '.join(func['helpers']) if func['helpers'] else 'нет',
                'is_requirement': 'нет'
            },
            {
                'element': 'Внутренние переменные',
                'value': ', '.join(func['internal_variables']) if func['internal_variables'] else 'нет',
                'is_requirement': 'нет'
            }
        ]
        
        # Добавляем общее требование только если оно было сгенерировано
        if function_requirement:
            function_elements.append({
                'element': f"Требование LLR_{requirement_counter:02d} (Общее для функции)",
                'value': function_requirement,
                'is_requirement': 'да'
            })
            requirement_counter += 1
        
        # Создаем строки таблицы для элементов функции
        for elem in function_elements:
            if elem['is_requirement'] == 'да':
                header_content = elem['value']  # Только текст требования
                requirement_id = f"LLR_{requirement_counter-1:02d}"
            else:
                header_content = f"{elem['element']}: {elem['value']}"
                requirement_id = ""
            
            row_data = {
                'ИД': requirement_id,
                'Заголовок/Текст': header_content,
                'Требование': elem['is_requirement'],
                'Функция': func_name,
                'Тип элемента': 'Функция',
                'Подэлемент': 'Основная информация',
                'Имя файла': original_filename if original_filename else os.path.basename(c_file_path)
            }
            
            excel_data.append(row_data)
        
        # Обрабатываем сегменты этой функции, если они есть
        if func_segments:
            print(f"  Обработка {len(func_segments)} сегментов функции {func_name}...")
            
            for seg_idx, segment in enumerate(func_segments):
                segment_index = segment['segment_index']
                segment_body = segment['segment_body']
                
                # Формируем код сегмента для анализа (только сегмент)
                segment_code = f"// Сегмент {segment_index + 1} из {segment['segment_total']} функции {func_name}\n{segment_body}"
                
                # Генерируем требование для сегмента
                try:
                    print(f"    Генерация требования для сегмента {segment_index + 1}...")
                    segment_requirement = generate_requirement(
                        segment_code,
                        system_prompt=system_prompt,
                        max_new_tokens=max_tokens
                    )
                    print(f"    ✓ Требование для сегмента {segment_index + 1} сгенерировано")
                except Exception as e:
                    print(f"    ✗ Ошибка генерации требования для сегмента: {e}")
                    segment_requirement = f"Ошибка генерации требования: {str(e)}"
                
                # Добавляем только требование для сегмента
                segment_elements = [
                    {
                        'element': f"Требование LLR_{requirement_counter:02d} (Сегмент {segment_index + 1})",
                        'value': segment_requirement,
                        'is_requirement': 'да'
                    }
                ]
                
                requirement_counter += 1
                
                # Создаем строки таблицы для элементов сегмента
                for elem in segment_elements:
                    if elem['is_requirement'] == 'да':
                        header_content = elem['value']  # Только текст требования
                        requirement_id = f"LLR_{requirement_counter-1:02d}"
                    else:
                        header_content = f"{elem['element']}: {elem['value']}"
                        requirement_id = ""
                    
                    row_data = {
                        'ИД': requirement_id,
                        'Заголовок/Текст': header_content,
                        'Требование': elem['is_requirement'],
                        'Функция': func_name,
                        'Имя файла': original_filename if original_filename else os.path.basename(c_file_path)
                    }
                    
                    excel_data.append(row_data)
                
                # Задержка между запросами к AI для сегментов
                if delay_between_requests > 0:
                    time.sleep(delay_between_requests)
        
        # Задержка между запросами к AI для функций
        if i < len(main_functions) and delay_between_requests > 0:
            time.sleep(delay_between_requests)
    
    # Создаем DataFrame
    df = pd.DataFrame(excel_data)
    
    # Настраиваем порядок колонок
    column_order = [
        'ИД', 'Заголовок/Текст', 'Требование', 'Функция', 'Имя файла'
    ]
    df = df[column_order]
    
    # Сохраняем в Excel
    print(f"Сохранение в Excel: {excel_file_path}")
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Требования', index=False)
        
        # Настраиваем ширину колонок для лучшего отображения
        worksheet = writer.sheets['Требования']
        column_widths = {
            'A': 10,   # ИД
            'B': 60,   # Заголовок/Текст
            'C': 12,   # Требование
            'D': 20,   # Функция
            'E': 15,   # Тип элемента
            'F': 15,   # Подэлемент
            'G': 15    # Имя файла
        }
        
        for col, width in column_widths.items():
            worksheet.column_dimensions[col].width = width
        
        # Включаем перенос текста для колонки "Заголовок/Текст"
        for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row, min_col=2, max_col=2):
            for cell in row:
                cell.alignment = cell.alignment.copy(wrap_text=True)
    
    print(f"✓ Анализ завершен. Файл сохранен: {excel_file_path}")
    print(f"✓ Всего сгенерировано требований: {requirement_counter - 1}")
    return excel_file_path


def batch_analyze_c_files(c_files_dir: str, output_dir: str = None, 
                         system_prompt: str = None, max_tokens: int = 2024,
                         delay_between_requests: float = 1.0) -> List[str]:
    """
    Пакетный анализ всех C-файлов в директории.
    
    Args:
        c_files_dir: Директория с C-файлами
        output_dir: Директория для сохранения Excel-файлов
        system_prompt: Кастомный системный промпт
        max_tokens: Максимальное количество токенов
        delay_between_requests: Задержка между запросами к AI
    
    Returns:
        List[str]: Список путей к созданным Excel-файлов
    """
    
    if output_dir is None:
        output_dir = c_files_dir
    
    # Создаем выходную директорию, если не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Ищем все C-файлы
    c_files = [f for f in os.listdir(c_files_dir) 
              if f.lower().endswith(('.c', '.cpp', '.h', '.hpp'))]
    
    results = []
    
    for c_file in c_files:
        c_file_path = os.path.join(c_files_dir, c_file)
        try:
            print(f"\n{'='*60}")
            print(f"Обработка файла: {c_file}")
            print(f"{'='*60}")
            
            excel_path = analyze_c_file_to_excel(
                c_file_path, 
                output_dir, 
                original_filename=c_file,
                system_prompt=system_prompt, 
                max_tokens=max_tokens,
                delay_between_requests=delay_between_requests
            )
            results.append(excel_path)
            print(f"✓ Файл обработан: {c_file}")
        except Exception as e:
            print(f"✗ Ошибка при обработке файла {c_file}: {e}")
    
    return results