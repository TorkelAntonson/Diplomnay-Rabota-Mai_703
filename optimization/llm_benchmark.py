# llm_benchmark.py
import json
import time
import psutil
import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tabulate import tabulate
import os

class LLMBenchmark:
    def __init__(self, test_data_path: str = "test_data.json"):
        """
        Инициализация бенчмарка для тестирования LLM моделей
        
        Args:
            test_data_path: Путь к JSON файлу с тестовыми данными
        """
        self.test_data_path = test_data_path
        self.results = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Используется устройство: {self.device}")
        
    def load_test_data(self) -> List[Dict[str, Any]]:
        """Загрузка тестовых данных из JSON файла"""
        try:
            with open(self.test_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Проверяем структуру данных
            if not isinstance(data, list):
                print(f"Ошибка: JSON должен содержать список тестовых случаев")
                return []
            
            print(f"Загружено {len(data)} тестовых случаев")
            return data
            
        except FileNotFoundError:
            print(f"Файл {self.test_data_path} не найден. Создаем пример данных...")
            return "Ошибка"
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON: {e}")
            return []
    
    def load_model(self, model_name: str):
        """Загрузка модели и токенизатора"""
        print(f"Загрузка модели {model_name}...")
        
        try:
            # Загружаем токенизатор
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Проверяем наличие pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Загружаем модель с оптимизациями для GPU
            if self.device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=None,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                ).to(self.device)
            
            model.eval()
            print(f"Модель {model_name} успешно загружена")
            return model, tokenizer
            
        except Exception as e:
            print(f"Ошибка загрузки модели {model_name}: {e}")
            return None, None
    
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Получение информации об использовании GPU памяти"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # в ГБ
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # в ГБ
            return {
                'allocated_gb': round(memory_allocated, 2),
                'reserved_gb': round(memory_reserved, 2)
            }
        return {'allocated_gb': 0, 'reserved_gb': 0}
    
    def prepare_evaluation_prompt(self, model_name: str, test_results: List[Dict[str, Any]]) -> str:
        """
        Подготовка одного краткого промпта для оценки всех тестовых случаев
        
        Args:
            model_name: Название модели
            test_results: Результаты по всем тестовым случаям
            
        Returns:
            Краткий промпт для оценки
        """
        prompt = f"""Оцени качество требований, сгенерированных моделью {model_name} на основе C-кода.

Критерии оценки (0-100):
- Полнота: покрывают ли все функции кода
- Точность: правильно ли отражают логику
- Ясность: понятны ли формулировки

Ответь в формате JSON:
{{
  "scores": {{
    "test_1": <оценка>,
    "test_2": <оценка>,
    ...
  }},
  "avg_score": <средняя оценка>,
  "summary": "<краткий вывод>"
}}

Тесты:
"""
        
        for i, test in enumerate(test_results, 1):
            prompt += f"\n=== Тест {i}: {test['test_name']} ===\n"
            prompt += f"Код:\n{test['code']}\n\n"
            prompt += f"Сгенерированные требования:\n{test['generated_text']}\n"
        
        return prompt
    
    def build_analysis_prompt(self, code: str) -> str:
        """Создание промпта для анализа C-кода и формулирования требований"""
        prompt = f"""
Ты — инженер по требованиям и документации. Проанализируй предоставленный C-код и сформулируй четкие, тестируемые требования.

Исходный код на C:
{code.strip()}

text

Твоя задача:
1. Проанализировать функциональность кода
2. Сформулировать требования в следующем формате:
   - Функциональное назначение: краткое описание
   - Основные требования: пронумерованный список
   - Ограничения и допущения
   - Критерии приемки (когда требование считается выполненным)

Требования должны быть:
- Конкретными и измеримыми
- Ориентированными на тестирование
- Полными (охватывать все аспекты кода)
- Недвусмысленными

Ответ предоставь на русском языке.
"""
        return prompt
    
    def test_model(self, model_name: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Тестирование одной модели на всех тестовых случаях"""
        print(f"\n{'='*60}")
        print(f"Начало тестирования модели: {model_name}")
        print(f"{'='*60}")
        
        # Загружаем модель
        model, tokenizer = self.load_model(model_name)
        if model is None or tokenizer is None:
            print(f"Не удалось загрузить модель {model_name}, пропускаем...")
            return None
        
        model_results = {
            'model_name': model_name,
            'total_time': 0,
            'total_tokens': 0,
            'test_cases': [],
            'avg_time_per_case': 0,
            'memory_usage': self.get_gpu_memory_usage(),
            'test_data_for_evaluation': []  # Данные для создания промпта оценки
        }
        
        # Тестируем на каждом кейсе
        for test_case in test_cases:
            print(f"\nТест: {test_case['name']} (ID: {test_case['id']})")
            
            # Создаем промпт для анализа
            prompt = self.build_analysis_prompt(test_case['code'])
            
            # Подготовка входных данных
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
            
            # Конфигурация генерации
            gen_config = GenerationConfig(
                max_new_tokens=2048,
                temperature=0.2,
                top_p=0.95,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            # Измеряем время и память
            start_time = time.time()
            
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=gen_config,
                )
            
            end_time = time.time()
            
            # Декодируем ответ
            generated_text = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            # Вычисляем метрики производительности
            generation_time = end_time - start_time
            total_tokens = generated.shape[1]
            tokens_per_second = round(total_tokens / generation_time, 1) if generation_time > 0 else 0
            
            # Сохраняем данные для создания промпта оценки
            model_results['test_data_for_evaluation'].append({
                'test_id': test_case['id'],
                'test_name': test_case['name'],
                'code': test_case['code'],
                'generated_text': generated_text
            })
            
            # Сохраняем результаты производительности
            case_result = {
                'test_id': test_case['id'],
                'test_name': test_case['name'],
                'generation_time': round(generation_time, 3),
                'total_tokens': total_tokens,
                'quality_score': "требует оценки",
                'tokens_per_second': tokens_per_second,
                'generated_text_preview': generated_text[:100] + "..." if len(generated_text) > 100 else generated_text
            }
            
            model_results['test_cases'].append(case_result)
            model_results['total_time'] += generation_time
            model_results['total_tokens'] += total_tokens
            
            print(f"  Время: {generation_time:.3f} сек")
            print(f"  Токенов: {total_tokens}")
            print(f"  Скорость: {tokens_per_second} токенов/сек")
        
        # Вычисляем средние значения производительности
        if model_results['test_cases']:
            model_results['avg_time_per_case'] = round(
                model_results['total_time'] / len(model_results['test_cases']), 
                3
            )
        
        # Очищаем память
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"\nЗавершено тестирование модели: {model_name}")
        print(f"Общее время: {model_results['total_time']:.2f} сек")
        avg_speed = round(model_results['total_tokens'] / model_results['total_time'], 1) if model_results['total_time'] > 0 else 0
        print(f"Средняя скорость: {avg_speed} токенов/сек")
        
        # Создаем один промпт для оценки всех тестов
        self.save_evaluation_prompt(model_results)
        
        return model_results
    
    def save_evaluation_prompt(self, model_results: Dict[str, Any]):
        """Сохранение одного промпта для оценки всех тестов"""
        model_name_safe = model_results['model_name'].replace('/', '_').replace('\\', '_')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_prompt_{model_name_safe}_{timestamp}.txt"
        
        # Создаем один краткий промпт для оценки всех тестов
        evaluation_prompt = self.prepare_evaluation_prompt(
            model_results['model_name'],
            model_results['test_data_for_evaluation']
        )
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(evaluation_prompt)
        
        print(f"Промпт для оценки сохранен в файл: {filename}")
        model_results['evaluation_prompt_file'] = filename
            
    def run_benchmark(self, models_to_test: List[str]):
        """Запуск бенчмарка для всех указанных моделей"""
        print(f"{'='*60}")
        print(f"ЗАПУСК LLM БЕНЧМАРКА")
        print(f"{'='*60}")
        print("Примечание: Качество сгенерированных требований будет оцениваться")
        print("отдельно с помощью другой LLM (например, GPT-4, Claude)")
        print(f"{'='*60}")
        
        # Загружаем тестовые данные
        test_cases = self.load_test_data()
        if not test_cases:
            print("Нет тестовых данных для выполнения бенчмарка")
            return
        
        # Тестируем каждую модель
        for model_name in models_to_test:
            result = self.test_model(model_name, test_cases)
            if result:
                self.results.append(result)
        
        # Генерируем отчет
        self.generate_report()
    
    def generate_report(self):
        """Генерация отчета о результатах тестирования"""
        if not self.results:
            print("Нет результатов для отчета")
            return
        
        print(f"\n{'='*60}")
        print(f"ОТЧЕТ О РЕЗУЛЬТАТАХ ТЕСТИРОВАНИЯ (ПРОИЗВОДИТЕЛЬНОСТЬ)")
        print(f"{'='*60}")
        print("Качество сгенерированных требований требует отдельной оценки")
        print(f"{'='*60}")
        
        # Создаем сводную таблицу производительности
        summary_data = []
        for result in self.results:
            avg_tokens_per_second = round(result['total_tokens'] / result['total_time'], 1) if result['total_time'] > 0 else 0
            summary_data.append([
                result['model_name'],
                f"{result['avg_time_per_case']} сек",
                f"{result['total_time']:.1f} сек",
                f"{result['total_tokens']}",
                f"{avg_tokens_per_second} ток/с",
                f"{result['memory_usage']['allocated_gb']} ГБ"
            ])
        
        headers = ["Модель", "Время/кейс", "Общ. время", "Всего токенов", "Сред. скорость", "GPU память"]
        
        # Выводим в консоль
        print("\nСВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (ПРОИЗВОДИТЕЛЬНОСТЬ):")
        print(tabulate(summary_data, headers=headers, tablefmt="grid"))
        
        # Детальный отчет по каждому тестовому случаю
        print(f"\n{'='*60}")
        print(f"ДЕТАЛЬНЫЙ ОТЧЕТ ПО ТЕСТОВЫМ СЛУЧАЯМ")
        print(f"{'='*60}")
        
        for result in self.results:
            print(f"\nМодель: {result['model_name']}")
            case_data = []
            for case in result['test_cases']:
                case_data.append([
                    case['test_id'],
                    case['test_name'],
                    f"{case['generation_time']} сек",
                    case['quality_score'],
                    f"{case['tokens_per_second']} ток/с",
                    case['total_tokens']
                ])
            
            case_headers = ["ID", "Тест", "Время", "Качество", "Скорость", "Токенов"]
            print(tabulate(case_data, headers=case_headers, tablefmt="simple_grid"))
        
        # Сохраняем отчет в файл
        self.save_report_to_file(summary_data, headers)
        
        # Инструкция для оценки качества
        print(f"\n{'='*60}")
        print(f"ИНСТРУКЦИЯ ДЛЯ ОЦЕНКИ КАЧЕСТВА:")
        print(f"{'='*60}")
        print("Для каждой модели создан один файл с промптом для оценки всех тестов:")
        
        for result in self.results:
            prompt_file = result.get('evaluation_prompt_file')
            if prompt_file:
                print(f"\nМодель: {result['model_name']}")
                print(f"Промпт для оценки: {prompt_file}")
        
        print("\n" + "="*60)
        print("ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ:")
        print("="*60)
        print("1. Для каждой модели создан один файл с промптом (evaluation_prompt_*.txt)")
        print("2. Отправьте этот файл в другую LLM (ChatGPT, Claude и т.д.)")
        print("3. LLM оценит все тесты сразу и вернет JSON с оценками")
        print("4. Формат ответа:")
        print("   {")
        print('     "scores": {"test_1": 85, "test_2": 90, ...},')
        print('     "avg_score": 87.5,')
        print('     "summary": "Краткий вывод"')
        print("   }")
        print("5. Обновите результаты бенчмарка полученными оценками")
    
    def save_report_to_file(self, summary_data, headers):
        """Сохранение отчета в текстовый файл"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"llm_benchmark_performance_report_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("ОТЧЕТ О ТЕСТИРОВАНИИ LLM МОДЕЛЕЙ (ПРОИЗВОДИТЕЛЬНОСТЬ)\n")
            f.write("="*60 + "\n\n")
            f.write("ПРИМЕЧАНИЕ: Качество сгенерированных требований требует отдельной оценки\n\n")
            
            f.write("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ПРОИЗВОДИТЕЛЬНОСТИ:\n")
            f.write(tabulate(summary_data, headers=headers, tablefmt="grid"))
            f.write("\n\n")
            
            f.write("="*60 + "\n")
            f.write("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО МОДЕЛЯМ\n")
            f.write("="*60 + "\n\n")
            
            for result in self.results:
                f.write(f"\nМодель: {result['model_name']}\n")
                f.write(f"Среднее время на кейс: {result['avg_time_per_case']} сек\n")
                f.write(f"Общее время: {result['total_time']:.1f} сек\n")
                f.write(f"Всего токенов: {result['total_tokens']}\n")
                avg_speed = round(result['total_tokens'] / result['total_time'], 1) if result['total_time'] > 0 else 0
                f.write(f"Средняя скорость: {avg_speed} токенов/сек\n")
                f.write(f"Использовано GPU памяти: {result['memory_usage']['allocated_gb']} ГБ\n")
                
                f.write("\nРезультаты по тестовым случаям:\n")
                for case in result['test_cases']:
                    f.write(f"  Тест {case['test_id']} ({case['test_name']}):\n")
                    f.write(f"    Время генерации: {case['generation_time']} сек\n")
                    f.write(f"    Оценка качества: {case['quality_score']}\n")
                    f.write(f"    Скорость: {case['tokens_per_second']} токенов/сек\n")
                    f.write(f"    Токенов сгенерировано: {case['total_tokens']}\n")
                
                prompt_file = result.get('evaluation_prompt_file', 'не указан')
                f.write(f"\n  Файл промпта для оценки: {prompt_file}\n")
                f.write("-"*40 + "\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("ИНСТРУКЦИЯ ДЛЯ ОЦЕНКИ КАЧЕСТВА:\n")
            f.write("="*60 + "\n")
            f.write("1. Для каждой модели создан один файл с промптом для оценки\n")
            f.write("2. Отправьте этот файл в другую LLM (ChatGPT, Claude и т.д.)\n")
            f.write("3. LLM оценит все тесты сразу в формате JSON\n")
            f.write("4. Критерии оценки: полнота, точность, ясность (0-100)\n")
        
        print(f"\nОтчет о производительности сохранен в файл: {filename}")


def main():
    """Основная функция запуска бенчмарка"""
    
    # Список моделей для тестирования
    models_to_test = [
        "models/deepseek-coder-7b-instruct-v1.5",
        "models/YandexGPT-5-Lite-8B-instruct",
        "models/T-lite-it-1.0",
        "models/Qwen2.5-Coder-7B-Instruct",
        "models/Mistral-7B-Instruct-v0.3",
    ]
    
    # Инициализация и запуск бенчмарка
    benchmark = LLMBenchmark(test_data_path="test_data.json")
    
    # Запуск тестирования
    benchmark.run_benchmark(models_to_test)


if __name__ == "__main__":
    # Проверяем наличие необходимых библиотек
    try:
        import psutil
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from tabulate import tabulate
    except ImportError as e:
        print(f"Ошибка: отсутствует необходимая библиотека: {e}")
        print("Установите недостающие библиотеки:")
        print("pip install torch transformers psutil tabulate")
        exit(1)
    
    main()