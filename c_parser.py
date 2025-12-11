"""
c_function_extractor.py

Скрипт на Python (только стандартная библиотека), который извлекает функции из файла .c (кодировка CP1251)
и возвращает словарь кортежей словарей с такими данными для каждой функции:
 - имя функции
 - найденные константы
 - вспомогательные функции (вызовы других функций, найденные в теле)
 - внутренние переменные
 - тело функции
"""

import re
from collections import OrderedDict
from typing import Dict, Tuple, List

# --- helpers -----------------------------------------------------------------

def read_c_file(path: str, encoding: str = 'cp1251') -> str:
    with open(path, 'r', encoding=encoding, errors='ignore') as f:
        return f.read()


def remove_comments(code: str) -> str:
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//.*?\n', '\n', code)
    return code


def find_global_declarations(code: str) -> Dict[str, str]:
    m = re.search(r'^[\s\S]*?^[a-zA-Z_][\w\s\*]*\b([a-zA-Z_]\w*)\s*\([^)]*\)\s*\{', code, flags=re.M)
    prefix = code
    if m:
        prefix = code[:m.start()]

    lines = prefix.splitlines()
    globals_ = {}
    decl_re = re.compile(r'''^\s*(?:extern\s+|static\s+)?([a-zA-Z_][\w\s\*<>]*)\s+([a-zA-Z_]\w*)(\s*=\s*[^;]+)?\s*;\s*$''')
    typedef_re = re.compile(r'^\s*typedef\b')
    for ln in lines:
        if typedef_re.search(ln):
            continue
        m = decl_re.match(ln)
        if m:
            varname = m.group(2)
            globals_[varname] = ln.strip()
    return globals_


def find_functions(code: str) -> List[Tuple[str, str, str]]:
    funcs = []
    sig_re = re.compile(
        r'(^|\n)\s*([a-zA-Z_][\w\s\*\(\)\[\]]*?)\s+([a-zA-Z_]\w*)\s*\(([^;\)]*)\)\s*\{',
        flags=re.M
    )
    for m in sig_re.finditer(code):
        func_name = m.group(3)
        if func_name in C_KEYWORDS or func_name in {'if', 'for', 'while', 'switch', 'do', 'else'}:
            continue

        start = m.end() - 1
        params = m.group(4).strip()
        body, endpos = extract_brace_block(code, start)
        if body is None:
            continue
        prefix = m.group(2)
        funcs.append((prefix.strip(), func_name, params, body))
    return funcs

def extract_brace_block(code: str, open_brace_pos: int):
    stack = []
    i = open_brace_pos
    n = len(code)
    while i < n:
        ch = code[i]
        if ch == '{':
            stack.append(i)
        elif ch == '}':
            stack.pop()
            if not stack:
                return code[open_brace_pos:i+1], i+1
        elif ch == '"':
            i = skip_string(code, i)
        elif ch == '\\' and i+1 < n:
            i += 1
        i += 1
    return None, None


def skip_string(code: str, i: int) -> int:
    i += 1
    n = len(code)
    while i < n:
        if code[i] == '"':
            return i
        if code[i] == '\\' and i+1 < n:
            i += 2
            continue
        i += 1
    return i


def split_large_function_body(body: str, target_lines: int = 200) -> List[str]:
    """
    Разбивает большое тело функции на подблоки примерно по target_lines строк,
    разбивая сразу после операторов или по достижении лимита строк.
    """
    lines = body.split('\n')
    if len(lines) <= target_lines:
        return [body]
    
    segments = []
    current_segment = []
    current_line_count = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        line_stripped = line.strip()
        
        current_segment.append(line)
        current_line_count += 1
        
        # Проверяем, является ли текущая строка оператором
        is_operator = any(line_stripped.startswith(kw) for kw in ['if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default'])
        
        # Проверяем, заканчивается ли строка точкой с запятой (завершение оператора)
        ends_with_semicolon = line_stripped.endswith(';')
        
        # Проверяем, заканчивается ли строка открывающей скобкой (начало блока)
        ends_with_open_brace = line_stripped.endswith('{')
        
        # Проверяем, заканчивается ли строка закрывающей скобкой (конец блока)
        ends_with_close_brace = line_stripped.endswith('}')
        
        # Условия для разбиения:
        # 1. Достигли целевого количества строк И (строка завершает оператор ИЛИ начинается новый оператор)
        # 2. ИЛИ начинается новый крупный оператор (if, for, while и т.д.)
        should_split = False
        
        if current_line_count >= target_lines:
            # Разбиваем если: завершился оператор или начинается новый оператор
            if ends_with_semicolon or ends_with_close_brace or is_operator:
                should_split = True
        
        # Также разбиваем перед крупными операторами, даже если не достигли лимита
        # (но только если уже накопили хотя бы половину целевого размера)
        elif (current_line_count >= target_lines // 2 and 
              is_operator and 
              len(current_segment) > 1):  # Не разбиваем если оператор первая строка
            should_split = True
        
        if should_split and current_segment:
            segments.append('\n'.join(current_segment))
            current_segment = []
            current_line_count = 0
        
        i += 1
    
    # Добавляем оставшиеся строки
    if current_segment:
        segments.append('\n'.join(current_segment))
    
    # Если разбиение не удалось или получился только один сегмент, возвращаем исходное тело
    if len(segments) <= 1:
        return [body]
    
    return segments


# полный список ключевых слов C и C11/C17
C_KEYWORDS = set('''
auto break case char const continue default do double else enum extern float 
for goto if inline int long register restrict return short signed sizeof static 
struct switch typedef union unsigned void volatile while _Bool _Complex _Imaginary
typeof alignas alignof _Atomic _Generic _Noreturn _Static_assert _Thread_local
'''.split())

TYPE_WORDS = r'(?:unsigned|signed|short|long|int|char|float|double|void|struct|enum|const|volatile|bool|size_t|ssize_t|uint32_t|uint16_t|uint8_t|uint_t|int32_t|int16_t|int8_t|int_t|char_t|float32_t|f32_t|float64_t|f64_t)'

VARIABLE_DECL_RE = re.compile(r'\b(' + TYPE_WORDS + r'(?:\s+' + TYPE_WORDS + r')?|[A-Za-z_][\w]*)\s+([A-Za-z_]\w*)(?:\s*\[\s*\d*\s*\])?(?:\s*=\s*[^;,]+)?(?:\s*,\s*[A-Za-z_]\w*(?:\s*\[\s*\d*\s*\])?)*\s*;')


def analyze_global_usages(body: str, globals_: Dict[str, str]):
    """
    Анализирует использование глобальных переменных:
    - входные потоки (чтение)
    - выходные потоки (запись)
    """
    reads = set()
    writes = set()

    body = remove_comments(body)
    body = re.sub(r'"(?:\\.|[^"\\])*"', '""', body)

    for g in globals_.keys():
        g_esc = re.escape(g)

        # --- запись ---
        # Прямое присваивание: x = ...
        if re.search(r'\b' + g_esc + r'\b\s*=', body):
            writes.add(g)
        # Присваивание в поле структуры: x.y = ...
        if re.search(r'\b' + g_esc + r'\s*\.\s*[A-Za-z_]\w*\s*=', body):
            writes.add(g)
        # Операции инкремента/декремента
        if re.search(r'(\+\+|--)\s*' + g_esc + r'\b|\b' + g_esc + r'\b\s*(\+\+|--)', body):
            writes.add(g)

        # --- чтение ---
        # Используется справа от = (x = ...)
        if re.search(r'=\s*[^;]*\b' + g_esc + r'\b', body):
            reads.add(g)
        # Используется в условиях (if, while, for)
        if re.search(r'\b(if|while|for)\b[^{;]*\b' + g_esc + r'\b', body):
            reads.add(g)
        # Передача в функцию как аргумент
        if re.search(r'\b[A-Za-z_]\w*\s*\([^)]*\b' + g_esc + r'\b', body):
            reads.add(g)

    return sorted(reads), sorted(writes)


def analyze_function(prefix: str, func_name: str, params: str, body: str, globals_: Dict[str, str]):
    body_nocom = remove_comments(body)

    # --- параметры ---
    params_list = []
    if params.strip():
        parts = split_params(params)
        for p in parts:
            p = p.strip()
            if not p or p == 'void':
                continue
            tokens = p.split()
            name = tokens[-1]
            name = re.sub(r'\[.*\]$', '', name)
            params_list.append(p)

    pointer_params = [p for p in params_list if '*' in p]

    # --- анализ глобальных переменных ---
    reads, writes = analyze_global_usages(body_nocom, globals_)

    # --- возвращаемые значения ---
    returns = re.findall(r'\breturn\b\s*([^;\n]+)', body_nocom)
    returns = [r.strip() for r in returns]

    # --- константы ---
    strings = re.findall(r'"(?:\\.|[^"\\])*"', body_nocom)
    numbers = re.findall(r'\b0x[0-9A-Fa-f]+\b|\b\d+\.\d+\b|\b\d+\b', body_nocom)
    constants = sorted(set(strings + numbers), key=lambda x: (x not in strings, x))

    # --- вспомогательные функции ---
    calls = re.findall(r'([A-Za-z_]\w*)\s*\(', body_nocom)
    helpers = []
    for c in calls:
        if c == func_name or c in C_KEYWORDS:
            continue
        helpers.append(c)
    helpers = sorted(set(helpers))

    # --- внутренние переменные ---
    internal_vars = []
    for m in VARIABLE_DECL_RE.finditer(body_nocom):
        varname = m.group(2)
        internal_vars.append(varname)
    internal_vars = sorted(set(internal_vars))

    # --- разбиение тела ---
    body_segments = split_large_function_body(body)

    # --- сбор результатов ---
    # prefix содержит "тип" функции и возможные модификаторы (например "static inline int")
    dict_name = {'name': func_name, 'return_type': prefix.strip()}
    dict_input = {'parameters': params_list, 'globals_used': reads}
    dict_output = {
        'pointer_parameters': pointer_params,
        'globals_written': writes,
        'returns': returns
    }
    dict_constants = {'constants': constants}
    dict_helpers = {'helpers': helpers}
    dict_internal = {'internal_variables': internal_vars}
    dict_body = {'body': body.strip(), 'body_segments': body_segments}

    return (dict_name, dict_input, dict_output, dict_constants, dict_helpers, dict_internal, dict_body)


def split_params(params: str) -> List[str]:
    parts = []
    cur = []
    depth = 0
    i = 0
    while i < len(params):
        ch = params[i]
        if ch == '(':
            depth += 1
            cur.append(ch)
        elif ch == ')':
            depth = max(0, depth-1)
            cur.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    if cur:
        parts.append(''.join(cur))
    return parts


def extract_from_string(code: str) -> Dict[str, Tuple[dict, dict, dict, dict, dict, dict, dict]]:
    code_raw = code
    code = remove_comments(code_raw)
    globals_ = find_global_declarations(code)
    functions = find_functions(code)

    result = OrderedDict()
    for prefix, fname, params, body in functions:
        analyzed = analyze_function(prefix, fname, params, body, globals_)
        result[fname] = analyzed
    return result


def extract_from_file(path: str, encoding: str = 'cp1251'):
    txt = read_c_file(path, encoding=encoding)
    return extract_from_string(txt)


def extract_function_data(code: str, encoding: str = 'cp1251'):
    """
    Возвращает список словарей с данными о функциях и их сегментах.

    Каждый элемент списка — это словарь одного из двух типов:

    1 Основная информация о функции (всегда присутствует):
    ----------------------------------------------------------------
    "name"              — имя функции
    "return_type"       — тип возвращаемого значения (включая модификаторы)
    "parameters"        — список параметров функции
    "globals_used"      — глобальные переменные, использованные как вход
    "globals_written"   — глобальные переменные, изменённые в функции (выход)
    "returns"           — возвращаемые выражения (уникальные)
    "pointer_parameters"— параметры-указатели
    "constants"         — найденные строковые и числовые константы
    "helpers"           — имена вспомогательных вызываемых функций
    "internal_variables"— локальные переменные функции
    "signature"         — строка с типом, именем и параметрами
    "full_body"         — полное тело функции (включая все сегменты)
    "body_lines"        — количество строк в теле
    "segment_total"     — количество сегментов (1 или более)
    ----------------------------------------------------------------

    2 Сегменты функции (если функция разбита на части):
    ----------------------------------------------------------------
    "signature"         — строка сигнатуры (тип + имя + параметры)
    "segment_index"     — индекс сегмента (0, 1, 2, ...)
    "segment_total"     — общее число сегментов
    "segment_body"      — текст конкретного сегмента тела
    "body_lines"        — количество строк в сегменте
    ----------------------------------------------------------------
    """

    result = extract_from_string(code)
    output = []

    for fname, data in result.items():
        dict_name, dict_input, dict_output, dict_constants, dict_helpers, dict_internal, dict_body = data

        return_type = dict_name.get('return_type', '').strip()
        params = dict_input.get('parameters', [])
        globals_used = sorted(set(dict_input.get('globals_used', [])))
        globals_written = sorted(set(dict_output.get('globals_written', [])))
        returns = sorted(set(dict_output.get('returns', [])))
        pointer_params = sorted(set(dict_output.get('pointer_parameters', [])))
        constants = sorted(set(dict_constants.get('constants', [])))
        helpers = sorted(set(dict_helpers.get('helpers', [])))
        internal_vars = sorted(set(dict_internal.get('internal_variables', [])))

        signature = f"{return_type} {fname}({', '.join(params)})".strip()
        full_body = dict_body.get('body', '').strip()
        body_segments = dict_body.get('body_segments', [])
        segment_total = len(body_segments)

        # --- 1. добавляем основной блок с полной информацией о функции ---
        output.append({
            "name": fname,
            "return_type": return_type,
            "parameters": params,
            "globals_used": globals_used,
            "globals_written": globals_written,
            "returns": returns,
            "pointer_parameters": pointer_params,
            "constants": constants,
            "helpers": helpers,
            "internal_variables": internal_vars,
            "signature": signature,
            "full_body": full_body,
            "body_lines": len(full_body.splitlines()),
            "segment_total": segment_total if segment_total > 0 else 1
        })

        # --- 2. если есть сегменты, добавляем отдельные записи для каждого ---
        if segment_total > 1:
            for idx, seg in enumerate(body_segments):
                seg_body = seg.strip()
                seg_lines = len(seg_body.splitlines())
                output.append({
                    "signature": signature,
                    "segment_index": idx,
                    "segment_total": segment_total,
                    "segment_body": seg_body,
                    "body_lines": seg_lines
                })

    return output



if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Извлекает информацию о функциях C из файла .c (CP1251).')
    p.add_argument('file', help='путь к .c файлу')
    p.add_argument('--encoding', default='cp1251', help='кодировка файла (по умолчанию cp1251)')
    args = p.parse_args()

    # Читаем файл и извлекаем данные о функциях
    code = read_c_file(args.file, encoding=args.encoding)
    functions_data = extract_function_data(code, encoding=args.encoding)

    # Группируем данные по функциям (основные записи и сегменты)
    main_functions = [f for f in functions_data if "name" in f]
    segments = [f for f in functions_data if "segment_index" in f]

    for func in main_functions:
        print("\n" + "=" * 80)
        print(f"Функция: {func['name']}")
        print("=" * 80)
        print(f"  Сигнатура: {func['signature']}")
        print(f"  Параметры: {', '.join(func['parameters']) if func['parameters'] else '-'}")
        print(f"  Глобальные переменные (вход): {', '.join(func['globals_used']) if func['globals_used'] else '-'}")
        print(f"  Глобальные переменные (выход): {', '.join(func['globals_written']) if func['globals_written'] else '-'}")
        print(f"  Возвращаемые значения: {', '.join(func['returns']) if func['returns'] else '-'}")
        print(f"  Указатели-параметры: {', '.join(func['pointer_parameters']) if func['pointer_parameters'] else '-'}")
        print(f"  Константы: {', '.join(func['constants']) if func['constants'] else '-'}")
        print(f"  Вспомогательные функции: {', '.join(func['helpers']) if func['helpers'] else '-'}")
        print(f"  Внутренние переменные: {', '.join(func['internal_variables']) if func['internal_variables'] else '-'}")
        print(f"  Тело функции: {func['body_lines']} строк")
        print(f"  Сегментов тела: {func['segment_total']}")

        # Полный текст функции
        full_text = func['signature'] + " {\n" + func['full_body'] + "\n}"
        print("\n  --- Полный текст функции ---")
        print(full_text)
        print("-" * 80)

        # Выводим сегменты, если функция была разбита
        if func['segment_total'] > 1:
            func_segments = [s for s in segments if s['signature'] == func['signature']]
            print(f"\n  --- Сегменты функции ({len(func_segments)} шт.) ---")
            for seg in func_segments:
                print(f"\n  Сегмент {seg['segment_index'] + 1} из {seg['segment_total']} ({seg['body_lines']} строк):")
                print("  " + "-" * 60)
                print(seg['segment_body'])
                print("  " + "-" * 60)
