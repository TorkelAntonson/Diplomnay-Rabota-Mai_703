import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

# Данные моделей
data = {
    'Модель': [
        'T-lite-it-1.0',
        'YandexGPT-5-Lite-8B-instruct', 
        'Mistral-7B-Instruct-v0.3',
        'deepseek-coder-7b-instruct-v1.5',
        'Qwen2.5-Coder-7B-Instruct'
    ],
    'Точность': [87, 84, 62, 60, 30],
    'Скорость_токенов': [18.3, 31.9, 19.4, 23.7, 18.4],
    'Токены_израсходовано': [11117, 2116, 6886, 10883, 10883],
    'GPU_память': [14.19, 14.98, 13.51, 12.87, 14.20]
}

df = pd.DataFrame(data)

# Нормализация критериев (все к максимизации)
df_norm = pd.DataFrame()

# Преобразуем к максимизации
df_norm['Точность_norm'] = df['Точность'] / 100.0
df_norm['Скорость_токенов_norm'] = df['Скорость_токенов'] / max(df['Скорость_токенов'])
df_norm['Токены_norm'] = 1 / (df['Токены_израсходовано'] / max(df['Токены_израсходовано']))
df_norm['GPU_norm'] = 1 / (df['GPU_память'] / max(df['GPU_память']))

# МЕТОД ВЗВЕШЕННОЙ СУММЫ НА ОСНОВЕ ПОПАРНЫХ СРАВНЕНИЙ
# Шаг 1: Определение важности критериев с помощью попарных сравнений
def pairwise_comparison_weights():
    """Генерация весов критериев методом попарных сравнений"""
    
    # Названия критериев
    criteria_names = ['Точность', 'Скорость токенов', 'Расход токенов', 'GPU память']
    
    # Матрица попарных сравнений (шкала Саати от 1 до 9)
    # 1 - одинаково важны, 3 - умеренно важнее, 5 - сильно важнее, 
    # 7 - значительно важнее, 9 - абсолютно важнее
    # Равномерное распределение с легким упором на точность и скорость
    pairwise_matrix = np.array([
        # Точность vs [Точность, Скорость, Расход, GPU]
        [1,    1,    5,    4],   # Точность: равна скорости, сильно важнее расхода и GPU
        # Скорость токенов vs [Точность, Скорость, Расход, GPU]
        [1,    1,    5,    4],   # Скорость: равна точности, сильно важнее расхода и GPU
        # Расход токенов vs [Точность, Скорость, Расход, GPU]
        [1/5,  1/5,  1,    1],   # Расход: сильно менее важен чем точность и скорость, равен GPU
        # GPU память vs [Точность, Скорость, Расход, GPU]
        [1/4,  1/4,  1,    1]    # GPU: сильно менее важен чем точность и скорость, равен расходу
    ])
    
    # Вычисление весов методом собственного вектора
    eigenvalues, eigenvectors = np.linalg.eig(pairwise_matrix)
    max_eigenvalue_idx = np.argmax(np.real(eigenvalues))
    weights = np.real(eigenvectors[:, max_eigenvalue_idx])
    weights = weights / np.sum(weights)  # Нормализация
    
    # Расчет индекса согласованности (Consistency Index)
    n = len(criteria_names)
    CI = (np.real(eigenvalues[max_eigenvalue_idx]) - n) / (n - 1)
    RI = 0.9  # Случайный индекс для n=4
    CR = CI / RI  # Коэффициент согласованности
    
    print("=" * 70)
    print("МАТРИЦА ПОПАРНЫХ СРАВНЕНИЙ КРИТЕРИЕВ:")
    print("=" * 70)
    print("Критерии: 1-Точность, 2-Скорость токенов, 3-Расход токенов, 4-GPU память")
    print("\nМатрица попарных сравнений (шкала Саати):")
    for i in range(n):
        row_str = " ".join([f"{pairwise_matrix[i, j]:5.2f}" for j in range(n)])
        print(f"Критерий {i+1}: {row_str}")
    
    print(f"\nИндекс согласованности (CI): {CI:.3f}")
    print(f"Коэффициент согласованности (CR): {CR:.3f}")
    print("CR < 0.1 - матрица согласована" if CR < 0.1 else "Внимание: CR > 0.1 - рекомендуется пересмотреть сравнения")
    
    return weights, criteria_names

# Шаг 2: Расчет взвешенных оценок для каждой модели
def calculate_weighted_scores(df_norm, weights):
    """Расчет взвешенных оценок для каждой модели"""
    scores = np.dot(df_norm.values, weights)
    return scores

# Шаг 3: Ранжирование моделей
def rank_models(df, scores):
    """Ранжирование моделей по взвешенной сумме"""
    df_ranked = df.copy()
    df_ranked['Взвешенный_балл'] = scores
    df_ranked['Ранг'] = df_ranked['Взвешенный_балл'].rank(ascending=False, method='dense').astype(int)
    df_ranked = df_ranked.sort_values('Взвешенный_балл', ascending=False)
    return df_ranked

# Шаг 4: Визуализация результатов (обновленная - без удаленных графиков)
def visualize_results(df, df_ranked, weights, criteria_names):
    """Визуализация результатов взвешенной суммы"""
    
    fig = plt.figure(figsize=(16, 8))
    
    # 1. Круговой график весов критериев
    ax1 = plt.subplot(1, 3, 1)
    colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))
    wedges, texts, autotexts = ax1.pie(weights, labels=criteria_names, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax1.set_title('Распределение весов критериев', fontsize=14, fontweight='bold')
    
    # 2. Столбчатая диаграмма взвешенных баллов
    ax2 = plt.subplot(1, 3, 2)
    bars = ax2.barh(range(len(df_ranked)), df_ranked['Взвешенный_балл'], 
                    color=plt.cm.viridis(np.linspace(0.3, 0.9, len(df_ranked))))
    ax2.set_yticks(range(len(df_ranked)))
    ax2.set_yticklabels(df_ranked['Модель'].values)
    ax2.set_xlabel('Взвешенный балл')
    ax2.set_title('Ранжирование моделей по взвешенной сумме', fontsize=14, fontweight='bold')
    
    # Добавляем значения на столбцы
    for i, (bar, score) in enumerate(zip(bars, df_ranked['Взвешенный_балл'])):
        ax2.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=9)
    
    # 3. Heatmap нормализованных значений
    ax3 = plt.subplot(1, 3, 3)
    norm_values = df_norm.values.T
    im = ax3.imshow(norm_values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels(df['Модель'], rotation=45, ha='right')
    ax3.set_yticks(range(len(criteria_names)))
    ax3.set_yticklabels(criteria_names)
    ax3.set_title('Нормализованные значения критериев', fontsize=14, fontweight='bold')
    
    # Добавляем цветовую шкалу
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

# Основной процесс выполнения
print("=" * 70)
print("МЕТОД ВЗВЕШЕННОЙ СУММЫ НА ОСНОВЕ ПОПАРНЫХ СРАВНЕНИЙ")
print("=" * 70)

# Получаем веса критериев
weights, criteria_names = pairwise_comparison_weights()

print("\n" + "=" * 70)
print("РАСЧЕТНЫЕ ВЕСА КРИТЕРИЕВ:")
print("=" * 70)
for i, (name, weight) in enumerate(zip(criteria_names, weights)):
    print(f"{i+1}. {name}: {weight:.3f} ({weight*100:.1f}%)")

# Рассчитываем взвешенные оценки
scores = calculate_weighted_scores(df_norm, weights)

# Ранжируем модели
df_ranked = rank_models(df, scores)

print("\n" + "=" * 70)
print("ИТОГОВОЕ РАНЖИРОВАНИЕ МОДЕЛЕЙ:")
print("=" * 70)
for i, (_, row) in enumerate(df_ranked.iterrows(), 1):
    optimal_mark = " ★ ЛУЧШАЯ МОДЕЛЬ ★" if i == 1 else ""
    print(f"{i}. {row['Модель']}:")
    print(f"   Точность: {row['Точность']}% | Скорость: {row['Скорость_токенов']}токен/сек")
    print(f"   Токены: {row['Токены_израсходовано']} | GPU: {row['GPU_память']}ГБ")
    print(f"   Взвешенный балл: {row['Взвешенный_балл']:.3f}{optimal_mark}")
    print()

# Детальный анализ лучшей модели
print("=" * 70)
print("АНАЛИЗ ЛУЧШЕЙ МОДЕЛИ:")
print("=" * 70)
best_model = df_ranked.iloc[0]
print(f"Модель: {best_model['Модель']}")
print(f"Общий взвешенный балл: {best_model['Взвешенный_балл']:.3f}")
print("\nПоказатели по критериям:")
print(f"  • Точность: {best_model['Точность']}% (нормализовано: {df_norm.iloc[best_model.name]['Точность_norm']:.3f}, вес: {weights[0]:.2%})")
print(f"  • Скорость токенов: {best_model['Скорость_токенов']}токен/сек (нормализовано: {df_norm.iloc[best_model.name]['Скорость_токенов_norm']:.3f}, вес: {weights[1]:.2%})")
print(f"  • Расход токенов: {best_model['Токены_израсходовано']} (нормализовано: {df_norm.iloc[best_model.name]['Токены_norm']:.3f}, вес: {weights[2]:.2%})")
print(f"  • GPU память: {best_model['GPU_память']}ГБ (нормализовано: {df_norm.iloc[best_model.name]['GPU_norm']:.3f}, вес: {weights[3]:.2%})")

print("\nРасчет взвешенной суммы:")
for i, criterion in enumerate(criteria_names):
    norm_value = df_norm.iloc[best_model.name, i]
    weighted = norm_value * weights[i]
    print(f"  {criterion}: {norm_value:.3f} × {weights[i]:.3f} = {weighted:.3f}")

# Визуализация результатов
visualize_results(df, df_ranked, weights, criteria_names)

# Дополнительный анализ чувствительности
print("\n" + "=" * 70)
print("АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ К ИЗМЕНЕНИЮ ВЕСОВ:")
print("=" * 70)
print("Изменение веса точности ±20%:")

original_weight = weights[0]
for change in [-0.2, -0.1, 0, 0.1, 0.2]:
    new_weights = weights.copy()
    new_weights[0] = max(0.01, original_weight * (1 + change))
    # Перераспределяем оставшиеся веса пропорционально
    remaining_sum = 1 - new_weights[0]
    original_remaining_sum = 1 - original_weight
    for j in range(1, len(new_weights)):
        new_weights[j] = weights[j] * (remaining_sum / original_remaining_sum)
    
    new_scores = calculate_weighted_scores(df_norm, new_weights)
    new_ranked = rank_models(df, new_scores)
    best_model_new = new_ranked.iloc[0]['Модель']
    
    change_pct = change * 100
    print(f"  • Вес точности: {new_weights[0]:.1%} ({change_pct:+.0f}%): Лучшая модель - {best_model_new}")

print("\n" + "=" * 70)
