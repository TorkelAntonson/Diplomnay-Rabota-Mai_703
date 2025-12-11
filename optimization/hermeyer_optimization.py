import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# СВЕРТКА ГЕРМЕЙЕРА С ВНЕШНИМИ ГРАНИЦАМИ
def germeier_convolution_with_bounds(criteria_matrix, weights, min_max, bounds):
    """
    Вычисление свертки Гермейера с внешними границами нормализации
    
    Parameters:
    -----------
    criteria_matrix : array-like, shape (n_models, n_criteria)
        Матрица значений критериев
    weights : array-like, shape (n_criteria,)
        Веса критериев (должны быть положительными и суммироваться в 1)
    min_max : array-like, shape (n_criteria,)
        Направление оптимизации: 
        1 - максимизация, 
        -1 - минимизация
    bounds : list of tuples, shape (n_criteria, 2)
        Границы для нормализации: [(min1, max1), (min2, max2), ...]
    
    Returns:
    --------
    scores : array-like, shape (n_models,)
        Значения свертки Гермейера для каждой модели
    norm_matrix : array-like, shape (n_models, n_criteria)
        Нормализованная матрица критериев
    """
    n_models, n_criteria = criteria_matrix.shape
    
    # Нормализуем веса
    weights = np.array(weights) / np.sum(weights)
    
    # Создаем матрицу нормализованных значений
    norm_matrix = np.zeros_like(criteria_matrix, dtype=float)
    
    for j in range(n_criteria):
        min_val, max_val = bounds[j]
        
        if min_max[j] == 1:  # максимизация
            if max_val != min_val:
                norm_matrix[:, j] = (criteria_matrix[:, j] - min_val) / (max_val - min_val)
            else:
                norm_matrix[:, j] = 1.0
        else:  # минимизация
            if max_val != min_val:
                norm_matrix[:, j] = (max_val - criteria_matrix[:, j]) / (max_val - min_val)
            else:
                norm_matrix[:, j] = 1.0
    
    # Ограничиваем значения в диапазоне [0, 1]
    norm_matrix = np.clip(norm_matrix, 0, 1)
    
    # Вычисляем свертку Гермейера: min(w_j * f_ij)
    germeier_scores = []
    for i in range(n_models):
        weighted_values = weights * norm_matrix[i, :]
        germeier_score = np.min(weighted_values)  # берем минимум
        germeier_scores.append(germeier_score)
    
    return np.array(germeier_scores), norm_matrix

# Подготовка данных для свертки Гермейера
# Матрица критериев
criteria_matrix = np.array([
    [87, 18.3, 11117, 14.19],   # T-lite-it-1.0
    [84, 31.9, 2116, 14.98],    # YandexGPT-5-Lite-8B-instruct
    [62, 19.4, 6886, 13.51],    # Mistral-7B-Instruct-v0.3
    [60, 23.7, 10883, 12.87],   # deepseek-coder-7b-instruct-v1.5
    [30, 18.4, 10883, 14.20]    # Qwen2.5-Coder-7B-Instruct
])

# Веса критериев (сумма = 1)
weights = np.array([0.3, 0.3, 0.2, 0.2])

# Направление оптимизации: 1 - максимизация, -1 - минимизация
min_max = np.array([1, 1, -1, -1])

# ВНЕШНИЕ ГРАНИЦЫ ДЛЯ НОРМАЛИЗАЦИИ
bounds = [
    (0, 100),       # Точность: от 0 до 100%
    (0, 50),        # Скорость токенов: от 0 до 50 токенов/сек
    (0, 20000),     # Токены израсходовано: от 0 до 20000
    (10, 16)        # GPU память: от 10 до 16 ГБ
]

# Вычисляем свертку Гермейера с внешними границами
germeier_scores, norm_matrix = germeier_convolution_with_bounds(
    criteria_matrix, weights, min_max, bounds
)

# Находим оптимальную модель (максимум свертки Гермейера)
optimal_idx = np.argmax(germeier_scores)
optimal_model = df.iloc[optimal_idx]

# Находим множество Парето для визуализации (используем исходные данные)
def find_pareto_front(criteria_matrix, min_max):
    """Нахождение множества Парето-оптимальных решений"""
    n_models = criteria_matrix.shape[0]
    pareto_front = np.ones(n_models, dtype=bool)
    
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                dominates = True
                for k in range(criteria_matrix.shape[1]):
                    if min_max[k] == 1:  # максимизация
                        if criteria_matrix[j, k] < criteria_matrix[i, k]:
                            dominates = False
                            break
                    else:  # минимизация
                        if criteria_matrix[j, k] > criteria_matrix[i, k]:
                            dominates = False
                            break
                if dominates:
                    pareto_front[i] = False
                    break
    
    return pareto_front

pareto_mask = find_pareto_front(criteria_matrix, min_max)
pareto_indices = np.where(pareto_mask)[0]

# Выводим результаты
print("=" * 80)
print("ОПТИМАЛЬНАЯ МОДЕЛЬ ПО СВЕРТКЕ ГЕРМЕЙЕРА (с внешними границами)")
print("=" * 80)
print(f"Модель: {optimal_model['Модель']}")
print(f"Точность: {optimal_model['Точность']}%")
print(f"Скорость токенов: {optimal_model['Скорость_токенов']} токен/сек")
print(f"Израсходовано токенов: {optimal_model['Токены_израсходовано']}")
print(f"GPU память: {optimal_model['GPU_память']} ГБ")
print(f"Значение свертки Гермейера: {germeier_scores[optimal_idx]:.6f}")
print("\nГраницы нормализации:")
print(f"  Точность: {bounds[0][0]} - {bounds[0][1]}%")
print(f"  Скорость токенов: {bounds[1][0]} - {bounds[1][1]} токен/сек")
print(f"  Токены израсходовано: {bounds[2][0]} - {bounds[2][1]}")
print(f"  GPU память: {bounds[3][0]} - {bounds[3][1]} ГБ")
print("=" * 80)

print("\nНОРМАЛИЗОВАННЫЕ ЗНАЧЕНИЯ КРИТЕРИЕВ (по внешним границам):")
print("=" * 80)
print(f"{'Модель':<35} {'Точность':<12} {'Скорость':<12} {'Токены':<12} {'GPU':<12} {'Свертка':<12}")
print("-" * 80)

for i in range(len(df)):
    model_name = df['Модель'].iloc[i]
    if len(model_name) > 30:
        model_name = model_name[:27] + "..."
    
    is_optimal = "★" if i == optimal_idx else ""
    
    print(f"{model_name:<35} {is_optimal:<1} "
          f"{norm_matrix[i, 0]:<11.4f} {norm_matrix[i, 1]:<11.4f} "
          f"{norm_matrix[i, 2]:<11.4f} {norm_matrix[i, 3]:<11.4f} "
          f"{germeier_scores[i]:<11.6f}")

print("=" * 80)

print("\nДЕТАЛЬНЫЙ РАСЧЕТ ДЛЯ ОПТИМАЛЬНОЙ МОДЕЛИ:")
print("=" * 80)
print(f"Модель: {optimal_model['Модель']}")
print(f"Веса критериев: {weights}")
print("\nНормализованные значения (внешние границы):")
for j, (criterion, w, bound) in enumerate(zip(['Точность', 'Скорость', 'Токены', 'GPU'], 
                                             weights, bounds)):
    original_value = criteria_matrix[optimal_idx, j]
    norm_value = norm_matrix[optimal_idx, j]
    weighted_value = w * norm_value
    print(f"  {criterion}: {original_value} → {norm_value:.4f} (границы {bound[0]}-{bound[1]})")
    print(f"        × {w:.3f} = {weighted_value:.6f}")
print(f"\nСвертка Гермейера (минимум): {germeier_scores[optimal_idx]:.6f}")
print("=" * 80)

# Визуализация результатов
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Многокритериальная оптимизация методом свертки Гермейера (с внешними границами)', 
             fontsize=16, fontweight='bold')

# 1. График значений свертки Гермейера
ax1 = axes[0, 0]
bars = ax1.bar(range(len(df)), germeier_scores, color=['gold' if i == optimal_idx else 'skyblue' for i in range(len(df))])
ax1.set_xticks(range(len(df)))
ax1.set_xticklabels([name[:15] + "..." if len(name) > 15 else name for name in df['Модель']], rotation=45, ha='right')
ax1.set_ylabel('Значение свертки Гермейера', fontsize=12)
ax1.set_title('Свертка Гермейера по моделям', fontsize=13)
ax1.grid(True, alpha=0.3, axis='y')
for i, (bar, score) in enumerate(zip(bars, germeier_scores)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{score:.4f}', ha='center', va='bottom', fontsize=9)

# 2. Радарная диаграмма для оптимальной модели
ax2 = axes[0, 1]
criteria_names = ['Точность\n(↑)', 'Скорость\n(↑)', 'Токены\n(↓)', 'GPU\n(↓)']
N = len(criteria_names)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

values = norm_matrix[optimal_idx, :].tolist()
values += values[:1]

ax2.plot(angles, values, 'o-', linewidth=2, color='gold', label='Оптимальная модель')
ax2.fill(angles, values, alpha=0.25, color='gold')
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(criteria_names, fontsize=10)
ax2.set_ylim(0, 1)
ax2.set_title(f'Нормализованные критерии\n{optimal_model["Модель"][:20]}', fontsize=13)
ax2.grid(True)
ax2.legend(loc='upper right')

# 3. График весов критериев с границами
ax3 = axes[0, 2]
criteria_full_names = [f'Точность (↑)\n[{bounds[0][0]}-{bounds[0][1]}]', 
                      f'Скорость (↑)\n[{bounds[1][0]}-{bounds[1][1]}]', 
                      f'Токены (↓)\n[{bounds[2][0]}-{bounds[2][1]}]', 
                      f'GPU (↓)\n[{bounds[3][0]}-{bounds[3][1]}]']
bars3 = ax3.barh(criteria_full_names, weights, color='lightgreen')
ax3.set_xlabel('Вес критерия', fontsize=12)
ax3.set_title('Веса критериев и границы нормализации', fontsize=13)
ax3.grid(True, alpha=0.3, axis='x')
for i, (bar, w) in enumerate(zip(bars3, weights)):
    ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f'{w:.3f}', ha='left', va='center', fontsize=10)

# 4. Точность vs Скорость токенов (Парето фронт)
ax4 = axes[1, 0]
pareto_points = []
for idx in pareto_indices:
    pareto_points.append([df['Точность'].iloc[idx], df['Скорость_токенов'].iloc[idx]])
pareto_points = np.array(pareto_points)

if len(pareto_points) > 2:
    hull = ConvexHull(pareto_points)
    for simplex in hull.simplices:
        ax4.plot(pareto_points[simplex, 0], pareto_points[simplex, 1], 'r--', alpha=0.7, linewidth=2)

colors = ['red' if i in pareto_indices else 'blue' for i in range(len(df))]
ax4.scatter(df['Точность'], df['Скорость_токенов'], c=colors, s=100, alpha=0.7, edgecolors='black')
ax4.scatter(optimal_model['Точность'], optimal_model['Скорость_токенов'], 
           c='gold', s=200, marker='*', edgecolors='black', linewidth=2, label='Оптимальная')

# Добавляем границы на график
ax4.axvline(x=bounds[0][0], color='gray', linestyle='--', alpha=0.3)
ax4.axvline(x=bounds[0][1], color='gray', linestyle='--', alpha=0.3)
ax4.axhline(y=bounds[1][0], color='gray', linestyle='--', alpha=0.3)
ax4.axhline(y=bounds[1][1], color='gray', linestyle='--', alpha=0.3)

for i, txt in enumerate(df['Модель']):
    if len(txt) > 20:
        txt = txt[:17] + "..."
    ax4.annotate(txt, (df['Точность'].iloc[i], df['Скорость_токенов'].iloc[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax4.set_xlabel('Точность, % (↑ лучше)', fontsize=12)
ax4.set_ylabel('Скорость токенов, токен/сек (↑ лучше)', fontsize=12)
ax4.set_title('Парето фронт: Точность vs Скорость', fontsize=13)
ax4.grid(True, alpha=0.3)
ax4.legend()

# 5. Точность vs GPU память
ax5 = axes[1, 1]
ax5.scatter(df['Точность'], df['GPU_память'], c=colors, s=100, alpha=0.7, edgecolors='black')
ax5.scatter(optimal_model['Точность'], optimal_model['GPU_память'], 
           c='gold', s=200, marker='*', edgecolors='black', linewidth=2, label='Оптимальная')

# Добавляем границы на график
ax5.axvline(x=bounds[0][0], color='gray', linestyle='--', alpha=0.3)
ax5.axvline(x=bounds[0][1], color='gray', linestyle='--', alpha=0.3)
ax5.axhline(y=bounds[3][0], color='gray', linestyle='--', alpha=0.3)
ax5.axhline(y=bounds[3][1], color='gray', linestyle='--', alpha=0.3)

for i, txt in enumerate(df['Модель']):
    if len(txt) > 20:
        txt = txt[:17] + "..."
    ax5.annotate(txt, (df['Точность'].iloc[i], df['GPU_память'].iloc[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax5.set_xlabel('Точность, % (↑ лучше)', fontsize=12)
ax5.set_ylabel('GPU память, ГБ (↓ лучше)', fontsize=12)
ax5.set_title('Точность vs GPU память', fontsize=13)
ax5.grid(True, alpha=0.3)
ax5.legend()

# 6. Сравнение нормализованных критериев для всех моделей
ax6 = axes[1, 2]
x = np.arange(len(criteria_names))
width = 0.15

for i in range(len(df)):
    offset = (i - len(df)/2) * width
    label = df['Модель'].iloc[i][:10] + "..." if len(df['Модель'].iloc[i]) > 10 else df['Модель'].iloc[i]
    color = 'gold' if i == optimal_idx else plt.cm.Set2(i/len(df))
    ax6.bar(x + offset, norm_matrix[i, :], width, label=label, color=color, alpha=0.8)

ax6.set_xlabel('Критерии', fontsize=12)
ax6.set_ylabel('Нормализованное значение', fontsize=12)
ax6.set_title('Сравнение нормализованных критериев (внешние границы)', fontsize=13)
ax6.set_xticks(x)
ax6.set_xticklabels(['Точн.', 'Скор.', 'Токен.', 'GPU'], fontsize=10)
ax6.grid(True, alpha=0.3, axis='y')
ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# Вывод Парето-оптимальных моделей
print("\nПАРЕТО-ОПТИМАЛЬНЫЕ МОДЕЛИ:")
print("=" * 80)
for idx in pareto_indices:
    model_name = df['Модель'].iloc[idx]
    is_optimal = "★ ОПТИМАЛЬНАЯ ПО ГЕРМЕЙЕРУ ★" if idx == optimal_idx else ""
    print(f"{model_name:<35} Точность={df['Точность'].iloc[idx]:>3}%, "
          f"Скорость={df['Скорость_токенов'].iloc[idx]:>5.1f}токен/сек, "
          f"Гермейер={germeier_scores[idx]:.6f} {is_optimal}")
print("=" * 80)

# Анализ чувствительности весов
print("\nАНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ К ВЕСАМ (с внешними границами):")
print("=" * 80)

# Проверяем разные наборы весов
weight_sets = [
    ([0.4, 0.3, 0.2, 0.1], "Сбалансированный с упором на скорость"),
    ([0.3, 0.4, 0.2, 0.1], "Упор на скорость токенов"),
    ([0.5, 0.2, 0.15, 0.15], "Упор на точность"),
    ([0.25, 0.25, 0.25, 0.25], "Равномерное распределение"),
    ([0.6, 0.1, 0.2, 0.1], "Сильный упор на точность")
]

for ws, description in weight_sets:
    scores, _ = germeier_convolution_with_bounds(criteria_matrix, ws, min_max, bounds)
    opt_idx = np.argmax(scores)
    print(f"\n{description}:")
    print(f"  Веса: {ws}")
    print(f"  Оптимальная: {df['Модель'].iloc[opt_idx]}")
    print(f"  Значение свертки: {scores[opt_idx]:.6f}")
print("=" * 80)

# Дополнительный вывод всех моделей с ранжированием
print("\n" + "=" * 80)
print("РАНЖИРОВАНИЕ ВСЕХ МОДЕЛЕЙ (с внешними границами):")
print("=" * 80)

# Сортируем модели по убыванию значения свертки Гермейера
sorted_indices = np.argsort(-germeier_scores)
for rank, idx in enumerate(sorted_indices, 1):
    model_name = df['Модель'].iloc[idx]
    is_optimal = "★" if idx == optimal_idx else ""
    print(f"{rank:2}. {model_name:<35} {is_optimal:<1} "
          f"Гермейер={germeier_scores[idx]:.6f} "
          f"Точность={df['Точность'].iloc[idx]:>3}% "
          f"Скорость={df['Скорость_токенов'].iloc[idx]:>5.1f}токен/сек")
print("=" * 80)