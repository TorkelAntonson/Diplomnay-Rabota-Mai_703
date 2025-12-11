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

# Преобразуем к максимизации (все критерии должны быть "чем больше, тем лучше")
df_norm['Точность_norm'] = df['Точность'] / 100.0  # уже максимизация
df_norm['Скорость_токенов_norm'] = df['Скорость_токенов'] / max(df['Скорость_токенов'])
df_norm['Токены_norm'] = 1 / (df['Токены_израсходовано'] / max(df['Токены_израсходовано']))
df_norm['GPU_norm'] = 1 / (df['GPU_память'] / max(df['GPU_память']))

# Многокритериальная оптимизация - поиск множества Парето
def find_pareto_front(models_data, models_names):
    """Нахождение множества Парето-оптимальных решений"""
    n = len(models_data)
    pareto_front = np.ones(n, dtype=bool)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Проверяем, доминирует ли j над i
                if np.all(models_data[j] >= models_data[i]) and np.any(models_data[j] > models_data[i]):
                    pareto_front[i] = False
                    break
    
    return pareto_front

# Получаем нормализованные данные для оптимизации
models_data = df_norm.values
models_names = df['Модель'].values

# Находим Парето-фронт
pareto_mask = find_pareto_front(models_data, models_names)
pareto_indices = np.where(pareto_mask)[0]
non_pareto_indices = np.where(~pareto_mask)[0]

# Вычисляем взвешенную сумму для ранжирования Парето-оптимальных решений
# Веса для 4 критериев: Точность, Скорость токенов, Токены (обратный), GPU (обратный)
weights = np.array([0.4, 0.3, 0.2, 0.1])  # веса критериев
scores = np.dot(models_data, weights)

# Находим оптимальную модель (с максимальным взвешенным баллом среди Парето optimal)
optimal_idx = pareto_indices[np.argmax(scores[pareto_indices])]
optimal_model = df.iloc[optimal_idx]

# Вывод всех моделей с пометкой Парето-оптимальности
print("=" * 100)
print("ВСЕ МОДЕЛИ И ИХ ПАРЕТО-ОПТИМАЛЬНОСТЬ:")
print("=" * 100)

for i in range(len(df)):
    model_name = df['Модель'].iloc[i]
    is_pareto = i in pareto_indices
    is_optimal = i == optimal_idx
    
    if is_optimal:
        status = "★ НАИБОЛЕЕ ОПТИМАЛЬНАЯ (ПАРЕТО) ★"
    elif is_pareto:
        status = "✓ ПАРЕТО-ОПТИМАЛЬНАЯ"
    else:
        status = "✗ НЕ ПАРЕТО-ОПТИМАЛЬНАЯ"
    
    print(f"\n{model_name}: {status}")
    print(f"  Точность: {df['Точность'].iloc[i]}%")
    print(f"  Скорость токенов: {df['Скорость_токенов'].iloc[i]} токен/сек")
    print(f"  Израсходовано токенов: {df['Токены_израсходовано'].iloc[i]}")
    print(f"  GPU память: {df['GPU_память'].iloc[i]} ГБ")
    print(f"  Взвешенный балл: {scores[i]:.3f}")

print("\n" + "=" * 100)
print("КРАТКАЯ СВОДКА ПО ПАРЕТО-ОПТИМАЛЬНЫМ МОДЕЛЯМ:")
print("=" * 100)

print(f"\nВсего моделей: {len(df)}")
print(f"Парето-оптимальных моделей: {len(pareto_indices)}")
print(f"Не Парето-оптимальных моделей: {len(non_pareto_indices)}")

print("\n" + "=" * 100)
print("НАИБОЛЕЕ ОПТИМАЛЬНАЯ МОДЕЛЬ (ПО ВЗВЕШЕННОЙ СУММЕ):")
print("=" * 100)
print(f"Модель: {optimal_model['Модель']}")
print(f"Точность: {optimal_model['Точность']}%")
print(f"Скорость токенов: {optimal_model['Скорость_токенов']} токен/сек")
print(f"Израсходовано токенов: {optimal_model['Токены_израсходовано']}")
print(f"GPU память: {optimal_model['GPU_память']} ГБ")
print(f"Взвешенный балл: {scores[optimal_idx]:.3f}")
print("=" * 100)

# Создаем графики 2D проекций множества Парето
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Множество Парето для многокритериальной оптимизации моделей', fontsize=16)

# Цвета для точек
colors = ['red' if i in pareto_indices else 'blue' for i in range(len(df))]
pareto_colors = ['red'] * len(pareto_indices)

# 1. Точность vs Скорость токенов
ax1 = axes[0, 0]
pareto_points_1 = []
for idx in pareto_indices:
    pareto_points_1.append([df['Точность'].iloc[idx], df['Скорость_токенов'].iloc[idx]])
pareto_points_1 = np.array(pareto_points_1)

if len(pareto_points_1) > 2:
    hull_1 = ConvexHull(pareto_points_1)
    for simplex in hull_1.simplices:
        ax1.plot(pareto_points_1[simplex, 0], pareto_points_1[simplex, 1], 'r--', alpha=0.7, linewidth=2)

ax1.scatter(df['Точность'], df['Скорость_токенов'], c=colors, s=100, alpha=0.7, edgecolors='black')
ax1.scatter(df['Точность'].iloc[optimal_idx], df['Скорость_токенов'].iloc[optimal_idx], 
           c='gold', s=200, marker='*', edgecolors='black', linewidth=2, label='Наиболее оптимальная')

for i, txt in enumerate(df['Модель']):
    offset_x = 5 if i != optimal_idx else 10
    offset_y = 5 if i != optimal_idx else 10
    ax1.annotate(txt, (df['Точность'].iloc[i], df['Скорость_токенов'].iloc[i]), 
                xytext=(offset_x, offset_y), textcoords='offset points', fontsize=8,
                fontweight='bold' if i in pareto_indices else 'normal')
ax1.set_xlabel('Точность, % (↑ лучше)', fontsize=12)
ax1.set_ylabel('Скорость токенов, токен/сек (↑ лучше)', fontsize=12)
ax1.set_title('Точность vs Скорость токенов', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Точность vs GPU память
ax2 = axes[0, 1]
pareto_points_2 = []
for idx in pareto_indices:
    pareto_points_2.append([df['Точность'].iloc[idx], df['GPU_память'].iloc[idx]])
pareto_points_2 = np.array(pareto_points_2)

if len(pareto_points_2) > 2:
    hull_2 = ConvexHull(pareto_points_2)
    for simplex in hull_2.simplices:
        ax2.plot(pareto_points_2[simplex, 0], pareto_points_2[simplex, 1], 'r--', alpha=0.7, linewidth=2)

ax2.scatter(df['Точность'], df['GPU_память'], c=colors, s=100, alpha=0.7, edgecolors='black')
ax2.scatter(df['Точность'].iloc[optimal_idx], df['GPU_память'].iloc[optimal_idx], 
           c='gold', s=200, marker='*', edgecolors='black', linewidth=2, label='Наиболее оптимальная')

for i, txt in enumerate(df['Модель']):
    offset_x = 5 if i != optimal_idx else 10
    offset_y = 5 if i != optimal_idx else 10
    ax2.annotate(txt, (df['Точность'].iloc[i], df['GPU_память'].iloc[i]), 
                xytext=(offset_x, offset_y), textcoords='offset points', fontsize=8,
                fontweight='bold' if i in pareto_indices else 'normal')
ax2.set_xlabel('Точность, % (↑ лучше)', fontsize=12)
ax2.set_ylabel('GPU память, ГБ (↓ лучше)', fontsize=12)
ax2.set_title('Точность vs GPU память', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. Скорость токенов vs GPU память
ax3 = axes[1, 0]
pareto_points_3 = []
for idx in pareto_indices:
    pareto_points_3.append([df['Скорость_токенов'].iloc[idx], df['GPU_память'].iloc[idx]])
pareto_points_3 = np.array(pareto_points_3)

if len(pareto_points_3) > 2:
    hull_3 = ConvexHull(pareto_points_3)
    for simplex in hull_3.simplices:
        ax3.plot(pareto_points_3[simplex, 0], pareto_points_3[simplex, 1], 'r--', alpha=0.7, linewidth=2)

ax3.scatter(df['Скорость_токенов'], df['GPU_память'], c=colors, s=100, alpha=0.7, edgecolors='black')
ax3.scatter(df['Скорость_токенов'].iloc[optimal_idx], df['GPU_память'].iloc[optimal_idx], 
           c='gold', s=200, marker='*', edgecolors='black', linewidth=2, label='Наиболее оптимальная')

for i, txt in enumerate(df['Модель']):
    offset_x = 5 if i != optimal_idx else 10
    offset_y = 5 if i != optimal_idx else 10
    ax3.annotate(txt, (df['Скорость_токенов'].iloc[i], df['GPU_память'].iloc[i]), 
                xytext=(offset_x, offset_y), textcoords='offset points', fontsize=8,
                fontweight='bold' if i in pareto_indices else 'normal')
ax3.set_xlabel('Скорость токенов, токен/сек (↑ лучше)', fontsize=12)
ax3.set_ylabel('GPU память, ГБ (↓ лучше)', fontsize=12)
ax3.set_title('Скорость токенов vs GPU память', fontsize=13)
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. Точность vs Израсходовано токенов
ax4 = axes[1, 1]
pareto_points_4 = []
for idx in pareto_indices:
    pareto_points_4.append([df['Точность'].iloc[idx], df['Токены_израсходовано'].iloc[idx]])
pareto_points_4 = np.array(pareto_points_4)

if len(pareto_points_4) > 2:
    hull_4 = ConvexHull(pareto_points_4)
    for simplex in hull_4.simplices:
        ax4.plot(pareto_points_4[simplex, 0], pareto_points_4[simplex, 1], 'r--', alpha=0.7, linewidth=2)

ax4.scatter(df['Точность'], df['Токены_израсходовано'], c=colors, s=100, alpha=0.7, edgecolors='black')
ax4.scatter(df['Точность'].iloc[optimal_idx], df['Токены_израсходовано'].iloc[optimal_idx], 
           c='gold', s=200, marker='*', edgecolors='black', linewidth=2, label='Наиболее оптимальная')

for i, txt in enumerate(df['Модель']):
    offset_x = 5 if i != optimal_idx else 10
    offset_y = 5 if i != optimal_idx else 10
    ax4.annotate(txt, (df['Точность'].iloc[i], df['Токены_израсходовано'].iloc[i]), 
                xytext=(offset_x, offset_y), textcoords='offset points', fontsize=8,
                fontweight='bold' if i in pareto_indices else 'normal')
ax4.set_xlabel('Точность, % (↑ лучше)', fontsize=12)
ax4.set_ylabel('Израсходовано токенов (↓ лучше)', fontsize=12)
ax4.set_title('Точность vs Израсходовано токенов', fontsize=13)
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.show()

# Подробная информация о Парето-оптимальных моделях
print("\n" + "=" * 100)
print("ПОДРОБНАЯ ИНФОРМАЦИЯ О ПАРЕТО-ОПТИМАЛЬНЫХ МОДЕЛЯХ:")
print("=" * 100)

# Сортируем Парето-оптимальные модели по взвешенному баллу (по убыванию)
sorted_pareto_indices = pareto_indices[np.argsort(-scores[pareto_indices])]

for rank, idx in enumerate(sorted_pareto_indices, 1):
    model_name = df['Модель'].iloc[idx]
    is_optimal = "★ НАИБОЛЕЕ ОПТИМАЛЬНАЯ" if idx == optimal_idx else f"№{rank}"
    
    print(f"\n{is_optimal}: {model_name}")
    print(f"  Точность: {df['Точность'].iloc[idx]}% (нормализовано: {df_norm.iloc[idx, 0]:.3f})")
    print(f"  Скорость токенов: {df['Скорость_токенов'].iloc[idx]} (нормализовано: {df_norm.iloc[idx, 1]:.3f})")
    print(f"  Израсходовано токенов: {df['Токены_израсходовано'].iloc[idx]} (нормализовано: {df_norm.iloc[idx, 2]:.3f})")
    print(f"  GPU память: {df['GPU_память'].iloc[idx]} ГБ (нормализовано: {df_norm.iloc[idx, 3]:.3f})")
    print(f"  Взвешенный балл: {scores[idx]:.3f}")

print("\n" + "=" * 100)
print("ОБЪЯСНЕНИЕ ПАРЕТО-ОПТИМАЛЬНОСТИ:")
print("=" * 100)
print("\nПарето-оптимальная модель - это модель, которая не может быть улучшена")
print("по одному критерию без ухудшения по другому критерию.")
print("\n✓ ПАРЕТО-ОПТИМАЛЬНЫЕ модели (красные точки) - нет других моделей, которые")
print("  лучше по всем критериям одновременно.")
print("\n✗ НЕ ПАРЕТО-ОПТИМАЛЬНЫЕ модели (синие точки) - существуют другие модели,")
print("  которые лучше по всем или некоторым критериям.")
print("\n★ НАИБОЛЕЕ ОПТИМАЛЬНАЯ модель (золотая звезда) - Парето-оптимальная модель")
print("  с максимальным взвешенным баллом при заданных весах критериев.")