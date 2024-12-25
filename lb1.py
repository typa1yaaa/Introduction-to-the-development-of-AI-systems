import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Функция для генерации простого облака точек
def generate_simple_cloud(center, n_points, scale):
    return np.random.normal(loc=center, scale=scale, size=(n_points, 5))

# Функция для генерации спирального облака точек
def generate_spiral_cloud(center, n_points, turns, height, scale):
    theta = np.linspace(0, 2 * np.pi * turns, n_points)
    z = np.linspace(center[2] - height / 2, center[2] + height / 2, n_points)
    radius = scale * theta / (2 * np.pi * turns)
    x = center[0] + radius * np.cos(theta) + np.random.normal(0, radius * 0.05, n_points)
    y = center[1] + radius * np.sin(theta) + np.random.normal(0, radius * 0.05, n_points)
    w = np.random.normal(center[3], radius * 0.1, n_points)
    v = np.random.normal(center[4], radius * 0.1, n_points)
    return np.vstack((x, y, z, w, v)).T

# Функция для генерации параболического облака точек
def generate_parabola_cloud(center, n_points, scale, height):
    z = np.linspace(center[2] - height / 2, center[2] + height / 2, n_points)
    x = scale * z**2 + center[0] + np.random.normal(0, scale * 0.5, n_points)
    y = scale * z**2 + center[1] + np.random.normal(0, scale * 0.5, n_points)
    w = np.random.normal(center[3], scale, n_points)
    v = np.random.normal(center[4], scale, n_points)
    return np.vstack((x, y, z, w, v)).T

# Функция для генерации сферического облака точек
def generate_spherical_cloud(center, n_points, radius, scale):
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    costheta = np.random.uniform(-1, 1, n_points)
    theta = np.arccos(costheta)
    r = radius + np.random.normal(0, scale, n_points)
    x = center[0] + r * np.sin(theta) * np.cos(phi)
    y = center[1] + r * np.sin(theta) * np.sin(phi)
    z = center[2] + r * np.cos(theta)
    w = np.random.normal(center[3], scale, n_points)
    v = np.random.normal(center[4], scale, n_points)
    return np.vstack((x, y, z, w, v)).T

# Функция для генерации цилиндрического облака точек
def generate_cylindrical_cloud(center, n_points, radius, height, scale):
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    r = radius + np.random.normal(0, scale, n_points)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    z = np.random.uniform(center[2] - height / 2, center[2] + height / 2, n_points) + np.random.normal(0, scale, n_points)
    w = np.random.normal(center[3], scale, n_points)
    v = np.random.normal(center[4], scale, n_points)
    return np.vstack((x, y, z, w, v)).T

# Параметры кол-ва точек в облаке и размаха генерации точек
n_points = 50
scale = 1

# Определение центров облаков
centers = {
    'Облако точек': [0, 0, 0, 0, 0],
    'Спираль': [25, 0, 0, 0, 0],
    'Парабола': [0, 25, 0, 0, 0],
    'Сфера': [0, 0, 25, 0, 0],
    'Цилиндр': [0, 0, 0, 25, 0]
}

# Генерация облаков различных функций
simple_cloud = generate_simple_cloud(centers['Облако точек'], n_points, scale)
spiral_cloud = generate_spiral_cloud(centers['Спираль'], n_points, turns=3, height=3, scale=1)
parabola_cloud = generate_parabola_cloud(centers['Парабола'], n_points, scale=1, height=3)
spherical_cloud = generate_spherical_cloud(centers['Сфера'], n_points, radius=2, scale=1)
cylindrical_cloud = generate_cylindrical_cloud(centers['Цилиндр'], n_points, radius=3, height=3, scale=0.2)

all_points = np.vstack([simple_cloud, spiral_cloud, parabola_cloud, spherical_cloud, cylindrical_cloud])
labels_true = np.array([0]*n_points + [1]*n_points + [2]*n_points + [3]*n_points + [4]*n_points)

# Применяем РСА для каждого облака точек
pca = PCA(n_components=2)
all_points_vector = pca.fit_transform(all_points)

# Вычисляем центры облаков
centers_arr = np.array(list(centers.values()))
centers_arr_pca = pca.transform(centers_arr)

# Вычисляем необходимое кол-во кластеров с помощью метода логтя
elbow_method = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(all_points_vector)
    elbow_method.append(kmeans.inertia_)

# Кластеризация с помощью KMeans (n_clusters=5)
kmeans = KMeans(n_clusters=5)
kmeans_value = kmeans.fit_predict(all_points_vector)
kmeans_center = kmeans.cluster_centers_

# Визуализация
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# График 1: PCA 
colors = ['b', 'g', 'r', 'c', 'm']
labels = ['Облако точек', 'Спираль', 'Парабола', 'Сфера', 'Цилиндр']

for i in range(5):
    ax1.scatter(
        all_points_vector[labels_true == i, 0],
        all_points_vector[labels_true == i, 1],
        label=labels[i],
        color=colors[i],
        alpha=0.6,
        edgecolor='k'
    )

# Отображение исходных центров облаков
ax1.scatter(
    centers_arr_pca[:, 0],
    centers_arr_pca[:, 1],
    marker='X',
    s=200,
    color='yellow',
    edgecolor='k',
    label='Центры облаков'
)

ax1.set_title("Применение PCA для облаков точек")
ax1.set_xlabel('Первая главная компонента')
ax1.set_ylabel('Вторая главная компонента')
ax1.legend()
ax1.grid(True)

# График 2: Кластеризация KMeans
scatter = ax2.scatter(
    all_points_vector[:, 0],
    all_points_vector[:, 1],
    c=kmeans_value,
    s=50,
    cmap='viridis',
    alpha=0.6,
    edgecolor='k'
)

# Отображение кластерных центров
ax2.scatter(
    kmeans_center[:, 0],
    kmeans_center[:, 1],
    marker='X',
    s=200,
    color='red',
    edgecolor='k',
    label='Центры кластеров'
)

ax2.set_title('Кластеризация KMeans с 5 кластерами')
ax2.set_xlabel('Первая главная компонента')
ax2.set_ylabel('Вторая главная компонента')
ax2.grid(True)

legend1 = ax2.legend(*scatter.legend_elements(),
                    title="Кластеры")
ax2.add_artist(legend1)
ax2.legend(loc='upper right')

# График 3: Метод локтя
ax3.plot(range(1, 11), elbow_method, marker='o')
ax3.set_title('Метод локтя')
ax3.set_xlabel('Число кластеров (k)')
ax3.set_ylabel('Сумма квадратов расстояний (Inertia)')
ax3.grid(True)

plt.tight_layout()
plt.show()

# Проверка оптимальности кластеризации
silhouette_avg = silhouette_score(all_points_vector, kmeans_value)
print(f"Средний силуэтный коэффициент: {silhouette_avg:.3f}")