import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

""" ИЗОБРАЖЕНИЕ """
# Загрузка изображения
image_0 = cv2.imread('snake.png', cv2.IMREAD_GRAYSCALE)
all_points0 = []
all_points = []
# Проверка, удалось ли загрузить изображение
if image_0 is None:
    print("Ошибка загрузки изображения.")
else:
    # Преобразуем изображение в бинарное, чтобы выделить черную фигуру на белом фоне
    _, binary_image = cv2.threshold(image_0, 127, 255, cv2.THRESH_BINARY_INV)

    # Находим контуры
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Убедимся, что мы получили хотя бы один контур
    if contours:
        # Выбираем первый контур
        first_contour = contours[0]

        # Получаем минимальные и максимальные координаты по оси x и y
        x_min, y_min = np.min(first_contour, axis=0)[0]
        x_max, y_max = np.max(first_contour, axis=0)[0]

        # Обрезаем изображение по этим координатам
        image = image_0[y_min:y_max, x_min:x_max]

        # Находим контуры в обрезанном изображении
        _, binary_cropped_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        cropped_contours, _ = cv2.findContours(binary_cropped_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Переводим все координаты в локальную систему обрезанного изображения
        for contour in cropped_contours:
            for point in contour:
                x, y = int(point[0][0]), int(point[0][1])
                all_points0.append((x, y))

        # Получаем высоту изображения
        image_height = image.shape[0]

        # Переводим все координаты в математическую систему
        for x, y in all_points0:
            y_new = image_height - y  # Инвертируем y-координату
            all_points.append((x, y_new))

        # Все точки первого контура в обрезанном изображении
        print("Точки первого контура в локальной системе координат обрезанного изображения:", all_points)

        # Для визуализации можно нарисовать контуры на обрезанном изображении
        image_with_contours = cv2.drawContours(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cropped_contours, -1, (0, 255, 0), 2)

        # Показываем результат
        cv2.imshow('Cropped Contours', image_with_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Контуры не найдены!")

"""ВХОДНЫЕ ДАННЫЕ"""

print('Введите наибольшую длину по оси x/y в сантиметрах:')
distance = float(input())
print('Введите массу фигуры в граммах:')
m = float(input())
print('Введите толщину фигуры в сантиметрах:')
d = float(input())
print('Введите плотность жидкости в г/см^3:')
ro = float(input())

""" ФУНКЦИИ """

def find_units(polygon):
    """Находит отношение пикселей на см"""
    max_x = max_y = -float('inf')
    min_x = min_y = float('inf')
    for point in polygon:
        x, y = point
        max_x = max(max_x, x)
        min_x = min(min_x, x)
        max_y = max(max_y, y)
        min_y = min(min_y, y)
    length_x = max_x - min_x
    length_y = max_y - min_y
    length = max(length_x, length_y)
    return length / distance
units = find_units(all_points)
print(f'Количество пикселей на 1 см: {units}')


def calculate_area(polygon):
    """Вычисляет площадь многоугольника"""
    n = len(polygon)
    area = 0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += (x1 + x2) * (y2 - y1)
    return abs(area) / (2*units**2)

area = calculate_area(all_points)


def find_water_level(vertices, target_area, epsilon=1e-3):
    """Находит уровень воды, при котором площадь подводной части равна заданной"""
    ymin = min(p[1] for p in vertices)
    ymax = max(p[1] for p in vertices)

    while ymax - ymin > epsilon:
        h = (ymin + ymax) / 2
        polygon = submerged_polygon(vertices, h)
        area = calculate_area(polygon)
        if area < target_area:
            ymin = h
        else:
            ymax = h
    return (ymin + ymax) / 2


def find_intersection(p1, p2, h):
    """Находит точку пересечения отрезка с линией y = h"""
    x1, y1 = p1
    x2, y2 = p2
    if (y1 - h) * (y2 - h) <= 0 and y1 != y2:
        x = x1 + (h - y1) * (x2 - x1) / (y2 - y1)
        return (x, h)
    else:
        return None


def submerged_polygon(vertices, h):
    """Находит подводную часть фигуры при уровне воды h."""
    submerged = []
    n = len(vertices)
    for i in range(n):
        p1, p2 = vertices[i], vertices[(i + 1) % n]

        # Если первая точка под водой или на уровне воды, добавляем ее
        if p1[1] <= h:
            submerged.append(p1)

        # Находим точку пересечения с уровнем воды, если она есть
        intersection = find_intersection(p1, p2, h)
        if intersection:
            submerged.append(intersection)

    return submerged

def calculate_centroid(polygon, A):
    """Вычисляет координаты центра масс (плавучести) фигуры."""
    n = len(polygon)
    if n <= 3:
        raise ValueError("Фигура должна содержать как минимум три вершины.")

    Bx = 0  # Центроид по X
    By = 0  # Центроид по Y

    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        cross_product = (x0 * y1 - x1 * y0)
        Bx += (x0 + x1) * cross_product
        By += (y0 + y1) * cross_product

    Bx /= (6 * A * units ** 2)
    By /= (6 * A * units ** 2)

    return Bx, By

def potencial_energy(Gy, By):
    U = m * 9.82 * 100 *  (Gy - By)
    return U

""" ОБРАБОТКА ДАННЫХ И ВЫВОД"""

S = m / (ro * d)
h = find_water_level(all_points, S) / units
h_global = h * units
polygon = submerged_polygon(all_points, h_global)
print(f'Полная площадь фигуры: {area} см^2')
B = calculate_centroid(polygon, S)
G = calculate_centroid(all_points, area)

Gx, Gy = G
# Преобразуем Gx и Gy в целочисленные значения
Gy_ = float(Gy)/units
Gx = int(Gx)
Gy = int(image.shape[0] - Gy)  # Перевод Y в глобальную систему
Bx, By = B
By_ = float(By)/units
Bx = int(Bx)
By = int(image.shape[0] - By)

U = potencial_energy(Gy_, By_)
print(Gy)
if area <= S:
    print('Тело утонуло:(')
else:
    print(f'Площадь погруженной части(по формуле Архимеда): {S} см^2')
    print(f'Площадь погруженной части(численным методом): {calculate_area(polygon)}')
    print(f"Уровень воды: {h} см")
    print("Координаты точек в подводной части:", polygon)
    print('Координаты центра масс:', G)
    print('Координаты центра плавучести:', B)
    print(f'Потенциальная энергия: {U} эргов')



# Создаем копию изображения для визуализации
image_with_submerged = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Рисуем линию уровня воды
cv2.line(image_with_submerged,
         (0, int(image.shape[0] - h_global)),
         (image.shape[1], int(image.shape[0] - h_global)),
         (255, 0, 0), 2)  # Синяя линия

# Рисуем центр масс (G)
cv2.circle(image_with_submerged, (Gx, Gy), 5, (0, 255, 255), -1)  # Жёлтый круг
cv2.putText(image_with_submerged, "G", (Gx + 10, Gy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)  # Метка "G"

# Рисуем центр плавучести (B) аналогично
cv2.circle(image_with_submerged, (Bx, By), 5, (0, 0, 255), -1)  # Красный круг для B
cv2.putText(image_with_submerged, "B", (Bx + 10, By - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Метка "B"

# Отображение обновленного изображения
cv2.imshow("Submerged Part with Buoyancy Center", image_with_submerged)
cv2.waitKey(0)
cv2.destroyAllWindows()



""" РАСЧЕТЫ ДЛЯ ПОТЕНЦИАЛЬНОЙ ЭНЕРГИИ """

# Сохраняем результаты для каждой ориентации
energies = []

for theta in range(0, 360):
    # Перевод угла в радианы
    rad = math.radians(theta)

    # Матрица поворота
    rotation_matrix = np.array([
        [math.cos(rad), -math.sin(rad)],
        [math.sin(rad), math.cos(rad)]
    ])

    # Преобразование координат
    rotated_points = [
        (
            rotation_matrix[0, 0] * x + rotation_matrix[0, 1] * y,
            rotation_matrix[1, 0] * x + rotation_matrix[1, 1] * y
        )
        for x, y in all_points
    ]
    # Вычисление площади новой фигуры
    rotated_area = calculate_area(rotated_points)
    # Уровень воды для новой фигуры
    h = find_water_level(rotated_points, S)
    # Центр масс и центр плавучести
    G = calculate_centroid(rotated_points, rotated_area)
    submerged_points = submerged_polygon(rotated_points, h)


    B = calculate_centroid(submerged_polygon(submerged_points, h), S)

    Gx, Gy = G
    Bx, By = B

    # Вычисление потенциальной энергии
    Gy_ = Gy / units
    By_ = By / units
    U = potencial_energy(Gy_, By_) * 10**(-2)

    # Сохраняем результаты
    energies.append((theta, U))


# График

angles, energy_values = zip(*energies)
plt.plot(angles, energy_values)
plt.xlabel(r"Угол $\theta$")
plt.ylabel(r"Потенциальная энергия (Дж $\cdot 10^{-5}$)")
plt.title(r"График зависимости потенциальной энергии от угла $\theta$")
plt.grid(True)
plt.show()
