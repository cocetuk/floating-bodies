import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

""" ИЗОБРАЖЕНИЕ """
# Загрузка изображения
image_0 = cv2.imread('star.png', cv2.IMREAD_GRAYSCALE)
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
d = 4
print('Введите плотность жидкости в г/см^3:')
ro = 1.26

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


def find_extrema(energies):
    """ Ищет положения устойчивого и неустойчивого равновесия"""
    angles = [energy[0] for energy in energies]
    potential_energies = [energy[1] for energy in energies]
    extrema = []
    n = len(energies)

    for i in range(n):
        prev_i = (i - 1) % n
        next_i = (i + 1) % n

        # Проверяем на экстремум
        if potential_energies[i] > potential_energies[prev_i] and potential_energies[i] > potential_energies[next_i]:
            # Это максимум
            stability = "неустойчивое"
            extrema.append((angles[i], potential_energies[i], stability))
        elif potential_energies[i] < potential_energies[prev_i] and potential_energies[i] < potential_energies[next_i]:
            # Это минимум
            stability = "устойчивое"
            extrema.append((angles[i], potential_energies[i], stability))

    return extrema

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

extrema = find_extrema(energies)
print("Экстремумы потенциальной энергии:")
for angle, energy, stability in extrema:
    print(f"Угол: {angle}° - Потенциальная энергия: {energy} Дж * 10^(-5) - Положение равновесия: {stability}")

# График
angles, energy_values = zip(*energies)
plt.plot(angles, energy_values)
plt.xlabel(r"Угол $\theta$")
plt.ylabel(r"Потенциальная энергия (Дж $\cdot 10^{-5}$)")
plt.title(r"График зависимости потенциальной энергии от угла $\theta$")
plt.grid(True)
plt.show()


# Допустим, у вас уже есть список экстремумов:
# extrema = [(theta1, U1, stability1), (theta2, U2, stability2), ... ]
print('Введите угoл, который хотите посмотреть:')
a = int(input())
chosen_angle = a

# Перевод угла в радианы
rad = math.radians(chosen_angle)

# Поворот исходного набора точек all_points
#    (предполагается, что all_points уже существует и содержит координаты в виде [(x,y), ...])
rotated_points = []
for (x, y) in all_points:
    x_new = x * math.cos(rad) - y * math.sin(rad)
    y_new = x * math.sin(rad) + y * math.cos(rad)
    rotated_points.append((x_new, y_new))

# Находим bounding box для повёрнутого многоугольника, чтобы отрисовать его без выхода за границы
xs = [p[0] for p in rotated_points]
ys = [p[1] for p in rotated_points]

min_x, max_x = int(min(xs)), int(max(xs))
min_y, max_y = int(min(ys)), int(max(ys))

width  = max_x - min_x + 100  # небольшой отступ по ширине
height = max_y - min_y + 100  # небольшой отступ по высоте

# Создаём пустое белое изображение нужного размера
rotated_image = np.ones((height, width, 3), dtype=np.uint8) * 255

# Сдвигаем точки так, чтобы вся фигура попала в "кадр"
shifted_points = []
dx, dy = -min_x + 50, -min_y + 50  # 50 пикселей отступа
for (x, y) in rotated_points:
    shifted_points.append((int(x + dx), int(y + dy)))

# Рисуем многоугольник
cv2.polylines(
    rotated_image,
    [np.array(shifted_points, dtype=np.int32)],
    isClosed=True,
    color=(0, 0, 0),
    thickness=2
)

# Определяем площадь повёрнутого многоугольника
rotated_area = calculate_area(rotated_points)

# Рассчитываем уровень воды (h) в локальных координатах, исходя из площади погруженной части
#    S = m / (ro * d) уже определена в вашем коде
h_local = find_water_level(rotated_points, S)  # h_local – уровень воды в координатах "повёрнутого" контура

# Определяем подводную часть
submerged_poly_local = submerged_polygon(rotated_points, h_local)

# Для отрисовки подводной части сдвигаем её точки аналогично основному контуру
shifted_submerged_poly = []
for (x, y) in submerged_poly_local:
    shifted_submerged_poly.append((int(x + dx), int(y + dy)))

# Рисуем подводную часть (можно другим цветом, например, синим)
if len(shifted_submerged_poly) > 2:
    cv2.polylines(
        rotated_image,
        [np.array(shifted_submerged_poly, dtype=np.int32)],
        isClosed=True,
        color=(255, 0, 0),
        thickness=2
    )

# Вычисляем центры масс и плавучести
Gx_local, Gy_local = calculate_centroid(rotated_points, rotated_area)
Bx_local, By_local = calculate_centroid(submerged_poly_local, S)

# Сдвигаем координаты центров так же, как и точки
Gx_shifted, Gy_shifted = int(Gx_local + dx), int(Gy_local + dy)
Bx_shifted, By_shifted = int(Bx_local + dx), int(By_local + dy)

# Переворот изображения по вертикали
rotated_image = cv2.flip(rotated_image, 0)

# Получаем высоту изображения
image_height, image_width = rotated_image.shape[:2]

# Пересчитываем координаты для точек и текста
Gx_new, Gy_new = Gx_shifted, image_height - Gy_shifted
Bx_new, By_new = Bx_shifted, image_height - By_shifted

# Рисуем точки
cv2.circle(rotated_image, (Gx_new, Gy_new), 5, (0, 0, 0), -1)  # чёрный
cv2.putText(
    rotated_image,
    "G",
    (Gx_new + 10, Gy_new - 10),  # Сдвигаем текст относительно новой точки
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0, 0, 0),  # чёрный
    1
)

cv2.circle(rotated_image, (Bx_new, By_new), 5, (0, 0, 255), -1)  # красный
cv2.putText(
    rotated_image,
    "B",
    (Bx_new + 10, By_new - 10),  # Сдвигаем текст относительно новой точки
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0, 0, 255),
    1
)

# Сохранение изображения в файл
cv2.imwrite(f"rotated_figure_{chosen_angle}_flipped_vertical.png", rotated_image)

# Отображение изображения
cv2.imshow(f"Flipped Figure at {chosen_angle} deg", rotated_image)
cv2.waitKey(0)
