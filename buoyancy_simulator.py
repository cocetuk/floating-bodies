import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tkinter.font as tkfont
import cv2
import numpy as np
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def calculate_signed_area(polygon):
    n = len(polygon)
    area = 0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2

def calculate_area(polygon):
    return abs(calculate_signed_area(polygon))

def find_units(polygon, distance_cm):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    px_length = max(max(xs) - min(xs), max(ys) - min(ys))
    return px_length / distance_cm

def find_intersection(p1, p2, h):
    x1, y1 = p1
    x2, y2 = p2
    if (y1 - h) * (y2 - h) <= 0 and y1 != y2:
        x = x1 + (h - y1) * (x2 - x1) / (y2 - y1)
        return (x, h)
    return None

def submerged_polygon(vertices, h):
    sub = []
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        if p1[1] <= h:
            sub.append(p1)
        inter = find_intersection(p1, p2, h)
        if inter:
            sub.append(inter)
    return sub

def find_water_level(vertices, target_area, eps=1e-3):
    ymin = min(y for _, y in vertices)
    ymax = max(y for _, y in vertices)
    while ymax - ymin > eps:
        h = (ymin + ymax) / 2
        if calculate_area(submerged_polygon(vertices, h)) < target_area:
            ymin = h
        else:
            ymax = h
    return (ymin + ymax) / 2

def calculate_centroid(polygon):
    if not polygon:
        return 0, 0
    A = calculate_signed_area(polygon)
    cx = cy = 0
    for i in range(len(polygon)):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % len(polygon)]
        cross = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    factor = 1 / (6 * A) if A != 0 else 0
    return cx * factor, cy * factor

def potential_energy(polygon, units, m, g, density, thickness):
    S_phys = m / (density * thickness)
    h = find_water_level(polygon, S_phys * units**2)
    submerged = submerged_polygon(polygon, h)
    Gx, Gy = calculate_centroid(polygon)
    Bx, By = calculate_centroid(submerged) if submerged else (Gx, Gy)
    return m * g * ((Gy - By) / units)

def find_equilibrium_angle(polygon, units, m, density, thickness, g=9.82):
    cx0, cy0 = calculate_centroid(polygon)
    base = [(p[0] - cx0, p[1] - cy0) for p in polygon]
    energies = []
    for theta in range(0, 360, 5):
        rad = math.radians(theta)
        R = np.array([[math.cos(rad), -math.sin(rad)],
                      [math.sin(rad),  math.cos(rad)]])
        rotated = [(cx0 + R.dot([dx, dy])[0], cy0 + R.dot([dx, dy])[1]) for dx, dy in base]
        U = potential_energy(rotated, units, m, g, density, thickness)
        energies.append((theta, U))
    return min(energies, key=lambda x: x[1])[0]

class BuoyancyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Buoyancy Simulator")
        self.configure(bg="#2E2E2E")
        self.geometry("520x420")
        self.resizable(False, False)
        
        header = ttk.Label(self, text="Buoyancy Simulator", font=("Nunito", 25), background="#2E2E2E", foreground="#ECECEC")
        header.grid(row=0, column=0, columnspan=2, pady=(20,10))

        default_font = tkfont.Font(family="Nunito", size=10)
        self.option_add("*Font", default_font)

        self.contour = None
        self.distance = tk.DoubleVar(value=10.0)
        self.mass = tk.DoubleVar(value=25.0)
        self.thickness = tk.DoubleVar(value=5.0)
        self.density = tk.DoubleVar(value=1.0)
        self.image_label = tk.StringVar(value="Файл не выбран или фигура не нарисована")

        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TLabel', background='#2E2E2E', foreground='#ECECEC', font=('Nunito', 11))
        style.configure('TButton', background='#444444', foreground='#ECECEC', padding=8, font=('Nunito', 11))
        style.map('TButton', background=[('active', '#555555')])
        style.configure('TEntry', fieldbackground='#3C3C3C', foreground='#ECECEC', font=('Nunito', 11))

        ttk.Button(self, text="Выбрать изображение", command=self.on_choose).grid(row=1, column=0, pady=10, padx=5)
        ttk.Button(self, text="Нарисовать фигуру", command=self.draw_shape).grid(row=1, column=1, pady=10, padx=5)
        ttk.Label(self, textvariable=self.image_label).grid(row=2, column=0, columnspan=2, pady=6)

        self._entry_row(3, "Макс. размер (см)", self.distance)
        self._entry_row(4, "Масса (г)", self.mass)
        self._entry_row(5, "Толщина (см)", self.thickness)
        self._entry_row(6, "Плотность (г/см³)", self.density)

        ttk.Button(self, text="Показать равновесие", command=self.show_equilibrium).grid(row=7, column=0, columnspan=2, pady=12)

    def _entry_row(self, row, label, var):
        ttk.Label(self, text=label).grid(row=row, column=0, sticky='W', padx=(10,5), pady=4)
        ttk.Entry(self, textvariable=var, width=20).grid(row=row, column=1, sticky='E', pady=4)

    def on_choose(self):
        path = filedialog.askopenfilename(title="Выберите изображение", filetypes=[("PNG files","*.png"), ("JPEG files","*.jpg"), ("All files","*.*")])
        if not path:
            return
        self.image_label.set(path.split('/')[-1])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            messagebox.showerror("Ошибка", "Контур не найден")
            return
        cnt = contours[0].reshape(-1, 2)
        height = img.shape[0]
        self.contour = [(x, height - y) for x, y in cnt]

    def draw_shape(self):
        win = tk.Toplevel(self)
        win.title("Нарисуйте фигуру")
        canvas_size = 400
        canvas = tk.Canvas(win, width=canvas_size, height=canvas_size, bg='white')
        canvas.pack()

        points = []

        def orient(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

        def intersects(a1, a2, b1, b2):
            return orient(a1, a2, b1) * orient(a1, a2, b2) < 0 and orient(b1, b2, a1) * orient(b1, b2, a2) < 0

        def to_math(x, y):
            return x, canvas_size - y

        def to_screen(x, y):
            return x, canvas_size - y

        def on_click(event):
            mx, my = to_math(event.x, event.y)
            if points:
                px, py = points[-1]
                for i in range(len(points) - 2):
                    q1, q2 = points[i], points[i+1]
                    if intersects((px, py), (mx, my), q1, q2):
                        messagebox.showwarning("Ошибка", "Грани фигуры пересекаются!")
                        return
                sx, sy = to_screen(px, py)
                canvas.create_line(sx, sy, event.x, event.y, fill='black', width=2)
            points.append((mx, my))

        def on_finish():
            if len(points) < 3:
                messagebox.showerror("Ошибка", "Нужно минимум 3 точки")
                return
            x0, y0 = points[0]
            xn, yn = points[-1]
            if math.hypot(x0 - xn, y0 - yn) > 10:
                messagebox.showwarning("Ошибка", "Фигура не замкнута")
                return
            # Убираем последнюю точку если она дублирует первую
            if len(points) > 3 and math.hypot(x0 - xn, y0 - yn) < 10:
                points.pop()
            self.contour = points.copy()
            self.image_label.set("Нарисованная фигура")
            win.destroy()

        canvas.bind('<Button-1>', on_click)
        ttk.Button(win, text="Готово", command=on_finish).pack(pady=5)

    def show_equilibrium(self):
        if not self.contour:
            messagebox.showwarning("Внимание", "Сначала выберите или нарисуйте фигуру")
            return
        units = find_units(self.contour, self.distance.get())
        area_physical = calculate_area(self.contour) / (units**2)
        S_phys = self.mass.get() / (self.density.get() * self.thickness.get())
        if area_physical <= S_phys:
            messagebox.showinfo("Результат", "Ваша фигурка утонула :(")
            return
        theta = find_equilibrium_angle(self.contour, units, self.mass.get(), self.density.get(), self.thickness.get())
        cx0, cy0 = calculate_centroid(self.contour)
        pts_centered = [(x-cx0, y-cy0) for x, y in self.contour]
        rad = math.radians(theta)
        R = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
        ctr = [(cx0 + R.dot([dx, dy])[0], cy0 + R.dot([dx, dy])[1]) for dx, dy in pts_centered]
        h = find_water_level(ctr, S_phys*units**2)
        sub = submerged_polygon(ctr, h)
        Gx, Gy = calculate_centroid(ctr)
        Bx, By = calculate_centroid(sub) if sub else (Gx, Gy)

        win = tk.Toplevel(self)
        win.title(f"Устойчивое равновесие: {theta}°")
        fig, ax = plt.subplots(facecolor='#1E1E1E')
        plt.style.use('dark_background')
        ax.set_facecolor('#1E1E1E')
        ax.add_patch(Polygon(ctr, closed=True, edgecolor='#00DDFF', fill=False))
        ax.add_patch(Polygon(sub, closed=True, edgecolor='#00DDFF', facecolor='#003344', alpha=0.5))
        ax.axhline(y=h, color='#00DDFF', linestyle='--')
        ax.plot(Gx, Gy, 'wo'); ax.text(Gx+units*0.02, Gy, 'G', color='white')
        ax.plot(Bx, By, 'ro'); ax.text(Bx+units*0.02, By, 'B', color='red')
        ax.axis('equal'); ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

if __name__ == '__main__':
    app = BuoyancyApp()
    app.mainloop()
