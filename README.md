Программа предназначена для анализа поведения 2d фигуры в жидкости с учетом ее ориентации и погружения.
Программа принимает следующие входные данные:

• Черно-белое изображение фигуры в плоскости погружения. 

• Плотность жидкости. 

• Максимальная длина фигуры по оси X или Y — служит для масштабирования изображения.

• Масса тела. 

• Толщина фигуры по оси Z — является постоянной и влияет на объем тела. 


Для работы с программой используйте "beaty_code", в качестве примера можете брать изображения из "example_of_images". Путь к изображению не должен содержать кириллицу. 
Для промежуточных проверок и лучшего понимания кода можно посмотреть "black_code".

BuoyancyApp (buoyancyapp.py) is a graphical Python application that simulates the stable equilibrium position of a 2D shape floating in a fluid. The user can upload an image with a shape or draw one manually, input physical properties, and visualize how the body floats according to Archimedes' principle and gravitational force.
