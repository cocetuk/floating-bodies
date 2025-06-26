# Buoyancy Simulator

<img src="[https://github.com/user-attachments/assets/95b51e76-0073-47a1-b43e-1b22c10c6bec]" width="500" />

![Status: In Development](https://img.shields.io/badge/status-в%20разработке-green) ![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)

Данный проект, название которого "Исследование поведения плавающих тел в жидкости", был выполнен в рамках проектной деятельности по программированию. Результатом проекта является наглядный симулятор плавучести ([buoyancy_simulator.py]([https://docs.python.org/3/library/tkinter.html](https://github.com/cocetuk/floating-bodies/blob/main/buoyancy_simulator.py))) плоских фигур в жидкости с расчётом положения равновесия и потенциальной энергии при любом угле поворота.

## Содержание

- [Технологии](#технологии)  
- [Начало работы](#начало-работы)  
- [Разработка](#разработка)  
- [Тестирование](#тестирование)  
- [Deploy и CI/CD](#deploy-и-cicd)  
- [Contributing](#contributing)  
- [To do](#to-do)  
- [Команда проекта](#команда-проекта)  
- [Источники](#источники)  

## Технологии

- **Python 3.7+**  
- [Tkinter](https://docs.python.org/3/library/tkinter.html) — графический интерфейс  
- [OpenCV](https://opencv.org/) — импорт и обработка изображений  
- [NumPy](https://numpy.org/) — численные расчёты и матрицы  
- [Matplotlib](https://matplotlib.org/) — отрисовка фигур и графиков  
- [pytest](https://docs.pytest.org/) — модульное тестирование  

## Начало работы

1. Склонируйте репозиторий:  
   ```sh
   git clone https://github.com/your-username/buoyancy-simulator.git
   cd buoyancy-simulator
   ```
2. Создайте и активируйте виртуальное окружение:  
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```
3. Установите зависимости:  
   ```sh
   pip install -r requirements.txt
   ```
4. Запустите приложение:  
   ```sh
   python main.py
   ```

## Разработка

### Требования

- Python 3.7 или новее  
- Git  
- На вашей системе должны быть установлены библиотеки из `requirements.txt`

### Установка зависимостей

```sh
pip install -r requirements.txt
```

### Запуск в режиме разработки

```sh
python main.py
```

### Сборка дистрибутива (опционально)

Для упаковки в исполняемый файл можно использовать [PyInstaller](https://www.pyinstaller.org/):
```sh
pip install pyinstaller
pyinstaller --onefile main.py
```

## Тестирование

Для запуска юнит-тестов (pytest):
```sh
pytest --maxfail=1 --disable-warnings -q
```

## Deploy и CI/CD

- Настроен GitHub Actions для автоматической прогонки тестов и отчёта покрытия кода  
- По пушу в `main` прогоняются CI-пайплайны из `.github/workflows/ci.yml`  

## Contributing

Будем рады вашим вкладкам:

1. Форкните репозиторий  
2. Создайте тему (issue) для обсуждения новой фичи или бага  
3. Сделайте ветку `feature/ваше-описание`  
4. Напишите код, добавьте тесты  
5. Откройте Pull Request с подробным описанием изменений  

Убедитесь, что всё покрыто тестами и соответствует стилю PEP8.

Подробнее — [CONTRIBUTING.md](./CONTRIBUTING.md).

## To do

- [x] Базовая загрузка и парсинг контура из изображения  
- [x] Рисование полигона и свободной формы в окне  
- [x] Расчёт положения равновесия и потенц. энергии  
- [x] Диалог выбора произвольного угла  
- [ ] Улучшить производительность поиска экстримумов  
- [ ] Добавить сохранение результатов в файл (CSV/JSON)  
- [ ] Локализация интерфейса на другие языки  

## Команда проекта

- Ваше Имя — дизайнер UX/UI  
- Ваше Имя — тестировщик  

## Источники

- Основная идея расчёта плавучести: Архимед, классические учебники по гидростатике  
- Обработка изображений — документация OpenCV  
- Визуализация — примеры Matplotlib из официального руководства  
- Тестирование — шаблоны pytest на GitHub  
