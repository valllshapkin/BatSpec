import sys
import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtWidgets import QApplication, QMainWindow
from BatSpec.QtUp.PolyROI import ROIData
import os

# Пути импорта должны соответствовать вашей структуре
from BatSpec.QtApp.中Analize.中SpecView.中Corp.Logic import CORP_STATE
from BatSpec.QtApp.中Analize.中SpecView.中Corp.Widget import CorpProcessMainWidget
from pathlib import Path
ScriptDir = Path(__file__).parent
os.chdir(ScriptDir)

class MockSpecFunc:
    def __init__(self, matrix):
        self.matrix = matrix
        self.time = np.arange(matrix.shape[1])
        self.freq = np.arange(matrix.shape[0])

def generate_test_image(path):
    """Генерирует изображение с синусоидой для теста"""
    w, h = 500, 300
    img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(img)
    
    # Рисуем яркую кривую
    points = []
    for x in range(w):
        y = h//2 + int(50 * np.sin(x * 0.05))
        points.append((x, y))
    
    draw.line(points, fill=255, width=3)
    img.save(path)

def load_image_to_specfunc(path):
    img = Image.open(path).convert('L')
    matrix = np.array(img).astype(np.float32)
    # Переворачиваем по Y, так как в картинках 0 сверху, а в спектрограммах обычно снизу
    return MockSpecFunc(np.flipud(matrix))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 1. Готовим данные
    img_path = "test_pattern.png"
    generate_test_image(img_path)
    spec = load_image_to_specfunc(img_path)
    
    # Создаем ROI где-то в середине синусоиды
    test_roi = ROIData(idx=0, x=100, y=100, w=150, h=100)
    
    # 2. Инициализируем состояние
    CORP_STATE.setData(spec, test_roi)
    
    # 3. Запускаем UI
    window = QMainWindow()
    window.setWindowTitle("ROI Corp & Process Laboratory")
    main_widget = CorpProcessMainWidget()
    window.setCentralWidget(main_widget)
    window.resize(1200, 800)
    window.show()
    
    sys.exit(app.exec())