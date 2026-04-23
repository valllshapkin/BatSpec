import numpy as np
from scipy.signal.windows import hann

from BatSpec.QtApp.Logic import application
from BatSpec.Logic.Loader import loadRecord
from BatSpec.Logic.ConvWindow import Window
from BatSpec.Logic.TransitionWorker import makeSpec, etalonNoise, integrateFreq
from BatSpec.Logic.SoundWorker import correctDC
from BatSpec.Logic.SpecWorker import subEtalonNoise

from PySide6.QtWidgets import QMainWindow, QToolBar, QLabel

from pathlib import Path
ScriptDir = Path(__file__).parent

with application():
    from BatSpec.QtApp.中Analize.中SpecView.中Graph.Widget import (
        MainGraph, DebugValue, LockYCheckBox, ColorMapSelector
    )
    from BatSpec.QtApp.中Analize.中SpecView.中Graph.Logic import GRAPH_STATE

    # --- Загрузка и подготовка данных ---
    spec = makeSpec(
        correctDC(loadRecord(ScriptDir / "MYODAS_20230624_005134.wav")),
        Window(hann, 0.001, norm="energy"),
        overlap=0.5,
        bins=200
    )
    noise = etalonNoise(spec, percentile=5)
    
    # Переводим в децибелы
    spec_db = spec.cloneApply(lambda arr: 20 * np.log10(np.clip(arr, 1e-9, None)))

    # --- Создание основных виджетов ---
    main_graph = MainGraph()
    GRAPH_STATE.setMinimapFunc(integrateFreq(subEtalonNoise(spec, noise)))
    GRAPH_STATE.setSpecFunc(spec_db)
    
    # --- Инициализация дополнительных элементов UI ---
    debug_label = DebugValue(main_graph.view_box, unit="dB")
    lock_checkbox = LockYCheckBox()
    cmap_selector = ColorMapSelector()

    # --- Настройка главного окна ---
    window = QMainWindow()
    window.setCentralWidget(main_graph)
    
    # 1. Создаем панель инструментов (ToolBar) сверху
    toolbar = QToolBar("Graph Controls")
    toolbar.addWidget(QLabel("  Colormap: "))
    toolbar.addWidget(cmap_selector)
    toolbar.addSeparator()
    toolbar.addWidget(lock_checkbox)
    window.addToolBar(toolbar)

    # 2. Дебаг-значения оставляем в строке состояния снизу
    window.statusBar().addWidget(debug_label)   
    
    window.setWindowTitle("Spectrogram Viewer")
    window.resize(1200, 800)
    
    # Устанавливаем начальные значения (UI обновится автоматически благодаря TriggerGraph)
    # GRAPH_STATE.setLockY(False)
    # GRAPH_STATE.setColorMap("viridis") # Можно задать любую дефолтную
    
    window.show()