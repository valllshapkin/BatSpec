from BatSpec.QtApp.Logic import application
from BatSpec.Logic.Loader import loadRecord
from BatSpec.Logic.SoundWorker import localRMS
from BatSpec.Logic.ConvWindow import Window
from scipy.signal.windows import hann
import pyqtgraph as pg
from pathlib import Path
ScriptDir = Path(__file__).parent

with application():
    from BatSpec.QtApp.中Analize.中SpecView.中Graph.Widget import MinimapPlot
    func = localRMS(loadRecord(ScriptDir / "MYODAS_20230624_005134.wav"), Window(hann, 0.001, norm="area"))
    
    # Создаём окно с GraphicsLayoutWidget
    win = pg.GraphicsLayoutWidget(show=True, title="Minimap")
    win.resize(800, 100)
    
    # Добавляем MinimapPlot (PlotItem) в layout
    plot = MinimapPlot()
    win.addItem(plot)

    from BatSpec.QtApp.中Analize.中SpecView.中Graph.Logic import GRAPH_STATE
    GRAPH_STATE.setMinimapFunc(func)
    
    # plot.updateData(func)
    win.show()
