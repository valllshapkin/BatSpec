from .func1d_visualizer import update_function, Function1DVisualizer
from .spec2d_visualizer import update_spec2d, Spec2DVisualizer

def run_visualizer():
    from PySide6 import QtWidgets
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    window2 = Function1DVisualizer()
    window2.show()
    window1 = Spec2DVisualizer()
    window1.show()

    sys.exit(app.exec())
