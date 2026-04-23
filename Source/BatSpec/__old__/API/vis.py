from BatSpec.Logic.Functions import SpecFunc
from BatSpec.Logic.TransitionWorker import integrateFreq

def showSettingsPanel():
    from BatSpec.QtApp.Logic import application

    with application():
        from BatSpec.QtApp.Shared.中ThemeSettings.Widget import ThemeSettingsPanel
        
        # from BatSpec.QtApp.中Analize.Widget import RecordAnalisys
        # # from BatSpec.QtApp.中Project.Logic import PROJECT_STATE
        # # PROJECT_STATE.loadProject(ScriptDir / "SomeProject")
        window = ThemeSettingsPanel()
        window.show()


def showSpecFuncs(*spec: SpecFunc):
    from BatSpec.QtApp.Logic import application


    with application():
        from BatSpec.QtApp.中Analize.中SpecView.中Graph.Widget import MainGraph
        from BatSpec.QtApp.中Analize.中SpecView.中Graph.Logic import GRAPH_STATE

        GRAPH_STATE.setSpecFunc(spec[0])
        GRAPH_STATE.setMinimapFunc(integrateFreq(spec[0]))
        print(spec[0].time)
        print(integrateFreq(spec[0]).time_axis)
        # from BatSpec.QtApp.中Analize.Widget import RecordAnalisys
        # # from BatSpec.QtApp.中Project.Logic import PROJECT_STATE
        # # PROJECT_STATE.loadProject(ScriptDir / "SomeProject")
        window = MainGraph()
        window.show()

