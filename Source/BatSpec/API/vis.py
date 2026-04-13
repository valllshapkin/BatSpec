from BatSpec.Logic.Functions import SpecFunc

def showSpecFuncs(*spec: SpecFunc):
    from BatSpec.QtApp.Logic import application

    for i in spec:

        with application():
            from BatSpec.QtApp.中Analize.中SpecView.中Graph.Widget import MainGraph
            from BatSpec.QtApp.中Analize.中SpecView.中Graph.Logic import GRAPH_STATE

            GRAPH_STATE.setSpecFunc(i)

            # from BatSpec.QtApp.中Analize.Widget import RecordAnalisys
            # # from BatSpec.QtApp.中Project.Logic import PROJECT_STATE
            # # PROJECT_STATE.loadProject(ScriptDir / "SomeProject")
            window = MainGraph()
            window.show()

