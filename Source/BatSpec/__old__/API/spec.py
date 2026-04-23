from BatSpec.Logic.Functions import SpecFunc
from BatSpec.Python.Graph import GraphCache, GraphNode
from BatSpec.QtApp.中Analize.中STFT.Logic import STFT_STATE
from BatSpec.QtApp.中Analize.中Record.Logic import RECORD_STATE
from BatSpec.QtApp.中Analize.中SpecView.Logic import SPEC_VIEW_STATE
from BatSpec.QtApp.中Analize.中SpecView.中Index.Logic import INDEX_STATE
from BatSpec.QtUp.PolyROI import ROIData

GRAPH_CACHE = GraphCache()

def getSpectrogramm() -> GraphNode[SpecFunc]:

    def core():
        if not STFT_STATE.output_stft: raise RuntimeError()
        return STFT_STATE.output_stft

    return GRAPH_CACHE.get_or_create_node(
        f"{RECORD_STATE.path}:{STFT_STATE.input_cache}",
        core 
    )

def updateVisLayer(name: str, func: GraphNode[SpecFunc] | SpecFunc):
    if name in ("raw", "denoised"):
        raise RuntimeError("Их нельзя менять")
    if isinstance(func, GraphNode):
        SPEC_VIEW_STATE.addSpec(name, func.value)
    elif isinstance(func, SpecFunc):
        SPEC_VIEW_STATE.addSpec(name, func)
    else:
        raise RuntimeError()

def getVisLayer(name: str):
    return SPEC_VIEW_STATE.specs

def updateGroup(rois: list[ROIData]):
    INDEX_STATE.updateGroup(1, rois)
