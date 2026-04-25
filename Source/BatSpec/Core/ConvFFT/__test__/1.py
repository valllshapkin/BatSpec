
from pathlib import Path

from BatSpec.Core.Spectral.Echo import deconvolveEcho
from BatSpec.Core.Spectral.Smooth import negativeLaplacianSpec, curveEnhanceSpec
ScriptDir = Path(__file__).parent
from BatSpec.Core.ConvWindow import TEST_HANN_WINODW
from BatSpec.Core.Record import loadRecord, correctDC
from BatSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
from BatSpec.Core.Spectral import makeSpec, makeLogDB, interpolateByFactors, blurSpec, highPassSpec
from BatSpec.Core.Spectral.Statistic import noiseZNormByFreq
from BatSpec.Core.Spectral.Echo import deconvolveEcho
from BatSpec.Visualize import update_spec2d, run_visualizer

@run_visualizer
def main():
    record = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
    record = correctDC(record)
    record = applyСalibration(record, FlatResponseModel(sensitivity_pa=20))
    spec = makeSpec(record, TEST_HANN_WINODW, overlap=0.8, bins=300)
    zspec = noiseZNormByFreq(spec, noise_percentile=10)
    update_spec2d("zlow", makeLogDB(zspec, add_one=True))

    lp = negativeLaplacianSpec(makeLogDB(zspec, add_one=True), sigma=3)
    update_spec2d("lp", lp)

    curves = curveEnhanceSpec(lp, sigma_width=4)
    update_spec2d("curves", curves)


    # zblur = blurSpec(zspec, sigma=(1, 1))
    # zlow = interpolateByFactors(zspec, (0.4, 0.4))
    # update_spec2d("zlow", makeLogDB(zlow, add_one=True))
    # seq, patt = discover_best_pattern(zlow, 0.060, epochs=400, lr=0.01, device="cuda", top_k_peaks=10)

    # clear, echo = deconvolveEcho(makeLogDB(zlow, True), device='cuda', num_iterations=1000, anchor_pct=0, 
    #     lambda_sparsity=0.5, echo_duration_ms=300, 
    # )
    # update_spec2d("clear", clear)
    # update_spec2d("echo", echo)


    # clean_spectrogram = remove_pattern_from_spec(
    #     spec=makeLogDB(zlow, add_one=True),
    #     pattern=patt, 
    # )
    # update_spec2d("clean_spectrogram", clean_spectrogram)

