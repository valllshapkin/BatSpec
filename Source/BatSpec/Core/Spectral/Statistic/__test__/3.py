from pathlib import Path

ScriptDir = Path(__file__).parent

# --- Импорты BatSpec ---
from BatSpec.Core.ConvWindow import TEST_HANN_WINODW, TEST_BLHA_WINODW, Window, WindowNorm
from BatSpec.Core.Record import loadRecord, correctDC
from BatSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
from BatSpec.Core.Spectral import interpolateByFactors, makeLogDB, DSPContext, makeSpec, makePiecewiseLog, makeRobustSpec
from BatSpec.Core.Spectral.Statistic import deadZoneFilter, medianDenoise, multiScaleGeometricMean, multiScaleLogProduct, noiseZNormByFreq, localNoiseZNorm, blurSpec
from BatSpec.Core.Spectral.Smooth import curveEnhanceSpec
from BatSpec.Visualize import update_spec2d, run_visualizer, update_function
from MultiArray.Core import ArrayContext, Framework, DeviceType
# --- Импорты MultiArray ---
from scipy.signal.windows import hann

@run_visualizer
def main():
    record = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")

    record = correctDC(record)
    record = applyСalibration(record, FlatResponseModel(sensitivity_pa=20))
    update_function("record", record)

    ctx_numpy_cpu = ArrayContext(Framework.NUMPY, DeviceType.CPU, None)
    
    print("--- Основные вычисления ---")
    with DSPContext(ctx_numpy_cpu):
        spec = makeSpec(
            record,
            window=Window(
                lambda n: hann(n, sym=False), 0.001, WindowNorm.ENERGY
            ),
            overlap=0.8,
            bins=300,
        )
        spec_robust = makeRobustSpec(
            record, 
            window=Window(
                lambda n: hann(n, sym=False), 0.001, WindowNorm.ENERGY
            ),
            overlap=0.8,
            bins=300,
            shifts=(-1, 1)
        )


    update_spec2d("Spec", makeLogDB(spec, add_one=False))
    update_spec2d("spec_robust", makeLogDB(spec_robust, add_one=False))

    zspec = noiseZNormByFreq(spec)
    update_spec2d("zspec", makePiecewiseLog(zspec))

    lz = localNoiseZNorm(zspec, window_time=0.005, window_freq=10000)
    update_spec2d("lz", makePiecewiseLog(lz))

    lz = medianDenoise(lz)
    update_spec2d("clz", lz)

    comb_zs = multiScaleLogProduct(zspec, sigmas=(1.0, 2.0, 4.0, 8.0, 16.0))
    update_spec2d("comb_zs", comb_zs)

    comb_lz = multiScaleLogProduct(lz, sigmas=(1.0, 2.0, 4.0, 8.0, 16.0))
    update_spec2d("comb_lz", comb_lz)
# 
    update_function("zs", comb_zs.integrateOverFreq())
    update_function("lz", comb_lz.integrateOverFreq())


    



   



    



    




