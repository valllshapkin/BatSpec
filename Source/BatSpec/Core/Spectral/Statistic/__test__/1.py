from pathlib import Path

ScriptDir = Path(__file__).parent

# --- Импорты BatSpec ---
from BatSpec.Core.ConvWindow import TEST_HANN_WINODW, TEST_BLHA_WINODW, Window, WindowNorm
from BatSpec.Core.Record import loadRecord, correctDC
from BatSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
from BatSpec.Core.Spectral import makeLogDB, DSPContext, makeSpec, makePiecewiseLog, interpolateByFactors
from BatSpec.Core.Spectral.Statistic import noiseZNormByFreq, localNoiseZNorm, blurSpec
from BatSpec.Visualize import update_spec2d, run_visualizer, update_function
from MultiArray.Core import ArrayContext, Framework, DeviceType
# --- Импорты MultiArray ---
from scipy.signal.windows import hann

@run_visualizer
def main():
    # record = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
    record = loadRecord(ScriptDir / "MYODAS_20230624_011300.wav")

    record = correctDC(record)
    record = applyСalibration(record, FlatResponseModel(sensitivity_pa=20))
    update_function("record", record)

    ctx_numpy_cpu = ArrayContext(Framework.NUMPY, DeviceType.CPU, None)
    
    print("--- Основные вычисления ---")
    with DSPContext(ctx_numpy_cpu):
        # spec_robust = makeSpec(
        #     record,
        #     window=(TEST_HANN_WINODW, TEST_BLHA_WINODW),
        #     overlap=0.8,
        #     bins=300,
        #     shifts=(-1, 1)  
        # ).to_context(ctx_numpy_cpu)
        spec_robust = makeSpec(
            record,
            window=Window(
                lambda n: hann(n, sym=False), 0.001, WindowNorm.ENERGY
            ),
            overlap=0.8,
            bins=300,
        )

    update_spec2d("Robust", makeLogDB(spec_robust, add_one=False))

    zspec = noiseZNormByFreq(spec_robust)
    # update_spec2d("zspec", zspec)
    update_spec2d("zspec_pl", makePiecewiseLog(zspec))

    lz = localNoiseZNorm(zspec, window_time=0.005, window_freq=10000)
    update_spec2d("lz", makePiecewiseLog(lz))

    lz1 = blurSpec(lz, sigma=1)
    update_spec2d("lz1", makePiecewiseLog(lz1))

    lz2 = blurSpec(lz, sigma=2)
    update_spec2d("lz2", makePiecewiseLog(lz2))

    lz4 = blurSpec(lz, sigma=4)
    update_spec2d("lz4", makePiecewiseLog(lz4))

    lz8 = blurSpec(lz, sigma=8)
    update_spec2d("lz8", makePiecewiseLog(lz8))

    lz16 = blurSpec(lz, sigma=16)
    update_spec2d("lz16", makePiecewiseLog(lz16))


    comb = makePiecewiseLog(lz) * makePiecewiseLog(lz1) * makePiecewiseLog(lz2) * makePiecewiseLog(lz4) * makePiecewiseLog(lz8) * makePiecewiseLog(lz16)
    update_spec2d("comb", makePiecewiseLog(comb))
    update_spec2d("comb", makeLogDB(comb, add_one=True))



    




