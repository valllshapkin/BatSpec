from BatSpec.Visualize import update_function, update_spec2d, run_visualizer
@run_visualizer
def main():
    from pathlib import Path
    ScriptDir = Path(__file__).parent
    from BatSpec.Core.Record import loadRecord, correctDC
    from BatSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
    from BatSpec.Core.SaveIntegral import SaveIntegral
    from BatSpec.Core.Spectral import makeSpec, makeLogDB
    from BatSpec.Core.ConvWindow import TEST_HANN_WINODW


    record_row = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
    record_row = correctDC(record_row)
    update_function("record_row", record_row)


    record = applyСalibration(record_row, FlatResponseModel(sensitivity_pa=20))
    update_function("record", record)

    print(f"""
    SaveIntegralEnergy(record_row): {SaveIntegral.Energy(record_row)}
    SaveIntegralEnergy(record): {SaveIntegral.Energy(record)}
    """)

    spec = makeSpec(record, TEST_HANN_WINODW, overlap=0.9, bins=300)
    update_spec2d("spec", spec)

    print(f"""
    SaveIntegralEnergy(spec): {SaveIntegral.Energy(spec)}
    SaveIntegralEnergy(record): {SaveIntegral.Energy(record)}
    """)

    SPSL = makeLogDB(spec)
    update_spec2d("SPSL", SPSL)

