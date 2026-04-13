from scipy.signal.windows import blackmanharris

EXPORT_NAME = "BlackmanHarris STFT 3ms"
EXPORT_WINDOW = lambda n: blackmanharris(n, sym=False)    
EXPORT_DURATION = 0.003
