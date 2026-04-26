from scipy.signal.windows import blackmanharris

EXPORT_NAME = "Blackman Harris STFT 2ms"
EXPORT_WINDOW = lambda n: blackmanharris(n, sym=False)    
EXPORT_DURATION = 0.002

