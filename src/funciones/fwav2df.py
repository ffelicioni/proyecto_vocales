import numpy as np
import pandas as pd

def extract_formants(file_path):
    import numpy as np
    import pandas as pd
    import librosa
    from librosa import lpc
    from scipy import signal
    from scipy.signal import find_peaks
    #version en python de código original de Gonzalo Sad de la materia Prodivoz 
    x, Fs = librosa.load(file_path,sr=None)
    x=x*.95/np.max(np.abs(x))  #se normaliza sx a 95%
    N=4096*8                   #frecuencias en fft
    P=12                       #coeficientes lpc
    #window = np.hamming(len(x))
    #x1 = x * window
    #y=np.abs(np.fft.fft(x1))
    #f=np.arange(0,Fs/2-Fs/N,Fs/N)
    #omega=2*np.pi*f               #frecuencia angular
    A = lpc(x, P)
    B=[1]
    B=np.append(B,np.zeros(P)) #[1 y p ceros]
    s1 = signal.TransferFunction(B, A,dt=1/Fs)
    w,mag, phase = signal.dbode(s1,n=2000)
    peaks, nn = find_peaks(mag)
    fmaximos=w[peaks]/(2*np.pi)
    if (fmaximos[0]>200):
        formant_freq=np.array([fmaximos[0], fmaximos[1]])
    else:
        formant_freq=np.array([fmaximos[1], fmaximos[2]])
    return pd.DataFrame(formant_freq).transpose()
