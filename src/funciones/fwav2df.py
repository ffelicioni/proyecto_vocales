def extract_formants(file_path):  
    import librosa
    from librosa import lpc
    from scipy import signal
    from scipy.signal import find_peaks
    import numpy as np
    import pandas as pd
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

# 'proyecto_vocales/data/raw/wav/*.wav'
def prepare_dataframe(folder_contents):
    import numpy as np
    import pandas as pd
    from pathlib import Path
    df=pd.DataFrame()
    for file_path in folder_contents:
        filename = Path(file_path).stem
        filename_parts = filename.split('_')
        metadata = {'file_path': file_path,
                  'vocal': filename_parts[0],
                  'id': filename_parts[1]}
        df1=extract_formants(file_path)
        df1['target']=filename_parts[0]
        df=pd.concat([df, df1], axis=0).reset_index(drop=True)

    df['target']=df['target'].str.upper()
    df.sort_values(by=['target'], ascending=True,inplace=True)
    return df