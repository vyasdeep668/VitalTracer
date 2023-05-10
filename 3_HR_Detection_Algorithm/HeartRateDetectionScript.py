__author__ = "Deep Vyas"
__name__ = "HeartRateDetection.py"
__version__ = "1.0.0"

# Project Imports
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import os


def measureHeartRate(ppg_data):
    """
    measureHeartRate: This Function will measure HeartRate from PPG Raw Data
    :param in -  ppg_data: List/Dataframe Column of Raw PPG Values
    :param out - heartRate: Heart Rate in bpm
    """ 

    # Step1: Filter the PPG signal with a bandpass filter
    # Low-pass filter the raw signal to remove noise
    fs = 100  # Sampling frequency
    fc =  1.5 # Cutoff frequency
    b, a = butter(2, 2 * fc / fs, 'low')

    ppg_data_raw = ppg_data
    #Applying LP Filter to ppg_data[channel]
    ppg_data = filtfilt(b, a, ppg_data)

    ppg_data_filtered = ppg_data
 
    # Step2: Check signal quality by calculating Signal to Noise Ratio.
    # Noise signal
    THRESHOLD = 25      #dB
    noise_data = ppg_data_raw - ppg_data_filtered

    signal_power = np.mean(np.abs(ppg_data_filtered)**2)
    noise_power = np.mean(np.abs(noise_data)**2)
    SNR = 10*np.log10(signal_power/noise_power)
    # print("SNR: ", SNR)
    if SNR < THRESHOLD:
        print("Signal Quality is Bad!")
        return 0

    # Step3: Differentiate the filtered signal
    # Differentiate the filtered signal to emphasize the high-frequency components
    ppg_data = np.gradient(ppg_data)

    # Step4: Square the differentiated  signal.
    # Square the differentiated signal to enhance QRS complex
    ppg_data = np.square(ppg_data)

    # Step5:  Integrate the squared signal with a sliding window.
    # Apply a moving average integration to smooth the signal
    window_size = int(0.5 * fs)  # 50 window size
    window = np.ones(window_size) / float(window_size)
    ppg_data = np.convolve(ppg_data, window, "same")

    # Step6: Find the R-peaks in the integrated signal.
    # Find peaks in the integrated signal
    ppg_peaks, _ = find_peaks(ppg_data, distance=0.2*fs)

    # Step7: Calculate the heart rate using the time difference between R-peaks.
    # Compute the inter-beat interval (IBI) and heart rate (HR) from the peak locations
    ibi = np.diff(ppg_peaks) / fs  # IBI in seconds
    hr = 60 / ibi  # HR in bpm
    return round(np.mean(hr), 2)



# Select channel from the dataset.
channel = 'pleth_5'
directory = '3_HR_Detection_Algorithm/FilesToBeMeasured/'
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        with open(os.path.join(directory, filename)) as f:
            df = pd.read_csv(f)
            # Data Sampled at 500Hz
            # Getting 20 Seconds of raw data
            ppg_data = df[['time', channel]].iloc[:10000]
            print(f.name + ' - HearRate ' + str(measureHeartRate(ppg_data[channel])) + ' bpm')








