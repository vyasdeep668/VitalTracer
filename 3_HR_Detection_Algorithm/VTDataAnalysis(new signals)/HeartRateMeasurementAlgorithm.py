__author__ = "Deep Vyas"
__name__ = "HeartRateMeasurementAlgorithm.py"
__version__ = "1.0.0"

# Project Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy import signal, ndimage
import os


#Project Defines
CSV_VALID_FOLDER_NAME = 'ValidCSVFiles'


def merge_ranges(ranges):
    """
    Merge overlapping ranges in a list of tuples.
    Args:ranges:A list of tuples representing ranges, where each tuple contains
                two integers representing the start and end points of the range.
    Returns:    A list of tuples representing the merged ranges, where each tuple contains
                two integers representing the start and end points of the range.
    Example:    Input ranges:  [(0, 5), (601, 783), (691, 789), (587, 744)]
                Merged ranges:  [(0, 5), (587, 789)]
    """
    # Check if the ranges list is empty
    if not ranges:
        return ranges
    
    # Sort the ranges in ascending order based on their start points
    ranges.sort()
    # Initialize the merged ranges list with the first range in the list
    merged = [ranges[0]]
    # Loop through the remaining ranges in the list
    for r in ranges[1:]:
        if r[0] <= merged[-1][1]:
            # If the current range overlaps with the last merged range, merge them
            merged[-1] = (merged[-1][0], max(merged[-1][1], r[1]))
        else:
            # If the current range doesn't overlap with the last merged range, add it to the merged list
            merged.append(r)
    # Return the merged ranges list
    return merged


def detect_high_fluctuation_ranges(data, motion_detection_limit):
    """
    Detects high fluctuation ranges in the given data based on the motion detection limit.

    Args:
        data (ndarray): Input data array.
        motion_detection_limit (float): Threshold for detecting high fluctuations.

    Returns:
        list: List of tuples representing the start and end indices of the high fluctuation ranges.
    """
    # Calculate the adjacent magnitude differences
    data_diff = np.abs(np.diff(data))
    
    # Add the last element of the data_diff list to avoid sample count error
    data_diff = np.append(data_diff, data_diff[-1])
    
    # Create a mask for high fluctuations based on the motion detection limit
    high_fluctuation_mask = data_diff > motion_detection_limit
    
    # Print the high fluctuation mask
    # print(high_fluctuation_mask)
    
    # Label the high fluctuation regions
    labelled_mask, num_labels = ndimage.label(high_fluctuation_mask)
    
    # Print the labelled mask and the number of labels
    # print(labelled_mask)
    # print(num_labels)
    
    high_fluctuation_ranges = []
    
    # Iterate over each label and extract the start and end time indices
    for label_num in range(1, num_labels+1):
        label_indices = np.where(labelled_mask == label_num)[0]
        start_time = label_indices[0]
        end_time = label_indices[-1]
        high_fluctuation_ranges.append((start_time, end_time))
    
    # Print the high fluctuation ranges
    # print(high_fluctuation_ranges)
    
    return high_fluctuation_ranges, data_diff


def merge_adjacent_ranges(high_fluctuation_ranges, max_samples):
    """
    Merge adjacent ranges of high fluctuations based on the maximum number of samples.

    Args:
        high_fluctuation_ranges (list): List of tuples representing the start and end indices of high fluctuation ranges.
        max_samples (int): Maximum number of samples for merging adjacent ranges.

    Returns:
        list: List of tuples representing the merged adjacent ranges.
    """
    merged_ranges = high_fluctuation_ranges.copy()
    i = 0
    while i < len(merged_ranges):
        if i < len(merged_ranges) - 1:
            if merged_ranges[i+1][0] - merged_ranges[i][1] < max_samples:
                merged_ranges[i] = (merged_ranges[i][0], merged_ranges[i+1][1])
                del merged_ranges[i+1]
            else:
                i += 1
        else:
            break
    return merged_ranges


def detectMotion(ppg_data_csv_file_path):
    """
    detectMotion: This Function will detect motion ranges from given acceleration data. 
    :param in -  ppg_data_csv_file_path: CSV File path with PPG Raw Data value 
    :param out - Sample Ranges for Detected Motion (Also Saves Plot)
    """
    # Open CSV File Data
    df = pd.read_csv(ppg_data_csv_file_path, skiprows=15)

    # Read Acceleration Data
    acceleration_data = df[['Time (ms)', 'IR (counts)', 'Ax (mg)', 'Ay (mg)', 'Az (mg)']]
    A_x = acceleration_data['Ax (mg)']
    A_y = acceleration_data['Ay (mg)']
    A_z = acceleration_data['Az (mg)']
    ppg_data = acceleration_data['IR (counts)']
    max_samples = acceleration_data.index.size

    fs = 200 # Hz

    # Step 1:Set threshold value for all 3 Axis Acceleration data
    #-------------------------------------------------------------
    MOTION_DETECTION_LIMIT = 700
    PPG_FLUCTUATION_LIMIT = 2000

    # Step2: Find the Sample range of high fluctuations for All axis
    #-------------------------------------------------------------
    # 1. A_x (X Axis Computation)
    A_x_high_fluctuation_ranges, A_x_diff = detect_high_fluctuation_ranges(A_x, MOTION_DETECTION_LIMIT)

    # 2. A_y (Y Axis Computation)
    A_y_high_fluctuation_ranges, A_y_diff = detect_high_fluctuation_ranges(A_y, MOTION_DETECTION_LIMIT)

    # 3. A_z (Z Axis Computation)
    A_z_high_fluctuation_ranges, A_z_diff = detect_high_fluctuation_ranges(A_z, MOTION_DETECTION_LIMIT)

    # 4. PPG Data Fluctuation
    ppg_data_high_fluctuation_ranges, ppg_data_diff = detect_high_fluctuation_ranges(ppg_data, PPG_FLUCTUATION_LIMIT)


    # Step3: Merge adjacent ranges of high fluctuations for all axis(Checks for 400 Samples)
    #---------------------------------------------------------------------------
    # 1. A_x (X Axis Computation)
    A_x_high_fluctuation_ranges = merge_adjacent_ranges(A_x_high_fluctuation_ranges, 400)
    # print(A_x_high_fluctuation_ranges)

    # 2. A_y (Y Axis Computation)
    A_y_high_fluctuation_ranges = merge_adjacent_ranges(A_y_high_fluctuation_ranges, 400)
    # print(A_y_high_fluctuation_ranges)

    # 3. A_z (Z Axis Computation)
    A_z_high_fluctuation_ranges = merge_adjacent_ranges(A_z_high_fluctuation_ranges, 400)
    # print(A_z_high_fluctuation_ranges)

    # 4. PPG Data
    ppg_data_high_fluctuation_ranges = merge_adjacent_ranges(ppg_data_high_fluctuation_ranges, 400)
    # print(ppg_data_high_fluctuation_ranges)


    # Step4: Merge ranges of of all axis to get final ranges
    #---------------------------------------------------------------------------
    # get a list of ranges of all 3 axis
    print("A_x_high_fluctuation_ranges: ", A_x_high_fluctuation_ranges)
    print("A_y_high_fluctuation_ranges: ", A_y_high_fluctuation_ranges)
    print("A_z_high_fluctuation_ranges: ", A_z_high_fluctuation_ranges)
    print("ppg_data_high_fluctuation_ranges: ", ppg_data_high_fluctuation_ranges)
    allAxisRanges = []
    for Range in A_x_high_fluctuation_ranges:
        allAxisRanges.append(Range)
    for Range in A_y_high_fluctuation_ranges:
        allAxisRanges.append(Range)
    for Range in A_z_high_fluctuation_ranges:
        allAxisRanges.append(Range)
    for Range in ppg_data_high_fluctuation_ranges:
        allAxisRanges.append(Range)
    
    allAxisRangesMerged = merge_ranges(allAxisRanges)
    # allAxisRangesMerged = remove_small_ranges(allAxisRangesMerged, fs/16)

    # Step5: Add 1 Seconds Padding to allAxisRangesMerged
    #---------------------------------------------------------------------------
    # So here we add 1 seconds (Sampling Frequency fs = 200Hz in our case)
    for i in range(len(allAxisRangesMerged)):
        allAxisRangesMerged[i] = (max(0, allAxisRangesMerged[i][0] - fs), min(max_samples, allAxisRangesMerged[i][1] + fs))

    # Step7 After padding make sure no ranges are overlapped if yes than merge it
    finalMotionRanges = merge_ranges(allAxisRangesMerged)


    # Plot the results
    fig, axs = plt.subplots(8, 1, figsize=(20,20), sharex=True)
    plt.style.use('fivethirtyeight')
    axs[0].plot(acceleration_data.index, acceleration_data['Ax (mg)'], linewidth=1)
    axs[0].set_ylabel('Ax (Mag)')
    axs[1].plot(acceleration_data.index, A_x_diff, linewidth=1)
    axs[1].plot(acceleration_data.index, [MOTION_DETECTION_LIMIT]*max_samples, linewidth=2)
    for A_x_high_fluctuation_range in A_x_high_fluctuation_ranges: 
        axs[1].axvline(A_x_high_fluctuation_range[0], color='red', linestyle='--', linewidth=1)
        axs[1].axvline(A_x_high_fluctuation_range[1], color='red', linestyle='--', linewidth=1)
    axs[1].set_ylabel('Ax (diff)')

    axs[2].plot(acceleration_data.index, acceleration_data['Ay (mg)'], linewidth=1)
    axs[2].set_ylabel('Ay (Mag)')
    axs[3].plot(acceleration_data.index, A_y_diff, linewidth=1)
    axs[3].plot(acceleration_data.index, [MOTION_DETECTION_LIMIT]*max_samples, linewidth=2)
    for A_y_high_fluctuation_range in A_y_high_fluctuation_ranges: 
        axs[3].axvline(A_y_high_fluctuation_range[0], color='red', linestyle='--', linewidth=1)
        axs[3].axvline(A_y_high_fluctuation_range[1], color='red', linestyle='--', linewidth=1)
    axs[3].set_ylabel('Ay (diff)')

    axs[4].plot(acceleration_data.index, acceleration_data['Az (mg)'], linewidth=1)
    axs[4].set_ylabel('Az (Mag)')
    axs[5].plot(acceleration_data.index, A_z_diff, linewidth=1)
    axs[5].plot(acceleration_data.index, [MOTION_DETECTION_LIMIT]*max_samples, linewidth=2)
    for A_z_high_fluctuation_range in A_z_high_fluctuation_ranges: 
        axs[5].axvline(A_z_high_fluctuation_range[0], color='red', linestyle='--', linewidth=1)
        axs[5].axvline(A_z_high_fluctuation_range[1], color='red', linestyle='--', linewidth=1)
    axs[5].set_ylabel('Az (diff)')

    
    axs[6].plot(acceleration_data.index, ppg_data_diff, linewidth=1)
    axs[6].plot(acceleration_data.index, [PPG_FLUCTUATION_LIMIT]*max_samples, linewidth=2)
    for ppg_data_high_fluctuation_range in ppg_data_high_fluctuation_ranges: 
        axs[6].axvline(ppg_data_high_fluctuation_range[0], color='red', linestyle='--', linewidth=1)
        axs[6].axvline(ppg_data_high_fluctuation_range[1], color='red', linestyle='--', linewidth=1)
    axs[6].set_ylabel('PPG Data (diff)')
    
    axs[7].plot(acceleration_data.index, ppg_data, linewidth=1)
    for finalMotionRange in finalMotionRanges: 
        axs[7].axvline(finalMotionRange[0], color='red', linestyle='--', linewidth=1)
        axs[7].axvline(finalMotionRange[1], color='red', linestyle='--', linewidth=1)
    axs[7].set_ylabel('PPG Data')

    plt.suptitle('Motion Detection')
    PATH = ppg_data_csv_file_path + '(Motion_Detection_Plot).png'
    plt.savefig(PATH)

    return finalMotionRanges


def generateValidPPGDataCSVFiles(ppg_data_csv_file_path, motionRanges, csv_file_name):
    """
    generateValidPPGDataCSVFiles: This Function will generate new CSV Files of Valid Data removing any motion based on motionRanges 
    :param in -     1) ppg_data_csv_file_path: CSV File path with PPG Raw Data value, 
                    2) motionRanges(See Details above def detectMotion(): )
                    3) csv_file_name: String (ex. '2023-05-10-001S-22-05-32-S-P1.csv')
    :param out -    List of PATHs of CSV Files(Generates CSV Files in '\ValidCSVFiles' Folder) '2023-05-10-001S-22-05-32-S-P1.csv', '2023-05-10-001S-22-05-32-S-P2.csv'..
    """
    # Open CSV File Data
    df = pd.read_csv(ppg_data_csv_file_path, skiprows=15)

    # number of samples
    max_samples = df.index.size
    fs = 200
    
    # Step1: Generating includeRages based on excludeRanges(In our cases it will be motionRanges)
    #---------------------------------------------------------------------------------------------
    exclude_ranges = motionRanges
    # Create a list of excluded indices
    excluded_indices = []
    for r in exclude_ranges:
        excluded_indices += list(range(r[0], r[1]+1))
    # Create a list of included ranges
    included_ranges = []
    start_index = 0
    for i in range(max_samples):
        if i in excluded_indices:
            if i > start_index:
                included_ranges.append((start_index, i-1))
            start_index = i + 1
    # Add the final included range, if applicable
    if start_index < max_samples:
        included_ranges.append((start_index, max_samples-1))
    # Print the list of included ranges
    print("Motion Ranges: ", exclude_ranges)
    print("No Motion Rages: ", included_ranges)


    # Step2: Keep only No Motion Rages which has more than 1 seconds of samples (In our case its fs=200)
    included_ranges = [range_tuple for range_tuple in included_ranges if range_tuple[1] - range_tuple[0] >= 4*fs]


    # Step3: Generating Multiple CSV Files based on included_ranges (In our cases it will be No Motion Ranges)
    #---------------------------------------------------------------------------------------------
    valid_csv_files = []
    if not os.path.exists(CSV_VALID_FOLDER_NAME):
        os.makedirs(CSV_VALID_FOLDER_NAME)
    # Loop through the included ranges and create a new CSV file for each range
    for i, r in enumerate(included_ranges):
        # Create a new DataFrame that includes only the rows corresponding to the current included range
        df_range = df.iloc[r[0]:r[1]+1]
        # Set the name of the output CSV file
        output_file = f'{CSV_VALID_FOLDER_NAME}/{csv_file_name}-P{i}.csv'
        print(f"Range: {r} --> {output_file}")
        
        # Write the DataFrame to a new CSV file
        df_range.to_csv(output_file, index=False)

        valid_csv_files.append(output_file) 

    return valid_csv_files


def measureHeartRate(ppg_data_csv_file_path, plot_save = False, no_skip_rows=0):
    """
    measureHeartRate: This Function will measure HeartRate from PPG Raw Data CSV File and saves peak detection plot in the same location. 
    :param in -  ppg_data_csv_file_path: CSV File path with PPG Raw Data value, plot_save: True: if user wants to save peak detection plot
    :param out - heartRate: Heart Rate in bpm or 0(Error)
    """ 
    ### Step0: Get Raw PPG signal
    #----------------------------------------------
    # Select channel from the dataset.
    channel = 'IR (counts)'

    # Open CSV File Data
    df = pd.read_csv(ppg_data_csv_file_path, skiprows=no_skip_rows)

    df.head()

    # Data Sampled at 200Hz
    ppg_data = df[['Time (ms)', channel]]

    max_samples = ppg_data[channel].size
    # max_samples = 5000

    ppg_data = ppg_data.iloc[:max_samples]

    ### Step1: Filter the PPG signal with a bandpass filter.
    #----------------------------------------------
    # Low-pass filter the raw signal to remove noise
    fs = 200  # Sampling frequency
    fc =  4 # Cutoff frequency
    b, a = butter(2, 2 * fc / fs, 'low')

    ppg_data_raw = ppg_data[channel]
    #Applying LP Filter to ppg_data[channel]
    ppg_data[channel] = filtfilt(b, a, ppg_data[channel])

    ppg_data_filtered = ppg_data[channel]

    ### Step2: Check signal quality by calculating Signal to Noise Ratio.
    #----------------------------------------------
    # Calculating SNR to find quality of a signal
    # Noise signal
    THRESHOLD = 25      #dB
    noise_data = ppg_data_raw - ppg_data_filtered

    signal_power = np.mean(np.abs(ppg_data_filtered)**2)
    noise_power = np.mean(np.abs(noise_data)**2)
    SNR = 10*np.log10(signal_power/noise_power)
    # print("SNR: ", SNR)
    if SNR < THRESHOLD:
        return "Error"
        # print("Signal Quality is Bad!")
    # else: 
        # print("Signal Quality is Good!")

    ### Step3: Differentiate the filtered signal
    #----------------------------------------------
    # Differentiate the filtered signal to emphasize the high-frequency components
    # ppg_data[channel] = np.gradient(ppg_data[channel])
    ppg_data_diff = np.diff(ppg_data[channel])
    ppg_data_diff = np.concatenate([ppg_data_diff, [ppg_data_diff[-1]]])
    ppg_data[channel] = ppg_data_diff

    ### Step4: Removing Abnormalities
    #----------------------------------------------  
    for i in range(2):
        ppg_data_mean = np.mean(ppg_data[channel])
        ppg_data_std = np.std(ppg_data[channel])

        # print(ppg_data_mean)
        # print(ppg_data_std)

        threshold = ppg_data_mean + 2 * ppg_data_std  # set the threshold as 2 times the standard deviation above the mean
        # print(threshold)

        ppg_data[channel] = ppg_data[channel].apply(lambda x: x if abs(x) <= threshold else threshold)

    ### Step5: Square the differentiated  signal.
    #---------------------------------------------- 
    # Square the differentiated signal to enhance QRS complex
    # ppg_data[channel] = np.square(ppg_data[channel])

    ### Step6: Integrate the squared signal with a sliding window.
    #---------------------------------------------- 
    # Apply a moving average integration to smooth the signal
    for i in range(2):
        window_size = int(0.25 * fs)  #  window size
        padded = np.pad(ppg_data[channel], (window_size, window_size), mode='edge')
        window = np.ones(window_size) / float(window_size)
        padded_convolved = np.convolve(padded, window, "same")
        ppg_data[channel] = padded_convolved[window_size:-window_size]
        ppg_data[channel] = np.convolve(ppg_data[channel], window, "same")

    ### Step7: Find the R-peaks in the integrated signal.
    #---------------------------------------------- 
    # Find peaks in the integrated signal
    ppg_peaks, _ = find_peaks(ppg_data[channel], distance=0.375*fs)
    # print("ppg_peaks:", ppg_peaks)
    # print("ppg_peaks size:", ppg_peaks.size)

    # Get a list of Amplitude of peaks
    ppg_peaks_amplitudes = [ppg_data[channel][i] for i in ppg_peaks]

    # max_peak_amplitude = max(ppg_peaks_amplitudes)
    # print("max_peak_amplitude:", max_peak_amplitude)

    # ppg_peaks_amplitudes_mean = np.mean(ppg_peaks_amplitudes)
    # ppg_peaks_amplitudes_std = np.std(ppg_peaks_amplitudes)

    # print(ppg_peaks_amplitudes_mean)
    # print(ppg_peaks_amplitudes_std)

    # threshold = ppg_peaks_amplitudes_mean + 0.25 * ppg_peaks_amplitudes_std  # set the threshold as 2 times the standard deviation above the mean
    # print(threshold)
    # PEAK_AMPLITUDE_THRESHOLD = threshold
    # print("PEAK_AMPLITUDE_THRESHOLD:", PEAK_AMPLITUDE_THRESHOLD)

    # ppg_peaks_refined = []
    # for ppg_peak in ppg_peaks:
    #     if(ppg_data[channel][ppg_peak] > PEAK_AMPLITUDE_THRESHOLD):
    #         # Amplitude is big enough to be considered as a peak
    #         ppg_peaks_refined.append(ppg_peak)

    # print(ppg_peaks_refined)

    # ppg_peaks_refined_amplitudes = [ppg_data[channel][i] for i in ppg_peaks_refined]

    if plot_save:
        # Plot the diff_ppg data
        plt.figure(figsize=(10,5))
        plt.style.use('fivethirtyeight')
        #plot old plot
        plt.plot(ppg_data.index, ppg_data[channel], linewidth=1)
        # plot red dots with detected peaks
        plt.plot(ppg_peaks, ppg_peaks_amplitudes, 'ro')
        plt.title("Peak Detection")
        plt.xlabel('Time')
        plt.ylabel('Moving Average PPG Data')
        PATH = ppg_data_csv_file_path + '(Peaks_Plot).png'
        plt.savefig(PATH)

    ### Step8: Calculate the heart rate using the time difference between R-peaks.
    #----------------------------------------------  
    # Minimum number of Peaks Check
    total_time_seconds = max_samples/fs

    # Minimum Heartbeat = 25bpm
    # Maximum Heartbeat = 250bpm
    HEARTRATE_LOWER_THRESHOLD = 25
    HEARTRATE_HIGHER_THRESHOLD = 250

    # Minmum samples between two peaks

    MIN_NO_OF_PEAKS_THRESHOLD = (HEARTRATE_LOWER_THRESHOLD/60)*total_time_seconds
    MAX_NO_OF_PEAKS_THRESHOLD = (HEARTRATE_HIGHER_THRESHOLD/60)*total_time_seconds

    # print("MIN_NO_OF_PEAKS_THRESHOLD = ", MIN_NO_OF_PEAKS_THRESHOLD)
    # print("MAX_NO_OF_PEAKS_THRESHOLD = ", MAX_NO_OF_PEAKS_THRESHOLD)

    TOTAL_PEAKS_DETECTED = len(ppg_peaks)
    # print("TOTAL_PEAKS_DETECTED = ", TOTAL_PEAKS_DETECTED)

    if (TOTAL_PEAKS_DETECTED < MIN_NO_OF_PEAKS_THRESHOLD) or (TOTAL_PEAKS_DETECTED > MAX_NO_OF_PEAKS_THRESHOLD) or (max_samples < 500):
        return "Error"
        # print("Heart rate for file sample_ppg.csv: Error")
    else:
        # Compute the inter-beat interval (IBI) and heart rate (HR) from the peak locations
        ibi = np.diff(ppg_peaks) / fs  # IBI in seconds
        hr = 60*TOTAL_PEAKS_DETECTED/total_time_seconds
        hrv = 60 / ibi  # HRV in bpm
        hrv = np.concatenate([hrv, [hrv[-1]]])

        #HRV Plot
        if plot_save:
            # Plot the HRV data
            plt.figure(figsize=(10,5))
            plt.style.use('fivethirtyeight')
            # plot red dots with detected peaks
            plt.plot(ppg_peaks, hrv, linewidth=1)
            plt.scatter(ppg_peaks, hrv, color='red')
            plt.ylim(30, 160)  # Custom range: from 0 to 40
            plt.title("HRV")
            plt.xlabel('Peaks')
            plt.ylabel('HR')
            PATH = ppg_data_csv_file_path + '(HRV_Plot).png'
            plt.savefig(PATH)

     
        # Output the heart rate measurement
        return hr
        # print("Heart rate for file sample_ppg.csv:", np.mean(hr), "bpm") 
 
    







