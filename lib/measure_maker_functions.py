import numpy as np
import pandas as pd

def measure_two_points_from_csv (path_csv_file, LANDMK_INIT_PT, LANDMK_END_PT, measure_name):
    
    # Read the Data frame from CSV
    csv_data_frame_in = pd.read_csv(path_csv_file)
    
    if "Unnamed: 0" in csv_data_frame_in.columns:
    # Remove the Unnamed columns
        csv_data_frame_in.drop(columns=["Unnamed: 0"], inplace=True)
    
    # Calculate the Number Of Frames
    NUMBER_OF_FRAMES_IN = len(csv_data_frame_in)
    
    # Measurements path
    CSV_IN_MEASUREMENTS = []
    
    # Iterate and Measure
    for idx in range(0, NUMBER_OF_FRAMES_IN):
        
        # INIT Point XY
        POINT_INIT_XYLMK = np.asarray(eval(csv_data_frame_in[str(LANDMK_INIT_PT)][idx]))
        
        # END Point XY
        POINT_END_XYLMK = np.asarray(eval(csv_data_frame_in[str(LANDMK_END_PT)][idx]))
        
        # Measure
        distance_open_mouth_basic = np.linalg.norm(POINT_END_XYLMK - POINT_INIT_XYLMK) 
        
        # Append in the array
        CSV_IN_MEASUREMENTS.append(distance_open_mouth_basic)
    
    # Create a DataFrame
    MEASURE_RESULTS_DATA_FRAME = pd.DataFrame(CSV_IN_MEASUREMENTS, columns=[measure_name])
    return MEASURE_RESULTS_DATA_FRAME

def measure_vertical_two_arrays_mean (path_csv_file, POINT_ARRAY_INIT, POINT_ARRAY_END, measure_name):
    
    # Read the Data frame from CSV
    csv_data_frame_in = pd.read_csv(path_csv_file)
    
    # Remove the Unnamed columns
    if "Unnamed: 0" in csv_data_frame_in.columns:
        csv_data_frame_in.drop(columns=["Unnamed: 0"], inplace=True)
    
    # Calculate the Number Of Frames
    NUMBER_OF_FRAMES_IN = len(csv_data_frame_in)
    
    # Measurements path
    CSV_IN_MEASUREMENTS = []
    
    # Iterate and Measure
    for idx in range(0, NUMBER_OF_FRAMES_IN):

        # INIT Point Y
        VALUE_Y_INIT = []
        for curr_colect in POINT_ARRAY_INIT:
            basic_to_add = np.asarray(eval(csv_data_frame_in[str(curr_colect)][idx]))
            VALUE_Y_INIT.append (basic_to_add[1])

        # END Point Y
        VALUE_Y_END = []
        for curr_colect in POINT_ARRAY_END:
            basic_to_add = np.asarray(eval(csv_data_frame_in[str(curr_colect)][idx]))
            VALUE_Y_END.append (basic_to_add[1])

        # Calculate the mean
        mean_initial = np.mean(VALUE_Y_INIT)
        mean_end = np.mean(VALUE_Y_END)

        # Measure
        distance_open_mouth_basic = abs(mean_initial - mean_end)
        
        # Append in the array
        CSV_IN_MEASUREMENTS.append(distance_open_mouth_basic)
    
    # Create a DataFrame
    MEASURE_RESULTS_DATA_FRAME = pd.DataFrame(CSV_IN_MEASUREMENTS, columns=[measure_name])
    return MEASURE_RESULTS_DATA_FRAME