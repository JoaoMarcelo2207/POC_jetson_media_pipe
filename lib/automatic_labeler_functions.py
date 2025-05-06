import stumpy
import os
import pandas as pd
import numpy as np
import ast
import re

#from stumpy import config
#config.STUMPY_EXCL_ZONE_DENOM = 1

# Class to manage the searches
class Comparing:
    def __init__(self, Q_df, T_df, distance_threshold):
        self.Q_df = Q_df # Seed
        self.T_df = T_df # Entire Video
        self.matches_idxs = [] # Store the matches with distance threshold equals to this: max(np.mean(D) - self.distance_threshold * np.std(D), np.min(D))
        self.all_matches_idxs = [] # Store the matches with distance threshold equals to 9e100
        self.all_mass_idxs = [] # Store all the matches (Using mass function to do this because the match function do not get all the possible matches\)
        self.filter_matches_idxs = [] # Indexes selected after the matching process
        self.measure_name = None
        self.distance_threshold = distance_threshold

    def calc_matches(self):
        self.matches_idxs = stumpy.match(self.Q_df, self.T_df, max_distance=lambda D: max(np.mean(D) - self.distance_threshold * np.std(D), np.min(D)))
        self.all_matches_idxs = stumpy.match(self.Q_df, self.T_df, max_distance=lambda D: max(9e100, np.min(D)))
        self.all_mass_idxs = stumpy.mass(self.Q_df, self.T_df)

def get_euclidean_distance(target, matrix):
    for subarray in matrix:
        if target in subarray:
            return subarray[0]
    return None

def label_current_series(current_path_location, RESUME_DT, selected_measures_in_frame_interval, dict_label_parameters, seed_name, LABELED_FILE_NAME='VD_LABELED_L0.CSV', distance_threshold=2, frame_threshold=3):
    VD_MEASURE_DT = pd.read_csv(current_path_location)
    VD_MEASURE_DT.drop(columns=['Unnamed: 0'], inplace=True)

    T_df = VD_MEASURE_DT[dict_label_parameters['reference_measures']]

    # Apply Stumpy functions
    object_list = []
    temp_row = pd.DataFrame()

    matches_memory = []
    all_matches_memory = []
    all_mass_memory = []
    
    for step in range(0, len(selected_measures_in_frame_interval.columns)):
        comp_object = Comparing(selected_measures_in_frame_interval[dict_label_parameters['reference_measures'][step]], T_df[dict_label_parameters['reference_measures'][step]], distance_threshold)
        comp_object.calc_matches()

        matches_memory.append(comp_object.matches_idxs)
        all_mass_memory.append(comp_object.all_mass_idxs)
        all_matches_memory.append(comp_object.all_matches_idxs)

        comp_object.measure_name = dict_label_parameters['reference_measures'][step]
        object_list.append(comp_object)
        
        # Count the number of rows
        temp_row.at[0, dict_label_parameters['reference_measures'][step]] = int(len(comp_object.matches_idxs))
        
    # Apply the matching filter
    all_index = []
    for c in object_list:  
        all_index.append(c.matches_idxs[:, 1])
    
    # Filter by coincidence from a distance threshold between the position of each indexes
    aux = all_index.copy()
    filter_index = find_all_matches(aux, frame_threshold)
    
    filter_index_list=list(filter_index[0])

    # Fix the subseries index by the original frame index (frame_seq)
    filter_index_begin = []
    for idx_tuple in filter_index_list:
        filter_index_begin.append(idx_tuple)

    idxs_match_frame_seq = []
    for idx in filter_index_begin:
        idx_frame_seq = VD_MEASURE_DT.loc[idx, 'frame_seq']
        for c in object_list:
            ed = get_euclidean_distance(idx, c.matches_idxs)
            if ed != None: break
        idxs_match_frame_seq.append([idx_frame_seq, ed])
            
    # Test if the Labeled File was already created
    test = os.path.exists((os.path.join(os.path.dirname(current_path_location), LABELED_FILE_NAME)))
    VD_LABEL_PATH = (os.path.join(os.path.dirname(current_path_location), LABELED_FILE_NAME))
    
    if test:
        VD_LABELED_DT = pd.read_csv(VD_LABEL_PATH)
        VD_LABELED_DT.drop(columns=['Unnamed: 0'], inplace=True)
        VD_LABELED_DT = VD_LABELED_DT.set_index(pd.Index(VD_LABELED_DT['frame_seq']))
    else:
        # First Initiate the labels = 0 means NO Label
        VD_LABELED_DT = VD_MEASURE_DT.copy()
        VD_LABELED_DT['label_measures'] = str({})
        VD_LABELED_DT = VD_LABELED_DT.set_index(pd.Index(VD_LABELED_DT['frame_seq']))
    
    temp_row['final'] = (len(filter_index[0]))

    gap_ocurrences = 0
    occurences_len = []
    
    # Adds information to label the frames.
    for label_idx in idxs_match_frame_seq:
        init_lab = label_idx[0]
        endd_lab = init_lab+len(selected_measures_in_frame_interval)-1
        e_distace = label_idx[1]
        FRAMES_DT = VD_LABELED_DT.query(f'frame_seq >= {init_lab} & frame_seq <= {endd_lab}')
        occurences_len.append(len(FRAMES_DT))
        
        # if there is not a descontinuity in the interval of frames
        if not FRAMES_DT['gap'].any()==1 and endd_lab in VD_LABELED_DT.index:
            
            # In cases that the missing frames are in the end of interval
            VD_LABELED_DT = UPDATE_LABEL_DF(init_lab, endd_lab, dict_label_parameters['label_name'], dict_label_parameters['reference_measures'], VD_LABELED_DT, seed_name, e_distace)
        else:
            gap_ocurrences += 1

    temp_row['final'] -= gap_ocurrences
    RESUME_DT = pd.concat([RESUME_DT, temp_row], axis=0)
    
    # Save CSV file
    VD_LABELED_DT.drop(columns=['frame_seq'], inplace=True)
    VD_LABELED_DT.reset_index(inplace=True)
    VD_LABELED_DT.to_csv(VD_LABEL_PATH)

    return RESUME_DT, matches_memory, all_matches_memory, all_mass_memory, idxs_match_frame_seq, occurences_len

def UPDATE_LABEL_DF (init_lab, endd_lab, label_name_in, label_measur_in, data_frame_in, seed_name, matches_idxs):
    
    # Check if END is Greater than Length
    for index_x in range(init_lab, endd_lab+1):
        idx_retur_str = data_frame_in['label_measures'][index_x]
        dicct_current = ast.literal_eval(idx_retur_str)

        match = re.search(r'VD_R_(\d+)', seed_name)
        video_num = int(match.group(1)) if match else seed_name

        # Insert Updating DICT
        dicct_current.update ({label_name_in: (label_measur_in, matches_idxs, video_num)})
        
        # Put Dict into the Current DATA FRAME
        data_frame_in.loc[index_x, 'label_measures'] = str(dicct_current)
    
    return data_frame_in

def find_close_values(idxs, threshold):
    close_values = []

    # Compare each pair of lists
    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            list1 = idxs[i]
            list2 = idxs[j]

            # Compare every element of both lists
            for num1 in list1:
                for num2 in list2:

                    # If the distance between the values are smaller than the threshold, consider it accepted.
                    if abs(num1 - num2) <= threshold:
                        close_values.append(min(num1,num2))
    close_values = set(close_values)
    return close_values

def find_all_matches(list_of_index, threshold):
    n = len(list_of_index)
    list_aux = []
    
    if n <= 1:  
        return list_of_index
    else: 
        # Select the first and second one on the similarity search
        list_aux.append(list_of_index.pop(0))
        list_aux.append(list_of_index.pop(0))

        result = find_close_values(list_aux, threshold)
        list_of_index.insert(0, result)
        
        return find_all_matches(list_of_index, threshold)
