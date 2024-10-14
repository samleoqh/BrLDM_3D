import os
import pandas as pd

import numpy as np

root = '/mnt/HDD18TB/brian/datasets/sailor_mni_v2'  
time_csv = os.path.join(root, "sailor_info.csv" )

key_list = ['t1']

# {
#     "edema_mask": "EdemaMask-CL.nii.gz",
#     "et_mask": "ContrastEnhancedMask-CL.nii.gz",
#     "t1": "T1-icor.nii.gz",
#     "t1c": "T1c-icor.nii.gz",
#     "flair": "Flair-icor.nii.gz",
#     "t2": "T2-icor.nii.gz",
# }

kEY_FILENAMES = {
    "t1": "T1.nii.gz",
}

all_patient_ids = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06',
    'sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-11', 'sub-12',
    'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18',
    'sub-19', 'sub-20', 'sub-21', 'sub-22', 'sub-23', 'sub-24',
    'sub-25', 'sub-26', 'sub-27'
]

def get_file_dict(patient_ids=all_patient_ids):
    file_dict = {}
    for subj in patient_ids:
        ses_ids = sorted([os.path.basename(f.path) for f in os.scandir(os.path.join(root, subj)) if f.is_dir()])
        for ses_id in ses_ids:
            session_files = {k: os.path.join(root, subj, ses_id, kEY_FILENAMES.get(k, '')) for k in key_list}
            file_dict[(subj, ses_id)] = session_files
    return file_dict


def get_time_intervals(file_csv=time_csv):
    times_dict = {}
    df = pd.read_csv(file_csv)
    df.set_index("patients", inplace=True)
    
    for patient_id in all_patient_ids:
        interval_str = df['interval_days'].loc[patient_id]
        time_intervals = np.asarray([int(s.strip()) for s in interval_str[1:-1].split(",")])
        times_dict[patient_id] = np.insert(time_intervals, 0, 0)
    return times_dict

def save_to_csv(file_dict, output_file='sailor_dict.csv'):

    rows = []
    for (subj, ses_id), files in file_dict.items():
        row = {'subject': subj, 'session': ses_id}
        row.update(files)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

def save_to_csv(file_dict, time_intervals, output_file='sailor_dict.csv'):
    rows = []
    for (subj, ses_id), files in file_dict.items():
        row = {'subject': subj, 'session': ses_id}
        row.update(files)
        row['image_path'] = files['t1']
        # Get all ses_ids from subj
        subj_ses_ids = sorted([os.path.basename(f.path) for f in os.scandir(os.path.join(root, subj)) if f.is_dir()])
        
        # Get subj_ses_ids index in ses_id
        try:
            index = subj_ses_ids.index(ses_id)
            row['time_interval'] = time_intervals[subj][index]
        except ValueError:
            row['time_interval'] = None 
            
        row['split'] = 'train'
        row['treatment'] = 0
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

files = get_file_dict()
time_intervals = get_time_intervals()

save_to_csv(files, time_intervals)