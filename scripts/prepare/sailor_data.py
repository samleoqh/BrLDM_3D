import os
import pandas as pd

from tqdm import tqdm

import numpy as np

root = '/mnt/scratch/brian/datasets/sailor-mni'  
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


def add_treatment_column(file_path, output_path):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)
    
    # 检查是否存在 'session' 列
    if 'session' not in df.columns:
        raise ValueError("CSV文件中缺少 'session' 列")
    
    # 新增 'treatment' 列，根据 'session' 列的值进行赋值
    df['treatment'] = df['session'].apply(lambda x: 0.2 if x == 'ses-01' or x == 'ses-02' or x == 'ses-03' or x == 'ses-04' else 0.4)
    
    # 保存修改后的 DataFrame 到新的 CSV 文件
    df.to_csv(output_path, index=False)
    print(f"已成功将文件保存到 {output_path}")

# 使用方法
# file_path = 'sailor_dict.csv'  # 输入的CSV文件路径
# output_path = 'sailor_dict2.csv'       # 输出的CSV文件路径
# add_treatment_column(file_path, output_path)


def add_latent_column(file_path, output_path):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)
    
    # 检查是否存在 'session' 列
    if 'image_path' not in df.columns:
        raise ValueError("CSV文件中缺少 'image_path' 列")
    
    # 新增 'treatment' 列，根据 'session' 列的值进行赋值
    df['latent_path'] = df['image_path'].apply(lambda x: x.replace('.nii.gz', '_latent.npz').replace('.nii', '_latent.npz'))
    
    # 保存修改后的 DataFrame 到新的 CSV 文件
    df.to_csv(output_path, index=False)
    print(f"已成功将文件保存到 {output_path}")
    
    
# def add_age_column(file_path, output_path):
#     # 读取 CSV 文件
#     df = pd.read_csv(file_path)
#     ag = pd.read_csv(time_csv)
    
def add_age_column(existing_file, output_file):
    # Load both CSV files
    df_existing = pd.read_csv(existing_file)
    df_age = pd.read_csv(time_csv)
    
    # Ensure the required columns exist in both dataframes
    if 'subject' not in df_existing.columns or 'patients' not in df_age.columns:
        raise ValueError("The existing CSV must contain 'subject' and the age CSV must contain 'patients' columns to match on.")
    
    # Perform the merge, matching 'subject' in existing CSV to 'patients' in age CSV
    df_merged = pd.merge(df_existing, df_age[['patients', 'age']], 
                         left_on='subject', right_on='patients', how='left')
    
    # Drop the extra 'patients' column that comes from the age CSV
    df_merged.drop(columns=['patients'], inplace=True)
    
    # Save the updated DataFrame to a new CSV file
    df_merged.to_csv(output_file, index=False)
    print(f"File saved successfully with the age column added: {output_file}")


def add_cumulative_time_column(file_path, output_file):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Sort by 'subject' and 'session' to ensure proper ordering
    df = df.sort_values(['subject', 'session']).reset_index(drop=True)
    
    # Calculate cumulative time for each subject
    df['time'] = df.groupby('subject')['time_interval'].cumsum() - df['time_interval'].iloc[0]

    # Save the updated DataFrame with the new 'time' column
    df.to_csv(output_file, index=False)
    print(f"File saved successfully with cumulative time column added: {output_file}")


def make_csv_B(df):
    """
    Creates CSV B, which contains all possible pairs (x_a, x_b) such that 
    both scans belong to the same patient and scan x_a is acquired before scan x_b.
    """
    sorting_field = 'months_to_screening' if 'months_to_screening' in df.columns else 'time'

    data = []
    for subject_id in tqdm(df.subject.unique()):
        subject_df = df[ df.subject == subject_id ].sort_values(sorting_field, ascending=True)
        for i in range(len(subject_df)):
            for j in range(i+1, len(subject_df)):
                s_rec = subject_df.iloc[i]
                e_rec = subject_df.iloc[j]
                record = { 'subjec': s_rec.subject, 'split': s_rec.split }
                remaining_columns = set(df.columns).difference(set(record.keys()))
                for column in remaining_columns:
                    record[f'starting_{column}'] = s_rec[column]
                    record[f'followup_{column}'] = e_rec[column]
                data.append(record)
    return pd.DataFrame(data)


csv_B = make_csv_B(pd.read_csv('A4.csv'))
csv_B.to_csv('B.csv', index=False)

# Usage example
# file_path = 'A3.csv'          # Path to the input CSV file
# output_file = 'A4.csv'      # Path to save the updated CSV with the new 'time' column
# add_cumulative_time_column(file_path, output_file)

# file_path = 'A2.csv'  # 输入的CSV文件路径
# output_path = 'A3.csv'       # 输出的CSV文件路径
# add_age_column(file_path, output_path)

# files = get_file_dict()
# time_intervals = get_time_intervals()

# save_to_csv(files, time_intervals)