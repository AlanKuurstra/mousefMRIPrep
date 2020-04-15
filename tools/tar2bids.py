import os, subprocess
import shutil
import tarfile
import pydicom
from glob import glob

import re


def extract_dicoms(tar_folder, extract_root):
    tar_files = glob(os.path.join(tar_folder, "*.tar"))
    for tar_file in tar_files:
        tar_obj = tarfile.open(tar_file)
        tar_obj.extractall(extract_root)


def bidsify_string(string_to_bidsify):
    # split by illegal characters
    return_string = re.split(' |_|-', string_to_bidsify)
    # convert to camel case
    return_string = [str(x).capitalize() for x in return_string]
    return_string = ''.join(return_string)
    # replace deicamls with p
    return_string = return_string.replace('.','p')
    return return_string


tar_folder = '/softdev/akuurstr/python/modules/frequency_estimation/scanner_data/tar'
dicom_root = '/softdev/akuurstr/python/modules/frequency_estimation/scanner_data/tar/CFMM'
heuristic = '/softdev/akuurstr/python/modules/mousersfMRIPrep/heudiconv_heuristics/cfmm_base.py'
dcm_dir_template = os.path.join(dicom_root, '*/*/{subject}/{session}.*/*/*.dcm')
bids_output = '/softdev/akuurstr/python/modules/frequency_estimation/scanner_data/bids'

if not os.path.exists(dicom_root):
    extract_dicoms(tar_folder, dicom_root)

# remove underscores from patient names (used for BIDS subjects)
subject_folders = os.path.join(dcm_dir_template.split('{subject}')[0].replace('{session}','*'),'*')
for subject_folder in glob(subject_folders):
    shutil.move(subject_folder,
                os.path.join(os.path.dirname(subject_folder), bidsify_string(os.path.basename(subject_folder))))
# remove underscores from StudyIDs (used for BIDS session)
session_folders = os.path.join(dcm_dir_template.split('{session}')[0].replace('{subject}','*'),'*')
for session_folder in glob(session_folders):
    shutil.move(session_folder,
                os.path.join(os.path.dirname(session_folder), bidsify_string(os.path.basename(session_folder))))

completed_patient_sessions = []
for root, dirs, files in os.walk(dicom_root):
    for file in files:
        if file.endswith(".dcm"):
            dcm_file = pydicom.read_file(os.path.join(root, file), stop_before_pixels=True)
            if 'rsFMRI' in dcm_file.ProtocolName or True:
                bids_subject = bidsify_string(str(dcm_file.PatientName))
                bids_session = bidsify_string(str(dcm_file.StudyID))
                if (bids_subject,bids_session) in completed_patient_sessions:
                    continue
                subprocess.call(
                    ["heudiconv", "-b", "-d", dcm_dir_template, "-o", bids_output, "-f", heuristic, "-s", bids_subject,
                     "-ss", bids_session, "--overwrite"])
                completed_patient_sessions.append((bids_subject,bids_session))

if os.path.exists(os.path.join(bids_output, '.heudiconv')):
    shutil.rmtree(os.path.join(bids_output, '.heudiconv'))

