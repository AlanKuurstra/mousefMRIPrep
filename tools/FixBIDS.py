from bids import BIDSLayout
import os, subprocess
import shutil
import nibabel as nib
import numpy as np
from bids.layout.models import BIDSJSONFile
import json
import tempfile
import tarfile
import pydicom
import tools.jcamp as jc
from glob import glob
import pickle


def fix_quadraped_orientation(nifti_file_location):
    nii_obj = nib.load(nifti_file_location)
    nii_header_obj = nii_obj.header

    rot_about_x = np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    sform_wrong = nii_header_obj.get_sform()
    sform_corrected = np.eye(4)
    sform_corrected[:3, :] = np.dot(rot_about_x, sform_wrong[:3, :])
    # print(sform_corrected)
    nii_obj.set_sform(sform_corrected)

    qform_wrong = nii_header_obj.get_qform()
    qform_corrected = np.eye(4)
    qform_corrected[:3, :] = np.dot(rot_about_x, qform_wrong[:3, :])
    nii_obj.set_qform(qform_corrected)

    tmpfile = os.path.join(tempfile.gettempdir(), 'orientation_fixed.nii.gz')
    nib.save(nii_obj, tmpfile)
    os.replace(tmpfile, nifti_file_location)


def get_slice_timing_in_seconds(dcm_obj):
    param_dict = jc.jcamp_read(dcm_obj[0x0177, 0x1100].value)
    TE = float(dcm_obj[0x5200, 0x9229][0][0x0018, 0x9114][0][0x0018, 0x9082].value)
    TR = float(dcm_obj[0x5200, 0x9229][0][0x0018, 0x9112][0][0x0018, 0x0080].value)
    slice_ordering = np.asarray(param_dict['$PVM_ObjOrderList']['value'])
    slice_timing_ms = slice_ordering * TR / len(slice_ordering) + TE
    return list(slice_timing_ms / 1000)


def get_fmri_runs_from_dicom(dicom_root):
    fmri_runs = {}
    for root, dirs, files in os.walk(dicom_root):
        for file in files:
            if file.endswith(".dcm"):
                dcm_file = pydicom.read_file(os.path.join(root, file), stop_before_pixels=True)
                if 'rsFMRI' in dcm_file.ProtocolName:
                    if str(dcm_file.PatientName) not in fmri_runs:
                        fmri_runs[str(dcm_file.PatientName)] = {}
                    if str(dcm_file.StudyDescription) not in fmri_runs[str(dcm_file.PatientName)]:
                        fmri_runs[str(dcm_file.PatientName)][str(dcm_file.StudyDescription)] = {}
                    if str(dcm_file.StudyID) not in fmri_runs[str(dcm_file.PatientName)][
                        str(dcm_file.StudyDescription)]:  # session
                        fmri_runs[str(dcm_file.PatientName)][str(dcm_file.StudyDescription)][str(dcm_file.StudyID)] = {}
                    if str(dcm_file.SeriesInstanceUID) not in \
                            fmri_runs[str(dcm_file.PatientName)][str(dcm_file.StudyDescription)][str(dcm_file.StudyID)]:
                        series = \
                            fmri_runs[str(dcm_file.PatientName)][str(dcm_file.StudyDescription)][str(dcm_file.StudyID)][
                                str(dcm_file.SeriesInstanceUID)] = []
                        series.append(os.path.join(root, file))
                        series.append(get_slice_timing_in_seconds(dcm_file))
                    else:
                        continue
                    # dcm_file.AcquisitionDateTime
                    # dcm_file.SeriesDescription
                    # dcm_file.ProtocolName
                    # dcm_file.PulseSequenceName
                    # dcm_file.MRAcquisitionType
    return fmri_runs


def extract_dicoms(tar_folder, extract_root):
    tar_files = glob(os.path.join(tar_folder, "*.tar"))
    for tar_file in tar_files:
        tar_obj = tarfile.open(tar_file)
        tar_obj.extractall(extract_root)


def bidsify_string(string_to_bidsify):
    return string_to_bidsify.replace('_', '')


def get_associated_json_path(BIDSImageFileOjb):
    associated_json_file_path = None
    if 0:
        # This method doesn't work for functional scans
        for associated_file in BIDSImageFileOjb.get_associations():
            if type(associated_file) == BIDSJSONFile:
                associated_json_file_path = associated_file.path
    else:
        associated_json_file_path = BIDSImageFileOjb.path.split(".nii")[0] + ".json"
    #what about using layout.get_metadata(BIDSImageFileObj.filename)
    return associated_json_file_path

if __name__=="__main__":
    tar_folder = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/tar'
    dicom_root = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/tar/cfmm2tar_intermediate_dicoms'
    heuristic = '/softdev/akuurstr/python/modules/mouse_resting_state/cfmm_bruker_mouse_heudiconv_heuristic.py'
    dcm_dir_template = os.path.join(dicom_root, '*/*/*/{subject}/{session}.*/*/*.dcm')
    bids_output = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids'

    if not os.path.exists(dicom_root):
        extract_dicoms(tar_folder, dicom_root)

    # remove underscores from patient names (used for BIDS subjects)
    subject_folders = os.path.join(dcm_dir_template.split('{subject}')[0].replace('{session}', '*'), '*')
    for subject_folder in glob(subject_folders):
        shutil.move(subject_folder,
                    os.path.join(os.path.dirname(subject_folder), bidsify_string(os.path.basename(subject_folder))))
    # remove underscores from StudyIDs (used for BIDS session)
    session_folders = os.path.join(dcm_dir_template.split('{session}')[0].replace('{subject}', '*'), '*')
    for session_folder in glob(session_folders):
        shutil.move(session_folder,
                    os.path.join(os.path.dirname(session_folder), bidsify_string(os.path.basename(session_folder))))

    pickle_file = os.path.join(dicom_root, 'fmri_runs.pkl')
    if not os.path.exists(pickle_file):
        fmri_runs = get_fmri_runs_from_dicom(dicom_root)
        with open(pickle_file, 'wb') as f:
            pickle.dump(fmri_runs, f)
    else:
        with open(pickle_file, 'rb') as f:
            fmri_runs = pickle.load(f)

    if not os.path.exists(bids_output):
        for patient in fmri_runs:
            for study in fmri_runs[patient]:
                for session in fmri_runs[patient][study]:
                    bids_subject = bidsify_string(patient)
                    bids_session = bidsify_string(session)
                    subprocess.call(
                        ["heudiconv", "-b", "-d", dcm_dir_template, "-o", bids_output, "-f", heuristic, "-s", bids_subject,
                         "-ss", bids_session, "--overwrite"])
        # ERROR: Embedding failed: The dim must be singular or not exist for the inputs.

    if os.path.exists(os.path.join(bids_output, '.heudiconv')):
        shutil.rmtree(os.path.join(bids_output, '.heudiconv'))

    # fix scan orientations
    bids_dir = bids_output
    layout = BIDSLayout(bids_dir)
    mouse_anats = layout.get(datatype='anat', suffix='T2w', extension=['.nii', '.nii.gz'])
    mouse_funcs = layout.get(datatype='func', suffix='bold', extension=['.nii', '.nii.gz'])
    mouse_imgs = mouse_anats + mouse_funcs

    for mouse_img in mouse_imgs:
        associated_json_path = get_associated_json_path(mouse_img)
        with open(associated_json_path, 'r') as f:
            json_dict = json.load(f)
        if 'QuadrapedOrientationFixed' in json_dict:
            if json_dict['QuadrapedOrientationFixed']:
                print('skipping orientation for {}'.format(mouse_img.path))
                continue
        print('fixing orientation for {}'.format(mouse_img.path))
        fix_quadraped_orientation(mouse_img.path)
        json_dict['QuadrapedOrientationFixed'] = True
        tmpfile = os.path.join(tempfile.gettempdir(), 'orientation_fixed.json')
        with open(tmpfile, 'w') as f:
            json.dump(json_dict, f)
        os.replace(tmpfile, associated_json_path)

    # fix functional json files to include sliceTiming
    # match bids (subject,session) to dicom study folder
    bids_to_dcm_mapping = {}
    for patient in fmri_runs:
        for study in fmri_runs[patient]:
            for session in fmri_runs[patient][study]:
                bids_patient = bidsify_string(patient)
                bids_session = bidsify_string(session)
                dcm_session_folder = \
                    glob(os.path.dirname(
                        os.path.dirname(dcm_dir_template.format(subject=bids_patient, session=bids_session))))[0]
                # print(dcm_session_folder)
                bids_runs = layout.get(datatype='func', suffix='bold', extension=['.nii', '.nii.gz'], subject=bids_patient,
                                       session=bids_session)
                # print(bids_runs)
                for bids_run in bids_runs:
                    bids_json_filename = get_associated_json_path(bids_run)
                    with open(bids_json_filename, 'r') as f:
                        json_dict = json.load(f)
                    if 'SliceTiming' in json_dict:
                        print('skipping sliceTiming for {}'.format(os.path.basename(bids_run)))
                        continue
                    run_dcm_folder = os.path.join(dcm_session_folder, str(json_dict['SeriesNumber']))
                    dcm_file = glob(os.path.join(run_dcm_folder, '*.dcm'))[0]
                    dcm_obj = pydicom.dcmread(dcm_file, stop_before_pixels=True)
                    # could also match by json's AcquisitionTime tag to dcm header if SeriesNumber folder structure changes
                    print('adding SliceTiming for {}'.format(os.path.basename(bids_run)))
                    # json_dict['SliceTiming'] = np.array2string(get_slice_timing_in_seconds(dcm_obj),separator=',', max_line_width=np.inf)
                    json_dict['SliceTiming'] = get_slice_timing_in_seconds(dcm_obj)
                    # print(json_dict['SliceTiming'])
                    tmpfile = os.path.join(tempfile.gettempdir(), 'SliceTiming_added.json')
                    with open(tmpfile, 'w') as f:
                        json.dump(json_dict, f)
                    os.replace(tmpfile, bids_json_filename)
