#!/usr/bin/env python3
import os, subprocess
import shutil
import tarfile
import pydicom
from glob import glob
import re
from argparse import ArgumentParser
import heudiconv_heuristics.cfmm_bruker_mouse_heudiconv_heuristic as default_heuristic
import tempfile
import fnmatch
from distutils.dir_util import copy_tree


def extract_dicoms(tar_folder, extract_root):
    tar_files = glob(os.path.join(tar_folder, "*.tar"))
    for tar_file in tar_files:
        tar_obj = tarfile.open(tar_file)
        tar_obj.extractall(extract_root)


def bidsify_string(string_to_bidsify):
    # split by illegal characters
    return_string = re.split(' |_|-', string_to_bidsify)
    # convert to camel case
    if len(return_string) > 1:
        return_string = [str(x).capitalize() for x in return_string]
        return_string = ''.join(return_string)
    else:
        return_string = return_string[0]
    # replace deicamls with p
    return_string = return_string.replace('.', 'p')
    return return_string


def get_full_search_pattern(templated_glob, search_item):
    templated_glob_split = templated_glob.split(search_item)
    search_full_re = fnmatch.translate(
        search_item.join([re.sub('\\{.*\\}', '*', part) for part in templated_glob_split])).replace(
        fnmatch.translate(search_item)[4:-3], '(.+?)')
    return search_full_re


def get_folder_search_patterns(templated_glob, search_item):
    templated_glob_split = templated_glob.split(search_item)
    search_folder_glob = os.path.join(re.sub('\\{.*\\}', '*', templated_glob_split[0]),
                                      search_item + templated_glob_split[1].split(os.path.sep)[0] + os.path.sep)
    search_folder_re = fnmatch.translate(search_folder_glob).replace(fnmatch.translate(search_item)[4:-3], '(.+?)')
    search_folder_glob = search_folder_glob.replace(search_item, '*')
    return search_folder_glob, search_folder_re


def get_dicom_root(data_folder, relative_dcm_dir_template, output_dicom_dir=None, make_valid=True):
    if output_dicom_dir is None:
        dicom_root = tempfile.TemporaryDirectory().name
    else:
        dicom_root = output_dicom_dir

    # doesn't deal with .zip or .tar.gz
    tar_files_exist = False
    dicom_files_exist = False
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".dcm"):
                dicom_files_exist = True
                break
            if file.endswith(".tar"):
                tar_files_exist = True

    data_folder_type = 'unknown'
    if dicom_files_exist:
        # could improve speed by symlinking .dcms instead of copying everything
        # but heudiconv is the bottleneck so not worth the effort
        # shutil.copytree(data_folder, dicom_root)
        if (data_folder != dicom_root) and make_valid:
            copy_tree(data_folder, dicom_root)
        else:
            dicom_root = data_folder
        data_folder_type = 'dcm'
    elif tar_files_exist:
        tar_folder = data_folder
        extract_dicoms(tar_folder, dicom_root)
        data_folder_type = 'tar'

    if make_valid:
        dcm_dir_template = os.path.join(dicom_root, relative_dcm_dir_template)

        # DEPRECATED: DO EVERYTHING IN SESSION FOLDERS (glob for subject folder and subject_re for subject folder string)
        #subject_glob, subject_re = get_folder_search_patterns(dcm_dir_template, '{subject}')

        # glob for session folders and session_re for session folder string
        session_glob, session_re = get_folder_search_patterns(dcm_dir_template, '{session}')

        # NEW WAY: subject_re for session folder string
        tmp = dcm_dir_template.split('{session}')
        session_glob_with_subject = tmp[0] + '*' + tmp[1].split(os.sep)[0] + os.sep
        subject_re = get_full_search_pattern(session_glob_with_subject, '{subject}')


        # DEPRECATED: DO EVERYTHING IN SESSION FOLDERS
        # # remove underscores from patient names (used for BIDS subjects)
        # for subject_folder in glob(subject_glob):
        #     subject_name = re.match(subject_re, subject_folder).group(1)
        #     valid_name = bidsify_string(subject_name)
        #     if subject_name != valid_name:
        #         valid_path = subject_folder.replace(subject_name, valid_name)
        #         if not os.path.exists(valid_path):
        #             shutil.move(subject_folder, valid_path)
        # # remove underscores from StudyIDs (used for BIDS session)
        # for session_folder in glob(session_glob):
        #     print(session_folder)
        #     continue
        #     session_name = re.match(session_re, session_folder).group(1)
        #     valid_name = bidsify_string(session_name)
        #     if session_name != valid_name:
        #         valid_path = session_folder.replace(session_name, valid_name)
        #         shutil.move(session_folder, valid_path)

        # NEW WAY: do everything with session folders
        for session_folder in glob(session_glob):
            subject_name = re.match(subject_re, session_folder).group(1)
            session_name = re.match(session_re, session_folder).group(1)
            valid_subject_name = bidsify_string(subject_name)
            valid_session_name = bidsify_string(session_name)
            if (subject_name != valid_subject_name) or (session_name != valid_session_name):
                valid_path = session_folder.replace(subject_name, valid_subject_name)
                valid_path = valid_path.replace(session_name, valid_session_name)
                print(f"Moving {session_folder} to {valid_path}")
                if os.path.exists(valid_path):
                    print(f'Warning: {valid_path} already exists, skipping.')
                    shutil.rmtree(session_folder)
                    continue
                shutil.move(session_folder, valid_path)
        # clean up empty invalid folders that had all their sessions moved
        subject_glob, _ = get_folder_search_patterns(dcm_dir_template, '{subject}')
        for subject_folder in glob(subject_glob):
            if not os.listdir(subject_folder):
                os.rmdir(subject_folder)
    return dicom_root, data_folder_type


if __name__ == "__main__":
    defstr = ' (default %(default)s)'
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('data_folder', help='Data directory. Either downloaded dicoms or tar folder from cfmm2tar.')
    parser.add_argument('bids_output',
                        help='Directory where processed output files are to be stored in bids format.')
    parser.add_argument('--intermediate_dicom_dir',
                        help='Directory where intermediate dicoms are stored. Defaults to a temporary directory that is deleted.')
    parser.add_argument('--heuristic_file',
                        default=os.path.abspath(default_heuristic.__file__),
                        help='')
    parser.add_argument('--dcm_dir_template',
                        default='*/*/*/{subject}/{session}.*/*/*.dcm',
                        help='The heudiconv template. Default works with cfmm2tar\'s tar directory. For unzipped '
                             'dicom folder from cfmm\'s dicom server, use */{subject}/{session}.*/*/*.dcm')
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='Overwrite bids (session) directories if they already exist.')

    parameters = ['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/cfmm2tar_output2',
                  '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids2',
                  '--intermediate_dicom_dir',
                  '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/intermediate_dicoms',
                  ]
    args = parser.parse_args()

    data_folder = args.data_folder
    bids_output = args.bids_output
    heuristic_file = args.heuristic_file
    relative_dcm_dir_template = args.dcm_dir_template
    intermediate_dicom_dir = args.intermediate_dicom_dir
    overwrite = args.overwrite

    dicom_root, _ = get_dicom_root(data_folder, relative_dcm_dir_template, output_dicom_dir=intermediate_dicom_dir)

    dcm_dir_template = os.path.join(dicom_root, relative_dcm_dir_template)

    completed_patient_sessions = []

    for root, dirs, files in os.walk(dicom_root):
        for file in files:
            if file.endswith(".dcm"):
                dcm_file = os.path.join(root, file)
                #dcm_obj = pydicom.read_file(dcm_file, stop_before_pixels=True)
                #pydicom.filereader.dcmread(dcm_file,specific_tags=[ProtocolName,PatientName,StudyID]) #is this faster?
                if True: #'rsFMRI' in dcm_obj.ProtocolName:
                    # we bidsify the dicom directory now
                    # bids_subject = bidsify_string(str(dcm_obj.PatientName))
                    # bids_session = bidsify_string(str(dcm_obj.StudyID))
                    subject_re = get_full_search_pattern(dcm_dir_template, '{subject}')
                    session_re = get_full_search_pattern(dcm_dir_template, '{session}')
                    bids_subject = re.match(subject_re, dcm_file).group(1)
                    bids_session = re.match(session_re, dcm_file).group(1)
                    if (bids_subject, bids_session) in completed_patient_sessions:
                        continue

                    subprocess.call(
                        ["heudiconv", "-b", "-d", dcm_dir_template, "-o", bids_output, "-f", heuristic_file, "-s",
                         bids_subject, "-ss", bids_session] + ['--overwrite'] * overwrite)

                    completed_patient_sessions.append((bids_subject, bids_session))

    if not os.path.exists(os.path.join(bids_output, 'derivatives')):
        os.makedirs(os.path.join(bids_output, 'derivatives'))
    if os.path.exists(os.path.join(bids_output, '.heudiconv')):
        shutil.rmtree(os.path.join(bids_output, '.heudiconv'))
