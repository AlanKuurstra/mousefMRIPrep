#!/usr/bin/env python3
from bids import BIDSLayout
import os
import nibabel as nib
import numpy as np
from bids.layout.models import BIDSJSONFile
import json
import tempfile
import pydicom
import tools.jcamp as jc
from tar2bids import get_dicom_root, get_full_search_pattern
from argparse import ArgumentParser
import stat
import shutil
from glob import glob

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

    replace_read_only(tmpfile, nifti_file_location)


def get_slice_timing_in_seconds(dcm_obj):
    param_dict = jc.jcamp_read(dcm_obj[0x0177, 0x1100].value)
    TE = float(dcm_obj[0x5200, 0x9229][0][0x0018, 0x9114][0][0x0018, 0x9082].value)
    TR = float(dcm_obj[0x5200, 0x9229][0][0x0018, 0x9112][0][0x0018, 0x0080].value)
    slice_ordering = np.asarray(param_dict['$PVM_ObjOrderList']['value'])
    slice_timing_ms = slice_ordering * TR / len(slice_ordering) + TE
    return list(slice_timing_ms / 1000)

def replace_read_only(replacement, readonly_file):
    if os.path.exists(readonly_file):
        st = os.stat(readonly_file)
        current = stat.S_IMODE(st.st_mode)
        readonly = not bool(current & stat.S_IWUSR)
        if readonly:
            os.chmod(readonly_file, current | stat.S_IWUSR)
    else:
        readonly = True
        current = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    # os.replace(tmpfile, bids_json_filename)
    shutil.move(replacement, readonly_file)
    if readonly:
        os.chmod(readonly_file, current)


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
    defstr = ' (default %(default)s)'
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('data_folder', help='Original data directory. Either downloaded dicoms or tar folder from cfmm2tar.')
    parser.add_argument('bids_folder',
                        help='BIDS directory associated with data_folder.')
    parser.add_argument('--dcm_dir_template',
                        default='*/*/*/{subject}/{session}.*/*/*.dcm',
                        help='The heudiconv template. Default works with cfmm2tar\'s tar directory. For unzipped '
                             'dicom folder from cfmm\'s dicom server, use */{subject}/{session}.*/*/*.dcm')
    parser.add_argument("--skip_orientation_fix",
                        action='store_true',
                        help="")
    parser.add_argument("--skip_slice_timing_fix",
                        action='store_true',
                        help="")
    parser.add_argument('--anat_suffix',
                        default='T2w',
                        help='')
    parser.add_argument('--func_suffix',
                        default='bold',
                        help='')
    parser.add_argument("--make_data_folder_bids_valid",
                        action='store_true',
                        help="")

    parameters = ['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/intermediate_dicoms',
                  '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids3',
                  ]
    args = parser.parse_args()

    data_folder = args.data_folder
    bids_folder = args.bids_folder
    relative_dcm_dir_template = args.dcm_dir_template
    skip_orientation = args.skip_orientation_fix
    skip_slice_timing = args.skip_slice_timing_fix
    anat_suffix = args.anat_suffix
    func_suffix = args.func_suffix
    make_valid = args.make_data_folder_bids_valid

    dicom_root, _ = get_dicom_root(data_folder, relative_dcm_dir_template, make_valid=make_valid)
    dcm_dir_template = os.path.join(dicom_root, relative_dcm_dir_template)

    layout = BIDSLayout(bids_folder)


    # fix scan orientations
    if not skip_orientation:
        mouse_anats = layout.get(datatype='anat', suffix=anat_suffix, extension=['.nii', '.nii.gz'])
        mouse_funcs = layout.get(datatype='func', suffix=func_suffix, extension=['.nii', '.nii.gz'])
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
            json_dict['QuadrapedOrientationFixed'] = True
            tmpfile = os.path.join(tempfile.gettempdir(), 'orientation_fixed.json')
            with open(tmpfile, 'w') as f:
                json.dump(json_dict, f)
            replace_read_only(tmpfile, associated_json_path)
            fix_quadraped_orientation(mouse_img.path)


    # fix functional json files to include sliceTiming
    if not skip_slice_timing:
        mouse_funcs = layout.get(datatype='func', suffix=func_suffix, extension=['.nii', '.nii.gz'])
        for mouse_func in mouse_funcs:
            bids_entities = mouse_func.get_entities()
            bids_subject = bids_entities['subject']
            bids_session = bids_entities['session']
            bids_file = mouse_func.path
            bids_json_filename = get_associated_json_path(mouse_func)

            with open(bids_json_filename, 'r') as f:
                json_dict = json.load(f)
            if 'SliceTiming' in json_dict:
                print('skipping sliceTiming for {}'.format(os.path.basename(mouse_func)))
                continue

            dcm_file = glob(dcm_dir_template.format(subject=bids_subject, session=bids_session))[0]
            dcm_obj = pydicom.dcmread(dcm_file, stop_before_pixels=True)

            # could also match by json's AcquisitionTime tag to dcm header if SeriesNumber folder structure changes
            print('adding SliceTiming for {}'.format(os.path.basename(bids_file)))
            # json_dict['SliceTiming'] = np.array2string(get_slice_timing_in_seconds(dcm_obj),separator=',', max_line_width=np.inf)
            json_dict['SliceTiming'] = get_slice_timing_in_seconds(dcm_obj)
            # print(json_dict['SliceTiming'])
            tmpfile = os.path.join(tempfile.gettempdir(), 'SliceTiming_added.json')
            with open(tmpfile, 'w') as f:
                json.dump(json_dict, f)
            replace_read_only(tmpfile,bids_json_filename)

