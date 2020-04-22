from bids import BIDSLayout
import os, subprocess
import shutil
from tools.split_exts import split_exts
from workflows.RegistrationAnatToAtlas import get_shrink_factors, get_atlas_smallest_dim_spacing
import random
import glob
from bids.layout.writing import build_path
from pathlib import Path
import os
from tools.split_exts import split_exts
from bids.layout.layout import parse_file_entities
import nibabel as nib
import datetime

bids_dir = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids'
derivatives_dir = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/derivatives'
work_dir = '/storage/akuurstr/mouse_model_anat' #mouse_model_func
bids_datatype = 'anat' #func
bids_suffix = 'T2w' #bold

max_imgs = None

build_model=False #None #none means only do it if the directory doesn't exist
transform_individual_masks = True
make_group_probability_mask = True
individual_mask_description = 'BrainsuiteBrainMask' #should default to ManualBrainMask or manualBrainMask


#not worth putting in a nipype pipeline
ANTSPATH='/usr/lib/ants/'
model_bash_script = os.path.join(os.getcwd(),'cfmm_antsMultivariateTemplateConstruction_modified_for_mice.sh')

if build_model == None:
    if not os.path.exists(work_dir):
        build_model=True
    else:
        build_model=False

if build_model:
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir)



layout = BIDSLayout(bids_dir)
layout.add_derivatives(derivatives_dir)

valid_desc = individual_mask_description.split('_')[0]
mouse_masks = layout.get(datatype=bids_datatype,suffix=bids_suffix,extension=['.nii','.nii.gz'],desc=valid_desc)
img_to_mask_mapping = {}
stop
for mouse_mask in mouse_masks:
    mouse_img_entities = mouse_mask.get_entities()
    if bids_datatype == 'anat':
        mouse_img_entities['desc'] = None
    elif bids_datatype == 'func':
        mouse_img_entities['desc'] = 'avg'

    mouse_img = layout.get(**mouse_img_entities)
    if len(mouse_img) == 1:
        img_to_mask_mapping[mouse_img[0]] = mouse_mask

if max_imgs is not None:
    img_to_mask_mapping = {k:v for k,v in random.sample(img_to_mask_mapping.items(),max_imgs)}


if build_model:
    #  USE MASKS DURING MODEL CREATION? doesn't seem to be an option with antsMultivariateTemplateConstruction2
    for bidsfile in img_to_mask_mapping.keys():
        os.symlink(bidsfile.path, os.path.join(work_dir, bidsfile.filename))

    smoothing_sigmas_mm = [[0.45, 0.3, 0.15, 0]]
    smallest_dim = get_atlas_smallest_dim_spacing(bidsfile.path)
    shrink_factors = get_shrink_factors(smoothing_sigmas_mm,smallest_dim)

    smoothing_string = 'x'.join(map(str,smoothing_sigmas_mm[0]))+'mm'
    shrink_string = 'x'.join(map(str,shrink_factors[0]))


    cmd = f'export ANTSPATH={ANTSPATH};export PATH={ANTSPATH}:${{PATH}};{model_bash_script} -d 3 \
    -i 6 \
    -k 1 \
    -t SyN \
    -g 0.1 \
    -q 250x250x100x50 \
    -s {smoothing_string} \
    -f {shrink_string} \
    -m CC[4] \
    -c 0 \
    -o common \
    *.nii*'

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, cwd=work_dir)
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout.decode(' utf-8'))


template = os.path.join(work_dir, 'commontemplate0.nii.gz')
transformed_mask_dir = os.path.join(work_dir, 'transformed_masks')
if transform_individual_masks:
    if not os.path.exists(transformed_mask_dir):
        os.makedirs(transformed_mask_dir)


    for imgbidsfile in img_to_mask_mapping.keys():
        img_name, img_exts = split_exts(imgbidsfile.filename)
        maskbidsfile = img_to_mask_mapping[imgbidsfile]
        mask_name,mask_exts = split_exts(maskbidsfile.filename)

        linear_transform = glob.glob(os.path.join(work_dir, 'common*' + img_name + '*.mat'))
        if len(linear_transform) > 0:
            linear_transform = linear_transform[0]
        else:
            continue
        inverse_warp = glob.glob(os.path.join(work_dir, 'common*' + img_name + '*InverseWarp.nii*'))[0]
        warp = glob.glob(os.path.join(work_dir, 'common*' + img_name + '*Warp.nii*'))
        warp.remove(inverse_warp)
        warp = warp[0]



        out_name = os.path.join(transformed_mask_dir, mask_name+'_to_template'+mask_exts)

        cmd = f'export ANTSPATH={ANTSPATH};export PATH={ANTSPATH}; antsApplyTransforms -d 3 \
        -i {maskbidsfile.path} \
        -r {template} \
        -t {warp} \
        -t {linear_transform} \
        -o {out_name}'

        print(cmd)

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, cwd=work_dir)
        proc_stdout = process.communicate()[0].strip()
        print(proc_stdout.decode(' utf-8'))

#====================================================================================================================

template_name,template_exts = split_exts(template)
probability_mask = os.path.join(work_dir,template_name+'_probability_mask'+template_exts)
masks_to_average = os.path.join(transformed_mask_dir,'*.nii*')
if make_group_probability_mask:
    cmd = f'export ANTSPATH={ANTSPATH};export PATH={ANTSPATH}; AverageImages 3 {probability_mask} 0 {masks_to_average}'
    print(cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, cwd=work_dir)
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout.decode(' utf-8'))


#save template and model in derivatives
original_bids_file =list(img_to_mask_mapping)[0].path
entities = parse_file_entities(os.path.sep+original_bids_file)
path_patterns = ['sub-{subject}[/ses-{session}]/{datatype<anat|func>}/sub-{subject}[_ses-{session}][_acq-{acquisition}][_task-{task}][_run-{run}][_ce-{ceagent}][_rec-{reconstruction}][_desc-{desc}]_{suffix<T1w|T2w|T1rho|T1map|T2map|T2star|FLAIR|FLASH|PDmap|PD|PDT2|inplaneT[12]|angio|bold>}.{extension<nii|nii.gz|json>|nii.gz}']
entities['datatype'] = bids_datatype
entities['session'] = None
entities['run'] = None
template_header = nib.load(original_bids_file).header

creation_datetime = datetime.datetime.now().strftime("%Y%m%d") #H-%M-%S
res = 'x'.join(template_header['pixdim'][1:4].astype(str)).replace('.','p') + template_header.get_xyzt_units()[0]

if 'desc' not in entities.keys():
    entities['desc']=''
entities['desc'] = f"{entities['desc']}{res}{creation_datetime}"

template_basename,_ = split_exts(build_path(entities,path_patterns))
template_basename= template_basename.replace(f"sub-{entities['subject']}_",'')

_,exts = split_exts(template)
template_newname = f'{bids_datatype.capitalize()}Template_{template_basename}'+exts
be_dir = os.path.join(derivatives_dir,'BrainExtractionTemplatesAndProbabilityMasks')
if not os.path.exists(be_dir):
    os.makedirs(be_dir)
shutil.copy(template,os.path.join(be_dir,template_newname))
_,exts = split_exts(probability_mask)
probability_mask_new_name = f'{bids_datatype.capitalize()}TemplateProbabilityMask_{template_basename}'+exts
shutil.copy(probability_mask,os.path.join(be_dir,probability_mask_new_name))

#maybe make json files to tell when the model was made and what files were involved, whether brainsuite or averaging
# was used to make probability...and maybe the resolution&number of mice should be in the model name...and maybe the scan contrast