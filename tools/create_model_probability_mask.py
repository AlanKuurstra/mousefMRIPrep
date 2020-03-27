from bids import BIDSLayout
import os, subprocess
import shutil
from tools.split_exts import split_exts
from workflows.StructuralToAtlasRegistration_wf import get_shrink_factors, get_atlas_smallest_dim_spacing


#not worth putting in a nipype pipeline
ANTSPATH='/usr/lib/ants/'
model_bash_script = os.path.join(os.getcwd(),'cfmm_antsMultivariateTemplateConstruction_modified_for_mice.sh')
empty_model_output_dir = os.path.join(os.getcwd(),'mouse_model')

build_model=False
transform_masks = True
make_average = True
if build_model:
    shutil.rmtree(empty_model_output_dir,ignore_errors=True)
    os.makedirs(empty_model_output_dir)

bids_dir = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids'

layout = BIDSLayout(bids_dir)
derivatives_dir = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/derivatives'
layout.add_derivatives(derivatives_dir)

mask_description = 'brainsuite_brain_mask'
valid_desc = mask_description.split('_')[0]
mouse_masks = layout.get(datatype='anat',suffix='T2w',extension=['.nii','.nii.gz'],desc=valid_desc)
anat_to_mask_mapping = {}

for mouse_mask in mouse_masks:
    mouse_anat_entities = mouse_mask.get_entities()
    mouse_anat_entities['desc'] = None
    mouse_anat = layout.get(**mouse_anat_entities)
    if len(mouse_anat) == 1:
        anat_to_mask_mapping[mouse_anat[0]] = mouse_mask


if build_model:
    #  USE MASKS DURING MODEL CREATION!!!!!!
    # hmm, that doesn't seem to be an option with antsMultivariateTemplateConstruction2
    for bidsfile in anat_to_mask_mapping.keys():
        os.symlink(bidsfile.path,os.path.join(empty_model_output_dir,bidsfile.filename))

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

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, cwd=empty_model_output_dir)
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout.decode(' utf-8'))


#====================================================================================================================
template = os.path.join(empty_model_output_dir,'commontemplate0.nii.gz')
transformed_mask_dir = os.path.join(empty_model_output_dir,'transformed_masks')
if transform_masks:
    if not os.path.exists(transformed_mask_dir):
        os.makedirs(transformed_mask_dir)

    import glob
    for anatbidsfile in anat_to_mask_mapping.keys():
        anat_name, anat_exts = split_exts(anatbidsfile.filename)
        maskbidsfile = anat_to_mask_mapping[anatbidsfile]
        mask_name,mask_exts = split_exts(maskbidsfile.filename)

        linear = glob.glob(os.path.join(empty_model_output_dir,'common*'+anat_name+'*.mat'))[0]
        inverse_warp = glob.glob(os.path.join(empty_model_output_dir, 'common*' + anat_name + '*InverseWarp.nii*'))[0]
        warp = glob.glob(os.path.join(empty_model_output_dir, 'common*' + anat_name + '*Warp.nii*'))
        warp.remove(inverse_warp)
        warp = warp[0]



        out_name = os.path.join(transformed_mask_dir, mask_name+'_to_template'+mask_exts)

        cmd = f'export ANTSPATH={ANTSPATH};export PATH={ANTSPATH}; antsApplyTransforms -d 3 \
        -i {maskbidsfile.path} \
        -r {template} \
        -t {warp} \
        -t {linear} \
        -o {out_name}'

        print(cmd)

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, cwd=empty_model_output_dir)
        proc_stdout = process.communicate()[0].strip()
        print(proc_stdout.decode(' utf-8'))

#====================================================================================================================

template_name,template_exts = split_exts(template)
probability_mask = template_name+'_probability_mask'+template_exts
masks_to_average = os.path.join(transformed_mask_dir,'*.nii*')
if make_average:
    cmd = f'export ANTSPATH={ANTSPATH};export PATH={ANTSPATH}; AverageImages 3 {probability_mask} 0 {masks_to_average}'
    print(cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, cwd=empty_model_output_dir)
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout.decode(' utf-8'))