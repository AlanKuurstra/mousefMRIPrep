#!/usr/bin/env python3
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
from bids.layout.models import BIDSImageFile
import nibabel as nib
import datetime
import csv
from argparse import ArgumentParser
import json
import sys
from tools.write_derivatives_description import write_derivative_description

def get_derivative_basename(original_bids_file):    
    entities = parse_file_entities(os.path.sep + original_bids_file)
    
    #entities['datatype'] = bids_datatype
    entities['session'] = None
    entities['run'] = None
    template_header = nib.load(original_bids_file).header
    
    creation_datetime = datetime.datetime.now().strftime("%Y%m%d")  # H-%M-%S
    res = 'x'.join(template_header['pixdim'][1:4].astype(str)).replace('.', 'p') + template_header.get_xyzt_units()[
        0]
    
    # desc can exist if datatype = func
    if 'desc' not in entities.keys():
        entities['desc'] = ''
    entities['desc'] = f"{entities['desc']}{res}{creation_datetime}"
    
    path_patterns = [
        'sub-{subject}[/ses-{session}]/{datatype<anat|func>}/sub-{subject}[_ses-{session}][_acq-{acquisition}][_task-{task}][_run-{run}][_ce-{ceagent}][_rec-{reconstruction}][_desc-{desc}]_{suffix<T1w|T2w|T1rho|T1map|T2map|T2star|FLAIR|FLASH|PDmap|PD|PDT2|inplaneT[12]|angio|bold>}.{extension<nii|nii.gz|json>|nii.gz}']
    template_basename, _ = split_exts(build_path(entities, path_patterns))
    template_basename = template_basename.replace(f"sub-{entities['subject']}_", '')
    return template_basename

def populate_img_to_mask(mouse_imgs, layout):
    img_to_mask_mapping={}
    for mouse_img in mouse_imgs:
        # mouse_img_entities = mouse_img.get_entities() #only works when layout created the BIDSImageFile
        mouse_img_entities = parse_file_entities(mouse_img)
        mouse_img_entities['desc'] = input_mask_description
        mouse_mask = layout.get(**mouse_img_entities)
        if len(mouse_mask) == 1:
            img_to_mask_mapping[mouse_img] = mouse_mask[0].path
    return img_to_mask_mapping

if __name__=="__main__":
    defstr = ' (default %(default)s)'
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('bids_dir',
                        help='BIDS data directory.')
    parser.add_argument('ants_work_dir',
                        help='Where antsMultivariateTemplateConstruction stores intermediate files.'
                        )
    parser.add_argument('--derivatives_dir',
                        help='BIDS derivatives directory where initial masks will be saved.')
    parser.add_argument('--bids_datatype',
                        default='anat',
                        help='Either anat or func')
    parser.add_argument('--bids_suffix',
                        default = 'T2w',
                        help='Eg. T2w for anat or bold for func')
    parser.add_argument('--input_mask_description',
                        default = 'ManualBrainMask',
                        help='')
    parser.add_argument('--input_files',
                        help='Overrides input_mask_description, bids_datatype, bids_suffix'
                        )
    parser.add_argument('--max_inputs',
                        type=int,
                        )
    parser.add_argument('--output_template_description',
                        help="Default is to create description from template resolution and today's date.")
    parser.add_argument("--skip_template",
                        action='store_true',
                        help="must provide full ants_work_dir")
    parser.add_argument("--skip_probability_mask",
                        action='store_true',
                        help="Don't create a probability mask. See also --include_images_without_mask.")
    parser.add_argument("--include_images_without_mask",
                        action='store_true',
                        help="Don't reject images which lack a matching mask from template creation. Useful when "
                             "creating only a template without a corresponding probability mask.")
    parser.add_argument("--overwrite_ants_work_dir",
                        action='store_true',
                        help="")

    parameters = ['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids',
                  '/storage/akuurstr/mouse_template_intermediate',
                  '--input_mask_description', 'CreateInitialMasksBrainsuiteBrainMask',
                  #'--bids_datatype','func',
                  #'--bids_suffix','bold',
                  #'--input_files','/softdev/akuurstr/python/modules/mousefMRIPrep/examples/subject_masks.txt',
                  '--max_inputs','2',
                  #'--skip_template',
                  '--overwrite_ants_work_dir',
                  ]

    args = parser.parse_args()

    bids_dir = args.bids_dir
    derivatives_dir = args.derivatives_dir
    if derivatives_dir is None:
        derivatives_dir = os.path.join(bids_dir,'derivatives')
    bids_datatype = args.bids_datatype
    bids_suffix = args.bids_suffix
    input_files = args.input_files
    derivatives_pipeline_name = 'TemplatesAndProbabilityMasks'
    skip_template = args.skip_template
    skip_probability_mask = args.skip_probability_mask
    ants_work_dir = args.ants_work_dir
    input_mask_description = args.input_mask_description
    output_template_description = args.output_template_description
    max_imgs = args.max_inputs
    include_images_without_mask = args.include_images_without_mask
    overwrite_ants_work_dir = args.overwrite_ants_work_dir
     

    # not worth putting in a nipype pipeline
    template_bash_script = os.path.join(os.getcwd(), 'tools','cfmm_antsMultivariateTemplateConstruction_modified_for_mice.sh')

    layout = BIDSLayout(bids_dir)
    layout.add_derivatives(derivatives_dir)

    img_to_mask_mapping = {}
    if input_files is None:
        search_description = None
        if bids_datatype == 'func':
            search_description = 'FuncAvg'
        mouse_imgs = layout.get(datatype=bids_datatype, suffix=bids_suffix, extension=['.nii', '.nii.gz'], desc = search_description)
        mouse_imgs = [x.path for x in mouse_imgs]
    else:
        mouse_imgs = []
        with open(input_files, 'r', encoding='utf-8-sig') as f:
            csv_file = csv.reader(f, delimiter='\t')
            for line in csv_file:
                mouse_imgs.append(line[0])
                # if user provides mask locations in input_files.txt then we already have the mapping
                if len(line)>1:
                    img_to_mask_mapping[line[0]]= line[1]
            #example_entities = mouse_imgs[0].get_entities() #only works when BIDSImageFile created by BIDSLayout
            example_entities = parse_file_entities(mouse_imgs[0])
            bids_datatype = example_entities['datatype']
            bids_suffix = example_entities['suffix']



    if not include_images_without_mask:
        mouse_imgs = list(populate_img_to_mask(mouse_imgs, layout).keys())

    if (max_imgs is not None) and (max_imgs<len(mouse_imgs)):
        mouse_imgs = [x for x in random.sample(mouse_imgs, max_imgs)]

    if not skip_template:
        if overwrite_ants_work_dir:
            shutil.rmtree(ants_work_dir)
        if os.path.exists(ants_work_dir):
            if (not os.path.isdir(ants_work_dir)) or os.listdir(ants_work_dir):
                print(f'{ants_work_dir} must be empty directory.')
                sys.exit()
        else:
            os.makedirs(ants_work_dir)

        # make copy of images being used
        with open(os.path.join(ants_work_dir,'imgs_used_for_template.txt'),'w') as f:
            for mouse_img in mouse_imgs:
                f.write(f'{mouse_img}\n')

        #  USE MASKS DURING MODEL CREATION? doesn't seem to be an option with antsMultivariateTemplateConstruction2
        for mouse_img in mouse_imgs:
            os.symlink(mouse_img, os.path.join(ants_work_dir, os.path.basename(mouse_img)))

        smoothing_sigmas_mm = [[0.45, 0.3, 0.15, 0]]
        smallest_dim = get_atlas_smallest_dim_spacing(mouse_img)
        shrink_factors = get_shrink_factors(smoothing_sigmas_mm,smallest_dim)

        smoothing_string = 'x'.join(map(str,smoothing_sigmas_mm[0]))+'mm'
        shrink_string = 'x'.join(map(str,shrink_factors[0]))

        cmd = f'{template_bash_script} -d 3 \
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

        print(f'running command:\n{cmd}')

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, cwd=ants_work_dir)
        proc_stdout = process.communicate()[0].strip()
        print(proc_stdout.decode(' utf-8'))

        # save template and dataset_description
        template = glob.glob(os.path.join(ants_work_dir, 'commontemplate0.nii*'))[0]
        template_basename = get_derivative_basename(mouse_imgs[0])
        _, exts = split_exts(template)
        template_newname = f'{bids_datatype.capitalize()}Template_{template_basename}' + exts        
        if not os.path.exists(os.path.join(derivatives_dir,derivatives_pipeline_name)):
            os.makedirs(os.path.join(derivatives_dir,derivatives_pipeline_name))
        shutil.copy(template, os.path.join(derivatives_dir,derivatives_pipeline_name, template_newname))
        write_derivative_description(derivatives_dir,derivatives_pipeline_name)

    if not skip_probability_mask:
        template = os.path.join(ants_work_dir, 'commontemplate0.nii.gz')
        transformed_mask_dir = os.path.join(ants_work_dir, 'transformed_masks')
        if not os.path.exists(transformed_mask_dir):
            os.makedirs(transformed_mask_dir)

        # open template's original mouse images
        with open(os.path.join(ants_work_dir,'imgs_used_for_template.txt'),'r') as f:
            mouse_imgs = [mouse_img.rstrip() for mouse_img in f.readlines()]

        img_to_mask_mapping = populate_img_to_mask(mouse_imgs,layout)

        for mouse_img in img_to_mask_mapping.keys():
            img_name, img_exts = split_exts(mouse_img)
            mask_img = img_to_mask_mapping[mouse_img]
            mask_name,mask_exts = split_exts(mask_img)

            linear_transform = glob.glob(os.path.join(ants_work_dir, 'common*' + img_name + '*.mat'))
            # only continue if we found a matching transform
            if len(linear_transform) > 0:
                linear_transform = linear_transform[0]
            else:
                continue
            inverse_warp = glob.glob(os.path.join(ants_work_dir, 'common*' + img_name + '*InverseWarp.nii*'))[0]
            warp = glob.glob(os.path.join(ants_work_dir, 'common*' + img_name + '*Warp.nii*'))
            warp.remove(inverse_warp)
            warp = warp[0]

            out_name = os.path.join(transformed_mask_dir, mask_name+'_to_template'+mask_exts)

            cmd = f'antsApplyTransforms -d 3 \
            -i {mask_img} \
            -r {template} \
            -t {warp} \
            -t {linear_transform} \
            -o {out_name}'

            print(f'running command:\n{cmd}')

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, cwd=ants_work_dir)
            proc_stdout = process.communicate()[0].strip()
            print(proc_stdout.decode(' utf-8'))

        # use transformed masks to create probability mask
        template_name,template_exts = split_exts(template)
        probability_mask = os.path.join(ants_work_dir,template_name+'_probability_mask'+template_exts)
        masks_to_average = os.path.join(transformed_mask_dir,'*.nii*')

        cmd = f'AverageImages 3 {probability_mask} 0 {masks_to_average}'
        print(f'running command:\n{cmd}')
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, cwd=ants_work_dir)
        proc_stdout = process.communicate()[0].strip()
        print(proc_stdout.decode(' utf-8'))

        #save probability mask and dataset_description
        original_bids_file =list(img_to_mask_mapping)[0]
        template_basename = get_derivative_basename(original_bids_file)        
        _,exts = split_exts(probability_mask)
        probability_mask_new_name = f'{bids_datatype.capitalize()}TemplateProbabilityMask_{template_basename}'+exts
        if not os.path.exists(os.path.join(derivatives_dir,derivatives_pipeline_name)):
            os.makedirs(os.path.join(derivatives_dir,derivatives_pipeline_name))
        shutil.copy(probability_mask,os.path.join(derivatives_dir,derivatives_pipeline_name,probability_mask_new_name))
        write_derivative_description(derivatives_dir, derivatives_pipeline_name)