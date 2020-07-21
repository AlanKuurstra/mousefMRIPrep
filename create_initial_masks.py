#!/usr/bin/env python3
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import os
from bids import BIDSLayout
from workflows.BrainExtraction import init_brainsuite_brain_extraction_wf, BrainExtractMethod
from nipype.interfaces.io import BIDSDataGrabber
from nipype.interfaces.utility import Function
from nipype.interfaces import utility as niu
from workflows.FuncReference import init_func_reference
from argparse import ArgumentParser
from nipype_interfaces.DerivativesDatasink import init_derivatives_datasink
import csv
from bids.layout.layout import parse_file_entities

if __name__=="__main__":
    defstr = ' (default %(default)s)'
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('bids_dir',
                        help='BIDS data directory.')
    parser.add_argument('--derivatives_dir',
                        help='BIDS derivatives directory where initial masks will be saved.')
    parser.add_argument('--bids_datatype',
                        default='anat',
                        help='Either anat or func')
    parser.add_argument('--bids_suffix',
                        default = 'T2w',
                        help='Eg. T2w for anat or bold for func')
    parser.add_argument('--derivatives_description',
                        default='CreateInitialMasksBrainsuiteBrainMask',
                        )
    parser.add_argument('--files',
                        help='File with line separated list of files to brain extract. Overrides BIDS search.'
                        )

    parameters = ['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids2',
                  '--bids_datatype','anat',
                  '--bids_suffix','T2w',
                  ]

    args = parser.parse_args()

    bids_dir = args.bids_dir
    derivatives_dir = args.derivatives_dir
    if derivatives_dir is None:
        derivatives_dir = os.path.join(bids_dir,'derivatives')
    bids_datatype = args.bids_datatype
    bids_suffix = args.bids_suffix
    bids_description = args.derivatives_description
    files = args.files
    derivatives_pipeline_name = 'CreateInitialMasks'


    #should we use mapdir or iterables...both are parallelizable

    # mapdir doesn't work with workflows...only with interfaces
    # so we can use the bidsdatagrabber to get the filenames, and then do a mapnode to identityinterface and connect
    # the identity interface to the workflow.

    # the other option is to just use iterables, then we don't use the bidsdatagrabber...we form the list of images
    # ourselves using the python bids module and and give it as an iterable to the brain extraction workflow

    if files is None:
        layout = BIDSLayout(bids_dir)
        mouse_imgs = layout.get(datatype=bids_datatype, suffix=bids_suffix, extension=['.nii', '.nii.gz'])
    else:
        mouse_imgs = []
        with open(files, 'r', encoding='utf-8-sig') as f:
            csv_file = csv.reader(f, delimiter='\t')
            for line in csv_file:
                mouse_imgs.append(line[0])
            example_entities = parse_file_entities(mouse_imgs[0])
            bids_datatype = example_entities['datatype']
            bids_suffix = example_entities['suffix']

    wf = pe.Workflow(name=derivatives_pipeline_name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']), name='inputnode')
    inputnode.iterables = ("in_file", mouse_imgs)

    brainsuite = init_brainsuite_brain_extraction_wf()

    if bids_datatype == 'anat':
        wf.connect([
            (inputnode, brainsuite, [('in_file', 'inputnode.in_file')]),
        ])
    elif bids_datatype == 'func':
        func_reference = init_func_reference(
            name='func_reference',
            perform_motion_correction=False,
            brain_extract_method=BrainExtractMethod.NO_BRAIN_EXTRACTION,
            nthreads_node=None,
            mem_gb_node=3,
        )
        wf.connect([
            (inputnode, func_reference, [('in_file', 'inputnode.func_file')]),
            (func_reference, brainsuite, [('outputnode.func_avg', 'inputnode.in_file')]),
            ])

    deriviatives_initial_masks = init_derivatives_datasink('deriviatives_initial_masks', bids_datatype=bids_datatype,
                                                              bids_description=bids_description,
                                                              derivatives_collection_dir=derivatives_dir,
                                                              derivatives_pipeline_name=derivatives_pipeline_name)
    wf.connect([
        (inputnode, deriviatives_initial_masks, [('in_file', 'inputnode.original_bids_file')]),
        (brainsuite, deriviatives_initial_masks,
         [('outputnode.out_file_mask', 'inputnode.file_to_rename')]),
    ])

    if bids_datatype == 'func':
        deriviatives_func_avg = init_derivatives_datasink('deriviatives_func_avg',
                                                               bids_datatype=bids_datatype,
                                                               bids_description='FuncAvg',
                                                               derivatives_collection_dir=derivatives_dir,
                                                               derivatives_pipeline_name=derivatives_pipeline_name)
        wf.connect([
            (inputnode, deriviatives_func_avg, [('in_file', 'inputnode.original_bids_file')]),
            (func_reference, deriviatives_func_avg,
             [('outputnode.func_avg', 'inputnode.file_to_rename')]),
        ])

    exec_graph = wf.run()