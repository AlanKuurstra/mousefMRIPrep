import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import os
from bids import BIDSLayout
from workflows.BrainExtraction import init_brainsuite_brain_extraction_wf, BrainExtractMethod
from nipype.interfaces.io import BIDSDataGrabber
from nipype.interfaces.utility import Function
from nipype.interfaces import utility as niu
from workflows.FuncReference import init_bold_reference
niu.IdentityInterface(fields=['in_file', 'in_file_mask', 'template', 'template_probability_mask']),

bids_dir = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids'
derivatives_dir = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/derivatives'
bids_datatype = 'anat' #'bold'
bids_suffix = 'T2w' #'func'

derivatives_pipeline_name = 'MasksForTemplateProbabilityMask'
bids_description ='BrainsuiteBrainMask'

#should we use

# mapdir doesn't work with workflows...only with interfaces
# so we can use the bidsdatagrabber to get the filenames, and then do a mapnode to identityinterface and connect
# the identity interface to the workflow.

# the other option is to just use iterables, then we don't use the bidsdatagrabber...we form the list of images
# ourselves using the python bids module and and give it as an iterable to the brain extraction workflow

# we will need a custom node to create derivative file structure and the bids filename for the masks
# in between custom node that can get the bidsfile and determine derviative file structure

# then a datasink to save it




layout = BIDSLayout(bids_dir)
mouse_imgs = layout.get(datatype=bids_datatype, suffix=bids_suffix, extension=['.nii', '.nii.gz'])

from bids.layout.layout import parse_file_entities
from bids.layout.writing import build_path
# full_entities = mouse_anats[0].get_entities()

original_bids_file = mouse_imgs[0].filename
filename_entities = parse_file_entities(os.path.sep + original_bids_file)


# config = 'bids'#'derivatives' #'bids'
# import json
# from bids.config import get_option
# if isinstance(config, str):
#     config_paths = get_option('config_paths')
#     if config in config_paths:
#         config = config_paths[config]
#     if not os.path.exists(config):
#         raise ValueError("{} is not a valid path.".format(config))
#     else:
#         with open(config, 'r') as f:
#             config = json.load(f)
# default_path_patterns = config['default_path_patterns']

path_patterns=['sub-{subject}[/ses-{session}]/{datatype<anat|func>}/sub-{subject}[_ses-{session}][_acq-{acquisition}][_task-{task}][_run-{run}][_ce-{ceagent}][_rec-{reconstruction}][_desc-{description}]_{suffix<T1w|T2w|T1rho|T1map|T2map|T2star|FLAIR|FLASH|PDmap|PD|PDT2|inplaneT[12]|angio|bold>}.{extension<nii|nii.gz|json>|nii.gz}']

filename_entities['datatype'] = bids_datatype
filename_entities['description'] = bids_description
print(build_path(filename_entities,path_patterns))



wf = pe.Workflow(name='initial_masks_for_template_probability_mask')
inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']), name='inputnode')
inputnode.iterables = ("in_file", mouse_imgs)

brainsuite = init_brainsuite_brain_extraction_wf()
#brainsuite_inputnode = brainsuite.get_node('inputnode')
#brainsuite_outputnode = brainsuite.get_node('outputnode')

if bids_datatype == 'anat':
    wf.connect([
        (inputnode, brainsuite, [('in_file', 'inputnode.in_file')]),
    ])
elif bids_datatype == 'func':
    bold_reference = init_bold_reference(
        name='bold_reference_wf',
        perform_motion_correction=False,
        brain_extract_method=BrainExtractMethod.NO_BRAIN_EXTRACTION,
        omp_nthreads=None,
        mem_gb=50,
    )
    wf.connect([
        (inputnode, bold_reference, [('in_file', 'inputnode.bold_file')]),
        (bold_reference, brainsuite, [('outputnode.bold_avg', 'inputnode.in_file')]),
        ])


def get_bids_derivative_details(original_bids_file,datatype_tag,derivative_description_tag,derivative_root_dir_name):
    from bids.layout.layout import parse_file_entities
    from bids.layout.writing import build_path
    from pathlib import Path
    import os
    from tools.split_exts import split_exts
    original_bids_file = Path(original_bids_file).name
    entities = parse_file_entities(os.path.sep+original_bids_file)
    path_patterns = ['sub-{subject}[/ses-{session}]/{datatype<anat|func>}/sub-{subject}[_ses-{session}][_acq-{acquisition}][_task-{task}][_run-{run}][_ce-{ceagent}][_rec-{reconstruction}][_desc-{description}]_{suffix<T1w|T2w|T1rho|T1map|T2map|T2star|FLAIR|FLASH|PDmap|PD|PDT2|inplaneT[12]|angio|bold>}.{extension<nii|nii.gz|json>|nii.gz}']
    entities['datatype'] = datatype_tag
    entities['description'] = derivative_description_tag
    relative_path = os.path.join(derivative_root_dir_name,build_path(entities,path_patterns))
    dirname,basename = os.path.split(relative_path)
    basename,_ = split_exts(basename)
    return dirname,basename

bids_derivative_details_mask = pe.Node(
    Function(input_names=["original_bids_file", "datatype_tag", "derivative_description_tag", 'derivative_root_dir_name'], output_names=["derivatives_dir","derivatives_filename"], function=get_bids_derivative_details),
    name="bids_derivative_details_mask")
bids_derivative_details_mask.inputs.derivative_root_dir_name = derivatives_pipeline_name
bids_derivative_details_mask.inputs.datatype_tag = bids_datatype
bids_derivative_details_mask.inputs.derivative_description_tag = bids_description


#we could roll the rename utility and datasink node into our custom derivative node using shutil
rename_mask = pe.Node(niu.Rename(format_string="%(new_name)s", keep_ext=True), "rename_mask")

datasink_mask = pe.Node(nio.DataSink(), name="datasink_mask")
datasink_mask.inputs.base_directory = derivatives_dir
datasink_mask.inputs.parameterization = False


bids_derivative_details_avg = pe.Node(
    Function(input_names=["original_bids_file", "datatype_tag", "derivative_description_tag", 'derivative_root_dir_name'], output_names=["derivatives_dir","derivatives_filename"], function=get_bids_derivative_details),
    name="bids_derivative_details_avg")
bids_derivative_details_avg.inputs.derivative_root_dir_name = derivatives_pipeline_name
bids_derivative_details_avg.inputs.datatype_tag = bids_datatype
bids_derivative_details_avg.inputs.derivative_description_tag = 'avg'


#we could roll the rename utility and datasink node into our own custom derivative node using shutil
rename_avg = pe.Node(niu.Rename(format_string="%(new_name)s", keep_ext=True), "rename_avg")

datasink_avg = pe.Node(nio.DataSink(), name="datasink_avg")
datasink_avg.inputs.base_directory = derivatives_dir
datasink_avg.inputs.parameterization = False

wf.connect([
(inputnode, bids_derivative_details_mask, [('in_file', 'original_bids_file')]),
(bids_derivative_details_mask, rename_mask, [('derivatives_filename', 'new_name')]),
(brainsuite, rename_mask, [('outputnode.out_file_mask', 'in_file')]),
(bids_derivative_details_mask, datasink_mask, [('derivatives_dir', 'container')]),
(rename_mask, datasink_mask, [('out_file', '@')]),
])

if bids_datatype == 'func':
    wf.connect([
        (inputnode, bids_derivative_details_avg, [('in_file', 'original_bids_file')]),
        (bids_derivative_details_avg, rename_avg, [('derivatives_filename', 'new_name')]),
        (bold_reference, rename_avg, [('outputnode.bold_avg', 'in_file')]),
        (bids_derivative_details_avg, datasink_avg, [('derivatives_dir', 'container')]),
        (rename_avg, datasink_avg, [('out_file', '@')]),
    ])

wf.base_dir = '/storage/akuurstr/mouse_model_mask_pipepline_output'
wf.config['execution']['remove_unnecessary_outputs'] = False
exec_graph = wf.run()






# only need participants file for it to be it's own freestanding bids dir
# if we add it as a derivatives directory inside of the existing bids dir, then we only need derivative description.json


from pathlib import Path
import json
def write_derivative_description(bids_dir, deriv_dir):
    bids_dir = Path(bids_dir)
    deriv_dir = Path(deriv_dir)
    desc = {
        'Name': f'{derivatives_pipeline_name} - initial mouse masks for creating a template and probability mask',
        'BIDSVersion': '1.1.1',
        'PipelineDescription': {
            'Name': f'{derivatives_pipeline_name}',
            'Version': '0.1',
            'CodeURL': 'unknown',
        },
        #'CodeURL': __url__,
        #'HowToAcknowledge':
        #    'Please cite our paper (https://doi.org), '
        #    'and include the generated citation boilerplate within the Methods '
        #    'section of the text.',
    }

    # Keys deriving from source dataset
    orig_desc = {}
    fname = bids_dir / 'dataset_description.json'
    if fname.exists():
        with fname.open() as fobj:
            orig_desc = json.load(fobj)

    if 'DatasetDOI' in orig_desc:
        desc['SourceDatasetsURLs'] = ['https://doi.org/{}'.format(
            orig_desc['DatasetDOI'])]
    if 'License' in orig_desc:
        desc['License'] = orig_desc['License']

    with (deriv_dir / 'dataset_description.json').open('w') as fobj:
        json.dump(desc, fobj, indent=4)



#
# from pathlib import Path
# import json
# def write_derivative_description(bids_dir, deriv_dir):
#     bids_dir = Path(bids_dir)
#     deriv_dir = Path(deriv_dir)
#     desc = {
#         f'Name': 'mousersfMRIPrep - mouse resting-state fMRI PREProcessing workflow',
#         'BIDSVersion': '1.1.1',
#         'PipelineDescription': {
#             'Name': 'mousersfMRIPrep',
#             'Version': '0.1',
#             'CodeURL': 'unknown',
#         },
#         #'CodeURL': __url__,
#         #'HowToAcknowledge':
#         #    'Please cite our paper (https://doi.org), '
#         #    'and include the generated citation boilerplate within the Methods '
#         #    'section of the text.',
#     }
#
#     # Keys deriving from source dataset
#     orig_desc = {}
#     fname = bids_dir / 'dataset_description.json'
#     if fname.exists():
#         with fname.open() as fobj:
#             orig_desc = json.load(fobj)
#
#     if 'DatasetDOI' in orig_desc:
#         desc['SourceDatasetsURLs'] = ['https://doi.org/{}'.format(
#             orig_desc['DatasetDOI'])]
#     if 'License' in orig_desc:
#         desc['License'] = orig_desc['License']
#
#     with (deriv_dir / 'dataset_description.json').open('w') as fobj:
#         json.dump(desc, fobj, indent=4)



write_derivative_description(bids_dir,derivatives_dir)