import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import os
from bids import BIDSLayout
from workflows.BrainExtractionWorkflows import init_brainsuite_brain_extraction_wf
from nipype.interfaces.io import BIDSDataGrabber
from nipype.interfaces.utility import Function
from nipype.interfaces import utility as niu


bids_dir = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids'
derivatives_dir = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/derivatives'

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
mouse_anats = layout.get(datatype='anat',suffix='T2w',extension=['.nii','.nii.gz'])

from bids.layout.layout import parse_file_entities
from bids.layout.writing import build_path
# full_entities = mouse_anats[0].get_entities()

original_bids_file = mouse_anats[0].filename
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

path_patterns=['sub-{subject}[/ses-{session}]/{datatype<anat>|anat}/sub-{subject}[_ses-{session}][_acq-{acquisition}][_run-{run}][_ce-{ceagent}][_rec-{reconstruction}][_desc-{description}]_{suffix<T1w|T2w|T1rho|T1map|T2map|T2star|FLAIR|FLASH|PDmap|PD|PDT2|inplaneT[12]|angio>}.{extension<nii|nii.gz|json>|nii.gz}']
filename_entities['datatype'] = 'anat'
filename_entities['description'] = 'brainsuite_brain_mask'
print(build_path(filename_entities,path_patterns))


wf = init_brainsuite_brain_extraction_wf()
wf_inputnode = wf.get_node('inputnode')
wf_outputnode = wf.get_node('outputnode')

wf_inputnode.iterables = ("in_file",mouse_anats)


def get_bids_derivative_details(original_bids_file,derivative_description_tag,derivative_root_dir_name):
    from bids.layout.layout import parse_file_entities
    from bids.layout.writing import build_path
    from pathlib import Path
    import os
    from tools.split_exts import split_exts
    original_bids_file = Path(original_bids_file).name
    entities = parse_file_entities(os.path.sep+original_bids_file)
    path_patterns = ['sub-{subject}[/ses-{session}]/{datatype<anat>|anat}/sub-{subject}[_ses-{session}][_acq-{acquisition}][_run-{run}][_ce-{ceagent}][_rec-{reconstruction}][_desc-{description}]_{suffix<T1w|T2w|T1rho|T1map|T2map|T2star|FLAIR|FLASH|PDmap|PD|PDT2|inplaneT[12]|angio>}.{extension<nii|nii.gz|json>|nii.gz}']
    entities['datatype'] = 'anat'
    entities['description'] = derivative_description_tag
    relative_path = os.path.join(derivative_root_dir_name,build_path(entities,path_patterns))
    dirname,basename = os.path.split(relative_path)
    return dirname,basename

bids_derivative_details = pe.Node(
    Function(input_names=["original_bids_file", "derivative_description_tag", 'derivative_root_dir_name'], output_names=["derivatives_dir","derivatives_filename"], function=get_bids_derivative_details),
    name="bids_derivative_details")
bids_derivative_details.inputs.derivative_root_dir_name = 'mousersfMRIPrep'
bids_derivative_details.inputs.derivative_description_tag = 'brainsuite_brain_mask'


#we could roll the rename utility and datasink node into our custom derivative node using shutil
rename_file = pe.Node(niu.Rename(format_string="%(new_name)s", keep_ext=False), "rename_file")

datasink = pe.Node(nio.DataSink(), name="datasink")
datasink.inputs.base_directory = derivatives_dir
datasink.inputs.parameterization = False



wf.connect([
(wf_inputnode, bids_derivative_details, [('in_file', 'original_bids_file')]),
(bids_derivative_details, rename_file, [('derivatives_filename', 'new_name')]),
(wf_outputnode, rename_file, [('out_file_mask', 'in_file')]),
(bids_derivative_details, datasink, [('derivatives_dir', 'container')]),
(rename_file, datasink, [('out_file', '@')]),
])

#wf.base_dir = '/storage/akuurstr/mouse_model_mask_pipepline_output'
#wf.config['execution']['remove_unnecessary_outputs'] = False
exec_graph = wf.run()






# only need participants file for it to be it's own freestanding bids dir
# if we add it as a derivatives directory inside of the existing bids dir, then we only need derivative description.json

from pathlib import Path
import json
def write_derivative_description(bids_dir, deriv_dir):
    bids_dir = Path(bids_dir)
    deriv_dir = Path(deriv_dir)
    desc = {
        'Name': 'mousersfMRIPrep - Mouse resting-state fMRI PREProcessing workflow',
        'BIDSVersion': '1.1.1',
        'PipelineDescription': {
            'Name': 'mousersfMRIPrep',
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



write_derivative_description(bids_dir,derivatives_dir)