#https://bids-specification.readthedocs.io/en/derivatives/05-derivatives/01-introduction.html

import nipype.pipeline.engine as pe
from nipype.interfaces.utility import Function

def derivatives_datasink_fn(
        derivatives_files_list,
        derivatives_description_list,
        derivatives_dir,
        pipeline_name,
        original_bids_file=None,
        dataset_description_dict = None,
):
    from bids.layout.layout import parse_file_entities
    from bids.layout.writing import build_path
    import os
    import shutil
    from tools.split_exts import split_exts

    if type(derivatives_files_list) == str:
        derivatives_files_list = [derivatives_files_list]
    if type(derivatives_description_list) == str:
        derivatives_description_list = [derivatives_description_list]
    if derivatives_description_list is None:
        derivatives_description_list = [None]*len(derivatives_files_list)
    if original_bids_file is None:
        original_bids_file = derivatives_files_list[0]
    original_bids_file = os.path.abspath(original_bids_file)
    original_entities = parse_file_entities(str(original_bids_file))
    # if the file given isn't a bids file, just use the filename as the subject label
    if 'subject' not in original_entities.keys():
        filename, _ = split_exts(os.path.basename(original_bids_file))
        original_entities = {'subject':filename}
    # this might not have every entity possible
    path_patterns = ['sub-{subject}'
                     '[/ses-{session}]'
                     '[/{datatype}]'
                     '/sub-{subject}'
                     '[_ses-{session}]'
                     '[_acq-{acquisition}]'
                     '[_task-{task}]'
                     '[_run-{run}]'
                     '[_ce-{ceagent}]'
                     '[_rec-{reconstruction}]'
                     '_desc-{description}'
                     '[_{suffix}]'
                     '.{extension<nii|nii.gz|json|png|mat|pkl>|nii.gz}']
    for derivatives_file,derivatives_description in zip(derivatives_files_list,derivatives_description_list):
        if derivatives_file is None:
            continue
        original_entities['description'] = derivatives_description
        _, exts = split_exts(derivatives_file)
        # overwrite original extension
        # use extension of the file being moved (but ignore BrainSuite's .mask_maths addition and also strip leading dot)
        original_entities['extension'] = exts.lstrip('.mask_maths').lstrip('.')

        full_derivatives_file_path = os.path.join(derivatives_dir, pipeline_name, build_path(original_entities, path_patterns))
        os.makedirs(os.path.dirname(full_derivatives_file_path), exist_ok=True)
        shutil.copy(derivatives_file,full_derivatives_file_path)

    if dataset_description_dict is not None:
        # overwrites existing dataset_description
        derivatives_pipeline_dir = os.path.join(derivatives_dir, pipeline_name.split(os.sep)[0])
        import json
        os.makedirs(derivatives_pipeline_dir, exist_ok=True)
        with open(os.path.join(derivatives_pipeline_dir, 'dataset_description.json'), 'w') as fobj:
            json.dump(dataset_description_dict, fobj, indent=4)
    return

def get_node_derivatives_datasink(name='derivatives_datasink'):
    node = pe.Node(
        Function(input_names=[
            'derivatives_files_list',
            'derivatives_description_list',
            'derivatives_dir',
            'pipeline_name',
            'original_bids_file',
            'dataset_description_dict',
        ],
             output_names=[],
             function=derivatives_datasink_fn),
        overwrite=True,
        name = name)
    return node

