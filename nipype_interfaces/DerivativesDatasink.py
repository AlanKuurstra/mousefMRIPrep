#https://bids-specification.readthedocs.io/en/derivatives/05-derivatives/01-introduction.html

from workflows.CFMMCommon import get_fn_node

def get_derivatives_entities(original_bids_file, derivatives_description):
    from bids.layout.layout import parse_file_entities
    import os
    from tools.split_exts import split_exts
    original_bids_file = os.path.abspath(original_bids_file)
    original_entities = parse_file_entities(str(original_bids_file))

    # when parse_file_entities config='bids' then entity desc is overlooked but its label is stored as the suffix!
    # this is due to a bad decision for suffix search pattern in:
    # .../lib/python3.6/site-packages/bids/layout/config/bids.json
    # when config='derivatives' only the desc entity is extracted according to:
    # .../lib/python3.6/site-packages/bids/layout/config/derivatives.json
    # when no value is given for config, it searches every possible config (which is just bids and derivatives at this point)
    # but we need to remove the faulty suffix
    # we could make a new config file with a better search for suffix and pass it to parse_file_entities
    # or we can do a quick hack here in the code
    if 'desc' in original_entities.keys() and 'suffix' in original_entities.keys():
        if original_entities['desc'] == original_entities['suffix']:
            del original_entities['suffix']

    # if the file given isn't a bids file, just use the filename as the subject label
    # assume a missing subject means not bids
    if 'subject' not in original_entities.keys():
        filename, _ = split_exts(os.path.basename(original_bids_file))
        # bidsify the filename
        filename = filename.replace('-', '').replace('_', '').replace('.', '')
        original_entities = {'subject': filename}

    original_entities['desc'] = original_entities.setdefault('desc', '') + derivatives_description

    # because we have no idea what processing has happened, we can't be sure of the extension
    if 'extension' in original_entities:
        original_entities.pop('extension')

    return original_entities

def get_derivatives_filename(original_bids_file,derivatives_description):
    from bids.layout.writing import build_path
    from nipype_interfaces.DerivativesDatasink import get_derivatives_entities
    derivatives_entities = get_derivatives_entities(original_bids_file,derivatives_description)

    # this might not have every entity possible
    # the only examples I've found are from the bids config file:
    # .../lib/python3.6/site-packages/bids/layout/config/bids.json (derivatives.json doesn't have any)
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
                     '_desc-{desc}'
                     '[_{suffix}]'
                     '[.{extension<nii|nii.gz|h5|json|png|mat|pkl>}]']
    return build_path(derivatives_entities, path_patterns)


def derivatives_datasink_fn(
        derivatives_files_list,
        derivatives_description_list,
        derivatives_dir,
        pipeline_name,
        original_bids_file=None,
        dataset_description_dict = None,
):
    import os
    import shutil
    from tools.split_exts import split_exts
    from nipype_interfaces.DerivativesDatasink import get_derivatives_filename

    if type(derivatives_files_list) == str:
        derivatives_files_list = [derivatives_files_list]
    if type(derivatives_description_list) == str:
        derivatives_description_list = [derivatives_description_list]
    if derivatives_description_list is None:
        derivatives_description_list = [None]*len(derivatives_files_list)

    if original_bids_file is None:
        original_bids_file = derivatives_files_list[0]

    for derivatives_file,derivatives_description in zip(derivatives_files_list,derivatives_description_list):
        if derivatives_file is None:
            continue

        derivatives_filename = get_derivatives_filename(original_bids_file,derivatives_description)

        # use extension of the file being moved (but ignore BrainSuite's .mask_maths addition)
        _, exts = split_exts(derivatives_file)
        ext_list = exts.split('.')
        ext_remove_list = ['.mask_maths']
        for ext in ext_remove_list:
            if ext in ext_list:
                ext_list.pop(ext)
        exts = '.'.join(ext_list)

        derivatives_filename,_ = split_exts(derivatives_filename)
        derivatives_filename = derivatives_filename+exts

        full_derivatives_file_path = os.path.join(derivatives_dir, pipeline_name, derivatives_filename)
        os.makedirs(os.path.dirname(full_derivatives_file_path), exist_ok=True)

        shutil.copy(derivatives_file,full_derivatives_file_path)

        #layout.add_derivatives(os.path.join(derivatives_dir, pipeline_name),
        # parent_database_path=self.layout_db
        # reset_database)

    if dataset_description_dict is not None:
        # overwrites existing dataset_description
        derivatives_pipeline_dir = os.path.join(derivatives_dir, pipeline_name.split(os.sep)[0])
        import json
        os.makedirs(derivatives_pipeline_dir, exist_ok=True)
        with open(os.path.join(derivatives_pipeline_dir, 'dataset_description.json'), 'w') as fobj:
            json.dump(dataset_description_dict, fobj, indent=4)
    return


def get_node_derivatives_datasink(name='derivatives_datasink', mapnode=False):
    if mapnode:
        iterfield = ['derivatives_files_list', 'original_bids_file']
    else:
        iterfield = None
    # if overwrite=True, the bids derivatives are re-saved and overwritten every single time (no caching).
    # but then if the next workflow depends on the derivatives, the next workflow will always rerun
    return get_fn_node(derivatives_datasink_fn, [], imports=None, name=name, overwrite=False,
                mapnode=mapnode, iterfield=iterfield)

def get_node_get_derivatives_entities(*args, name='get_derivatives_entities', **kwargs):
    return get_fn_node(get_derivatives_entities,['ent_vals'],*args,name=name,**kwargs)



