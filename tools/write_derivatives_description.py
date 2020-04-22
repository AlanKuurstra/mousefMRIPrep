#def write_derivative_description(bids_dir, deriv_dir,derivatives_pipeline_name):
def write_derivative_description(deriv_dir,derivatives_pipeline_name):
    from pathlib import Path
    import json


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

    #get some info from original dataset_desc in bids_dir
    # Keys deriving from source dataset
    # bids_dir = Path(bids_dir)
    # orig_desc = {}
    # fname = bids_dir / 'dataset_description.json'
    # if fname.exists():
    #     with fname.open() as fobj:
    #         orig_desc = json.load(fobj)
    #
    # if 'DatasetDOI' in orig_desc:
    #     desc['SourceDatasetsURLs'] = ['https://doi.org/{}'.format(orig_desc['DatasetDOI'])]
    # if 'License' in orig_desc:
    #     desc['License'] = orig_desc['License']

    deriv_dir = Path(deriv_dir)
    with (deriv_dir / derivatives_pipeline_name / 'dataset_description.json').open('w') as fobj:
        json.dump(desc, fobj, indent=4)