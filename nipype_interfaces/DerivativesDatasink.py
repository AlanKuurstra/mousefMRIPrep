#https://bids-specification.readthedocs.io/en/derivatives/05-derivatives/01-introduction.html

import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
from nipype.interfaces.utility import Function
from nipype.interfaces import utility as niu
from tools.write_derivatives_description import write_derivative_description

def get_bids_derivative_details(original_bids_file,datatype_tag,derivative_description_tag,derivative_root_dir_name):
    from bids.layout.layout import parse_file_entities
    from bids.layout.writing import build_path
    from pathlib import Path
    import os
    from tools.split_exts import split_exts
    original_bids_file = Path(original_bids_file).name
    entities = parse_file_entities(os.path.sep+original_bids_file)
    #path_patterns = ['sub-{subject}[/ses-{session}]/{datatype<anat|func>}/sub-{subject}[_ses-{session}][_acq-{acquisition}][_task-{task}][_run-{run}][_ce-{ceagent}][_rec-{reconstruction}][_desc-{description}]_{suffix<T1w|T2w|T1rho|T1map|T2map|T2star|FLAIR|FLASH|PDmap|PD|PDT2|inplaneT[12]|angio|bold>}.{extension<nii|nii.gz|json|png|mat|pkl>|nii.gz}']
    path_patterns = [
        'sub-{subject}[/ses-{session}]/{datatype}/sub-{subject}[_ses-{session}][_acq-{acquisition}][_task-{task}][_run-{run}][_ce-{ceagent}][_rec-{reconstruction}][_desc-{description}]_{suffix}.{extension<nii|nii.gz|json|png|mat|pkl>|nii.gz}']
    entities['datatype'] = datatype_tag
    entities['description'] = derivative_description_tag


    relative_path = os.path.join(derivative_root_dir_name,build_path(entities,path_patterns))

    dirname,basename = os.path.split(relative_path)
    basename,_ = split_exts(basename)
    return dirname,basename


def init_derivatives_datasink(name='derivatives_datasink', bids_datatype=None, bids_description=None, derivatives_collection_dir=None, derivatives_pipeline_name=None):
    wf = pe.Workflow(name)

    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'original_bids_file',
        'file_to_rename',
        'bids_datatype',
        'bids_description',
        'derivatives_root_dir',
        'derivatives_pipeline_name',
    ]), name='inputnode')

    #if bids_dir is not None:
    #    inputnode.inputs.bids_dir=bids_dir
    if bids_datatype is not None:
        inputnode.inputs.bids_datatype=bids_datatype
    if bids_description is not None:
        inputnode.inputs.bids_description=bids_description
    if derivatives_collection_dir is not None:
        inputnode.inputs.derivatives_root_dir=derivatives_collection_dir
    if derivatives_pipeline_name is not None:
        inputnode.inputs.derivatives_pipeline_name=derivatives_pipeline_name

    bids_derivative_details = pe.Node(
        Function(input_names=["original_bids_file", "datatype_tag", "derivative_description_tag", 'derivative_root_dir_name'], output_names=["derivatives_dir","derivatives_filename"], function=get_bids_derivative_details),
        name="bids_derivative_details")

    write_deriv_desc = pe.Node(
        Function(#input_names=["bids_dir", "deriv_dir", "derivatives_pipeline_name"],
                 input_names=["deriv_dir", "derivatives_pipeline_name"],
                 function=write_derivative_description), name="write_deriv_desc")


    #we could roll the rename utility and datasink node into our custom derivative node using shutil
    rename = pe.Node(niu.Rename(format_string="%(new_name)s", keep_ext=True), "rename")

    datasink = pe.Node(nio.DataSink(), name="datasink")
    datasink.inputs.parameterization = False


    wf.connect([
        (inputnode, bids_derivative_details, [('original_bids_file', 'original_bids_file')]),
        (inputnode, bids_derivative_details, [('bids_datatype', 'datatype_tag')]),
        (inputnode, bids_derivative_details, [('bids_description', 'derivative_description_tag')]),
        (inputnode, bids_derivative_details, [('derivatives_pipeline_name', 'derivative_root_dir_name')]),

        #(, write_deriv_desc, [('', 'bids_dir')]),
        (inputnode, write_deriv_desc, [('derivatives_root_dir', 'deriv_dir')]),
        (inputnode, write_deriv_desc, [('derivatives_pipeline_name', 'derivatives_pipeline_name')]),


        (bids_derivative_details, rename, [('derivatives_filename', 'new_name')]),
        (inputnode, rename, [('file_to_rename', 'in_file')]),

        (inputnode, datasink, [('derivatives_root_dir', 'base_directory')]),
        (bids_derivative_details, datasink, [('derivatives_dir', 'container')]),
        (rename, datasink, [('out_file', '@')]),
    ])
    return wf

