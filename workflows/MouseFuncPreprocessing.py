from workflows.MouseBrainExtraction import MouseBrainExtraction4D
from workflows.CFMMBIDS import CFMMBIDSWorkflowMixer, CMDLINE_VALUE
from workflows.CFMMWorkflow import CFMMWorkflow
from workflows.CFMMBIDS import BIDSInputExternalSearch
from nipype.pipeline import engine as pe
from workflows.CFMMCommon import NipypeWorkflowArguments, delistify
import nipype.interfaces.afni as afni
from workflows.MouseMotionCorrection import MouseMotionCorrection
from nipype.interfaces import utility as niu
from nipype.interfaces.utility import Function
from workflows.BrainExtraction import CFMMVolumesToAvg
from niworkflows.interfaces.utils import CopyXForm
from workflows.CFMMLogging import NipypeLogger as logger


class MouseFuncPreprocessing(CFMMWorkflow):
    group_name = 'Functional preprocessing'
    flag_prefix = 'preproc_'

    def _add_parameters(self):
        self._add_parameter('in_file',
                            help='Explicitly specify location of the input file for functional preprocessing.',
                            )
        self._add_parameter('in_file_mask',
                            help='',
                            )
        self._add_parameter('slice_timing',
                            help='',
                            type=eval,
                            )
        self._add_parameter('tr',
                            help='',
                            )
        self._add_parameter('slice_encoding_direction',
                            help='',
                            default='k',
                            )
        self._add_parameter('skip_mc',
                            action='store_true',
                            help=f'Do not perform motion correction.',
                            )
        self._add_parameter('skip_stc',
                            action='store_true',
                            help=f'Do not perform slice timing correction.',
                            )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nipype = NipypeWorkflowArguments(owner=self, exclude_list=['nthreads_mapnode', 'mem_gb_mapnode'])
        self.roi = CFMMVolumesToAvg(owner=self)
        self.mc = MouseMotionCorrection(owner=self, exclude_list=['in_file'], )
        self.be = MouseBrainExtraction4D(owner=self, exclude_list=['in_file', 'in_file_mask'])
        self.outputs = ['preprocessed', 'avg', 'brain_mask', 'preprocessed_before_mc', 'mc_transform']

    def create_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        omp_nthreads = self.nipype.get_parameter('nthreads_node').user_value

        # !!!??? more preprocessing steps ???!!!
        # melodic high-pass or bandpass filtering
        # in-plane smoothing?
        # ica denoising? FSL's FIX program?
        # despiking?
        # remove global signal?
        # regressors? (white matter, ventricle, vascular)
        # remove motion outliers?

        mc_wf = self.mc.create_workflow()
        mc_placeholder = pe.Node(niu.IdentityInterface(fields=['placeholder']), name='mc_placeholder')

        stc = pe.Node(afni.TShift(outputtype='NIFTI_GZ'), name='stc')
        copy_xform = pe.Node(CopyXForm(), name='copy_xform')
        stc_placeholder = pe.Node(niu.IdentityInterface(fields=['placeholder']), name='stc_placeholder')

        be_wf = self.be.create_workflow()

        inputnode, outputnode, wf = self.get_io_and_workflow()

        if not self.get_parameter('skip_mc').user_value:
            wf.connect([
                (inputnode, mc_wf, [('in_file', 'inputnode.in_file')]),
                (mc_wf, mc_placeholder, [('outputnode.motion_corrected_output', 'placeholder')]),
                (mc_wf, outputnode, [('outputnode.motion_correction_transform', 'mc_transform')]),
            ])
        else:
            wf.connect(inputnode, 'in_file', mc_placeholder, 'placeholder')

        if not self.get_parameter('skip_stc').user_value:
            wf.connect([
                (mc_placeholder, stc, [('placeholder', 'in_file')]),
                # ( , stc, [('', 'ignore')]), #any volumes that don't need stc??
                (inputnode, stc, [('slice_timing', 'slice_timing')]),
                (inputnode, stc, [('tr', 'tr')]),
                (inputnode, stc, [('slice_encoding_direction', 'slice_encoding_direction')]),
                (stc, copy_xform, [('out_file', 'in_file')]),
                (inputnode, copy_xform, [('in_file', 'hdr_file')]),
                (copy_xform, stc_placeholder, [('out_file', 'placeholder')]),
            ])
        else:
            wf.connect(mc_placeholder, 'placeholder', stc_placeholder, 'placeholder')

        wf.connect([
            # outputnode.preprocessed_before_mc needs to change if additional preprocessing steps are included
            # between inputnode.in_file and mc_wf.in_file
            (inputnode, outputnode, [('in_file', 'preprocessed_before_mc')]),
            (stc_placeholder, outputnode, [('placeholder', 'preprocessed')]),
            (stc_placeholder, be_wf, [('placeholder', 'inputnode.in_file')]),
            (inputnode, be_wf, [('in_file_mask', 'inputnode.in_file_mask')]),
            (be_wf, outputnode, [('outputnode.out_file_n4_corrected', 'avg')]),
            (be_wf, outputnode, [('outputnode.out_file_mask', 'brain_mask')]),
        ])
        return wf


def get_param_from_json_fn(filename, parameter, type_conversion=None, default_value=None):
    # A value to return if the specified key does not exist.
    import json
    from workflows.CFMMLogging import NipypeLogger as logger
    if type(filename) == list:
        filename = filename[0]
    with open(filename) as f:
        metadata = json.load(f)
        parameter_value = metadata.get(parameter)
        if not bool(parameter_value):
            if default_value:
                parameter_value = default_value
            else:
                logger.error(f'{parameter} is missing from {filename}')
        elif type_conversion:
            parameter_value = type_conversion(parameter_value)
    f.close()
    return parameter_value


def get_param_from_json_node(*args, **kwargs):
    return pe.Node(*args, interface=Function(input_names=["filename", "parameter", "type_conversion", "default_value"],
                                             output_names=["parameter_value"],
                                             function=get_param_from_json_fn), **kwargs)


def get_sidecar(bids_img_file):
    import pathlib
    bids_img_file = pathlib.Path(bids_img_file)
    return str(bids_img_file.absolute()).split(".nii")[0] + ".json"


def get_sidecar_node(*args, **kwargs):
    return pe.Node(*args, interface=Function(input_names=["bids_img_file"],
                                             output_names=["bids_sidecar"],
                                             function=get_sidecar), **kwargs)


class MouseFuncPreprocessingBIDS(MouseFuncPreprocessing, CFMMBIDSWorkflowMixer):
    def __init__(self, *args, **kwargs):
        self.exclude_parameters(['slice_timing', 'tr', 'slice_encoding_direction', ])
        super().__init__(*args, **kwargs)

        # can this be a function in bids mixer?
        self.add_bids_parameter_group()
        self.bids._modify_parameter('analysis_level', 'choices', ['participant'])

        self.in_file_bids = BIDSInputExternalSearch(self,
                                         'in_file',
                                         entities_to_overwrite={'subject': CMDLINE_VALUE,
                                                                'session': CMDLINE_VALUE,
                                                                'run': CMDLINE_VALUE,
                                                                'extension': ['.nii', '.nii.gz'],
                                                                },
                                         output_derivatives={
                                             'preprocessed': 'Preproc',
                                             'avg': 'Avg',
                                             'brain_mask': 'BrainMask',
                                             'mc_transform': 'MCTransform',
                                         })
        self.in_file_mask_bids = BIDSInputExternalSearch(self,
                                              'in_file_mask',
                                              dependent_entities=['subject', 'session', 'run'],
                                              create_base_bids_string=False,
                                              entities_to_overwrite={
                                                  'desc': CMDLINE_VALUE,
                                                  'extension': ['.nii', '.nii.gz'],
                                              },
                                              )
        # should the exclude check happen in _modify_parameters??
        # doing that would hide potential mistakes from the user. if they make a typo in the parameter, it would just
        # be ignored - the modification to the parameter would not be made and the user might not notice
        if 'in_file_mask' not in self.exclude_list:
            self._modify_parameter('in_file_mask_desc', 'default', "'ManualBrainMask'")

    def create_workflow(self):

        wf = super().create_workflow()
        self.add_bids_to_workflow(wf)
        inputnode = self.inputnode

        # bids related operations
        get_sidecar = get_sidecar_node(name='get_sidecar')
        get_slice_timing = get_param_from_json_node(name='get_slice_timing')
        get_slice_timing.inputs.parameter = 'SliceTiming'
        get_tr = get_param_from_json_node(name='get_tr')
        get_tr.inputs.parameter = 'RepetitionTime'
        get_tr.inputs.type_conversion = str
        get_encode_dir = get_param_from_json_node(name='get_encode_dir')
        get_encode_dir.inputs.parameter = 'SliceEncodingDirection'
        get_encode_dir.inputs.default_value = 'k'
        if not self.get_parameter('skip_stc').user_value:
            wf.connect([
                (inputnode, get_sidecar, [('in_file_original_file', 'bids_img_file')]),
                (get_sidecar, get_slice_timing, [('bids_sidecar', 'filename')]),
                (get_sidecar, get_tr, [('bids_sidecar', 'filename')]),
                (get_sidecar, get_encode_dir, [('bids_sidecar', 'filename')]),
            ])

        self.replace_srcnode_connections(inputnode, 'slice_timing', get_slice_timing, 'parameter_value')
        inputnode.inputs.remove_trait('slice_timing')
        self.replace_srcnode_connections(inputnode, 'tr', get_tr, 'parameter_value')
        inputnode.inputs.remove_trait('tr')
        self.replace_srcnode_connections(inputnode, 'slice_encoding_direction', get_encode_dir, 'parameter_value')
        inputnode.inputs.remove_trait('slice_encoding_direction')

        return wf


if __name__ == "__main__":
    cmd_args = [
        # bidsapp
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids'",
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives'",
        "'participant'",
        '--input_derivatives_dirs',
        "['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives']",
        '--bids_layout_db', './func_preprocessing_test/bids_database',
        '--in_file_base_bids_string', 'task-rs_bold.nii.gz',
        '--in_file_subject', "'Nl311f9'",
        '--in_file_session', "'2020021001'",
        # '--run_labels', '01',
        '--be4d_ants_be_antsarg_float',
        '--be4d_brain_extract_method', 'BRAINSUITE',
        '--nipype_processing_dir', './func_preprocessing_test',
        '--keep_unnecessary_outputs',
        '--skip_mc',
    ]

    tmp = MouseFuncPreprocessingBIDS()
    tmp.run(cmd_args)
