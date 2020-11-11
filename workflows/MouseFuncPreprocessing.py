from workflows.MouseBrainExtraction import MouseBrainExtraction4D
from workflows.CFMMBIDS import CFMMBIDSWorkflowMixin, CMDLINE_VALUE
from workflows.CFMMWorkflow import CFMMWorkflow
from workflows.CFMMBIDS import BIDSInputExternalSearch
from nipype.pipeline import engine as pe
from workflows.CFMMCommon import NipypeWorkflowArguments, delistify
import nipype.interfaces.afni as afni
from workflows.MouseMotionCorrection import MouseMotionCorrection
from nipype.interfaces import utility as niu
from nipype.interfaces.utility import Function
from workflows.BrainExtraction import CFMMVolumesToAvg
import warnings
with warnings.catch_warnings():
    # niworkflows uses bids.BIDSLayout in a way that is going to be removed in the future
    # UserWarning: The ability to pass arguments to BIDSLayout that control indexing is likely to be removed
    # it's okay for now, so we suppress the warning
    warnings.simplefilter('ignore')
    from niworkflows.interfaces.utils import CopyXForm
from workflows.CFMMFSL import CFMMSpatialSmoothing, CFMMTemporalFilterPhysical
from workflows.CFMMLogging import NipypeLogger as logger


class MouseFuncPreprocessing(CFMMWorkflow):
    group_name = 'Functional preprocessing'
    flag_prefix = 'preproc_'

    def _add_parameters(self):
        self._add_parameter('in_file',
                            help='Explicitly specify location of the input file for functional preprocessing.',
                            )
        self._add_parameter('in_file_mask',
                            help='Explicitly specify location of the input file mask',
                            )
        self._add_parameter('slice_timing',
                            help='Time offsets from the volume acquisition onset for each slice. Used for slice timing correction.',
                            type=eval,
                            )
        self._add_parameter('tr',
                            help='Scan repetition time used in slice timing correction and temporal filtering.',
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
        self._add_parameter('skip_ss',
                            action='store_true',
                            help=f'Do not perform spatial smoothing.',
                            )
        self._add_parameter('skip_tf',
                            action='store_true',
                            help=f'Do not perform temporal filtering.',
                            )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.roi = CFMMVolumesToAvg(owner=self)
        self.mc = MouseMotionCorrection(owner=self, exclude_list=['in_file'], )
        self.be = MouseBrainExtraction4D(owner=self, exclude_list=['in_file', 'in_file_mask'],)
        #stc
        self.ss = CFMMSpatialSmoothing(owner=self, exclude_list=['in_file','USAN_image','USAN_image_mask'])
        self.tf = CFMMTemporalFilterPhysical(owner=self, exclude_list=['in_file','tr'])

        self.outputs = ['preprocessed', 'avg', 'brain_mask', 'preprocessed_before_mc', 'mc_transform']

    def create_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        mc_wf = self.mc.create_workflow()
        mc_placeholder = pe.Node(niu.IdentityInterface(fields=['placeholder']), name='mc_placeholder')
        be = self.be.create_workflow()
        stc = pe.Node(afni.TShift(outputtype='NIFTI_GZ'), name='stc')
        copy_xform = pe.Node(CopyXForm(), name='copy_xform')
        stc_placeholder = pe.Node(niu.IdentityInterface(fields=['placeholder']), name='stc_placeholder')
        spatial_smooth = self.ss.create_workflow()
        spatial_smooth_placeholder = pe.Node(niu.IdentityInterface(fields=['placeholder']), name='spatial_smooth_placeholder')
        temporal_filter = self.tf.create_workflow()
        tf_placeholder = pe.Node(niu.IdentityInterface(fields=['placeholder']), name='tf_placeholder')

        inputnode, outputnode, wf = self.get_io_and_workflow()

        # motion correction
        if not self.get_parameter('skip_mc').user_value:
            wf.connect([
                (inputnode, mc_wf, [('in_file', 'inputnode.in_file')]),
                (mc_wf, mc_placeholder, [('outputnode.motion_corrected_output', 'placeholder')]),
                (mc_wf, outputnode, [('outputnode.motion_correction_transform', 'mc_transform')]),
            ])
        else:
            wf.connect(inputnode, 'in_file', mc_placeholder, 'placeholder')

        # masking - required for susan to find the median value and can also be used during func to anat registration
        wf.connect([
            (mc_wf, be, [('outputnode.motion_corrected_output', 'inputnode.in_file')]),
            (inputnode, be, [('in_file_mask', 'inputnode.in_file_mask')]),
            ])

        # slice timing correction
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

        if not self.get_parameter('skip_ss').user_value:
            wf.connect([
                (stc_placeholder, spatial_smooth, [('placeholder', 'inputnode.in_file')]),
                (be, spatial_smooth, [('outputnode.out_file_n4_corrected', 'inputnode.USAN_image')]),
                (be, spatial_smooth, [('outputnode.out_file_mask', 'inputnode.USAN_image_mask')]),
                (spatial_smooth, spatial_smooth_placeholder, [('outputnode.smoothed_file', 'placeholder')]),
            ])
        else:
            wf.connect(stc_placeholder, 'placeholder', spatial_smooth_placeholder, 'placeholder')

        # temporal filtering
        # cutoff fwhm: 100ms  TR: 1.5ms  so fwhm is 66.6 TRs and sigma should be (since it's hwhm) 33.3 TRs
        if not self.get_parameter('skip_tf').user_value:
            #The 'cutoff' is defined as the FWHM of the filter, so if you ask for 100s that means 50 Trs, so the sigma, or HWHM, is 25 TRs.
            wf.connect([
                (inputnode, temporal_filter, [('tr', 'inputnode.tr')]),
                (spatial_smooth_placeholder,temporal_filter, [('placeholder','inputnode.in_file')]),
                (temporal_filter, tf_placeholder, [('outputnode.out_file', 'placeholder')]),
                ])
        else:
            wf.connect(spatial_smooth_placeholder,'placeholder',tf_placeholder,'placeholder')

        wf.connect([
            # outputnode.preprocessed_before_mc needs to change if additional preprocessing steps are included
            # between inputnode.in_file and mc_wf.in_file
            (inputnode, outputnode, [('in_file', 'preprocessed_before_mc')]),
            (tf_placeholder, outputnode, [('placeholder', 'preprocessed')]),
        ])
        # only connect subworkflow results if they are connected -
        # pass on the non-connected signal for bids derivatives
        if self.be.outputnode_field_connected('out_file_mask'):
            wf.connect([(be, outputnode, [('outputnode.out_file_mask', 'brain_mask')])])
        if self.be.outputnode_field_connected('out_file_n4_corrected'):
            wf.connect([(be, outputnode, [('outputnode.out_file_n4_corrected', 'avg')])])
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


class MouseFuncPreprocessingBIDS(MouseFuncPreprocessing, CFMMBIDSWorkflowMixin):
    def __init__(self, *args, **kwargs):
        self.exclude_parameters(['slice_timing', 'tr', 'slice_encoding_direction', ])
        super().__init__(*args, **kwargs)

        # can this be a function in bids mixin?
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
                                                         dependent_search=self.in_file_bids,

                                              dependent_entities=['subject', 'session', 'run'],
                                              create_base_bids_string=False,
                                              entities_to_overwrite={
                                                  'desc': CMDLINE_VALUE,
                                                  'extension': ['.nii', '.nii.gz'],
                                              },
                                              )
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
        inputnode.inputs.remove_trait('slice_timing') #hack, removes trait from inputnode without node's knowledge
        self.replace_srcnode_connections(inputnode, 'tr', get_tr, 'parameter_value')
        inputnode.inputs.remove_trait('tr')
        self.replace_srcnode_connections(inputnode, 'slice_encoding_direction', get_encode_dir, 'parameter_value')
        inputnode.inputs.remove_trait('slice_encoding_direction')

        return wf


if __name__ == "__main__":
    bids_args = [
        # bidsapp
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids'",
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives'",
        "'participant'",
        '--input_derivatives_dirs',
        "['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives']",
        '--bids_layout_db', "'./func_preprocessing_test/bids_database'",
        # '--reset_db',
        # '--ignore_derivatives_cache',

        '--in_file_base_bids_string', "'task-rs_bold.nii.gz'",
        '--in_file_subject', "'Nl311f9'",
        '--in_file_session', "'2020021001'",
        '--in_file_run', "'02'",
        '--be4d_ants_be_antsarg_float',
        '--be4dbrain_extract_method', 'BRAINSUITE',


        '--smooth_fwhm','1.0',
        '--smooth_dimension', '2',
        '--tf_highpass_fwhm','100',
        '--skip_mc',

        '--nipype_processing_dir', "'./func_preprocessing_test'",
        '--keep_unnecessary_outputs',
    ]

    tmp = MouseFuncPreprocessingBIDS()
    tmp.run_bids(bids_args)
