# apps for container
# tar2bids
# fix orientation and slicetiming

# create initial masks (either brainsuite or with model and probability mask)
# create model and probability mask (either with create brainsuite initial masks or with manual masks).
# downsample atlas to the same resolution as functional

# we should do something different with atlas downsampling if hgihres atlas not same res as highres labels - need to make
# sure the downsampled atlas is shifted the correct way.




from workflows.CFMMWorkflow import CFMMWorkflow
from workflows.MouseFuncToAtlasFull import MouseFuncToAtlas, MouseFuncToAtlasBIDS
from nipype_interfaces.ComputeCorrelationMatrix import CFMMComputeCorrelationMatrix
from nipype_interfaces.ExtractLabels import ExractLabelMeans, get_node_read_label_mapping_file
import nipype.pipeline.engine as pe
from workflows.CFMMLogging import NipypeLogger as logger
from workflows.CFMMBIDS import CFMMBIDSWorkflowMixer
class MouseCorrelationMatrix(CFMMWorkflow):
    group_name = 'Correlation Matrix'
    flag_prefix = 'corr_'

    def _add_parameters(self):
        # how can we get the same help as the children?
        self._add_parameter('func',
                            help='')
        self._add_parameter('func_mask',
                            help='')
        self._add_parameter('anat',
                            help='')
        self._add_parameter('anat_mask',
                            help='')
        self._add_parameter('label_mapping',
                            help='',
                            required=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func2atlas = MouseFuncToAtlas(owner=self, exclude_list=['func', 'func_mask', 'anat', 'anat_mask'])
        self.corr_mtx = CFMMComputeCorrelationMatrix(owner=self)
        self.outputs = ['label_signals_mat',
                        'label_signals_pkl',
                        'corr_mtx_pkl',
                        'corr_mtx_mat',
                        'corr_mtx_png',
                        'corr_mtx_shift_png',
                        ]

    def create_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        func2atlas_wf = self.func2atlas.create_workflow()
        read_label_mapping = get_node_read_label_mapping_file(name='read_label_mapping')
        extract_label_means = pe.Node(interface=ExractLabelMeans(), name='extract_label_means')
        compute_corr_mtx = self.corr_mtx.get_node(name='compute_corr_mtx')
        inputnode, outputnode, wf = self.get_io_and_workflow()

        wf.connect([
            (inputnode, func2atlas_wf, [('func', 'inputnode.func')]),
            (inputnode, func2atlas_wf, [('func_mask', 'inputnode.func_mask')]),
            (inputnode, func2atlas_wf, [('anat', 'inputnode.anat')]),
            (inputnode, func2atlas_wf, [('anat_mask', 'inputnode.anat_mask')]),

            (inputnode, read_label_mapping, [('label_mapping', 'label_mapping_file')]),

            (func2atlas_wf, extract_label_means, [('outputnode.func_to_atlas', 'fmri_volume')]),
            (read_label_mapping, extract_label_means, [('label_mapping', 'label_mapping')]),
            (extract_label_means, outputnode, [('output_file_pkl', 'label_signals_pkl')]),
            (extract_label_means, outputnode, [('output_file_mat', 'label_signals_mat')]),
            (extract_label_means, compute_corr_mtx, [('output_file_pkl', 'label_signals_pkl')]),
            (compute_corr_mtx, outputnode, [('output_file_pkl', 'corr_mtx_pkl')]),
            (compute_corr_mtx, outputnode, [('output_file_mat', 'corr_mtx_mat')]),
            (compute_corr_mtx, outputnode, [('output_file_png', 'corr_mtx_png')]),
            (compute_corr_mtx, outputnode, [('output_file_shift_png', 'corr_mtx_shift_png')]),
        ])

        inputnode, outputnode, wf = self.get_io_and_workflow()

        return wf



from workflows.CFMMCommon import delistify
from workflows.CFMMBIDS import CMDLINE_VALUE, BIDSAppArguments, BIDSInputExternalSearch


class MouseCorrelationMatrixBIDS(MouseCorrelationMatrix,CFMMBIDSWorkflowMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_bids_parameter_group()
        self.bids._modify_parameter('analysis_level', 'choices', ['participant'])

        self._remove_subcomponent('func2atlas')
        self.func2atlas = MouseFuncToAtlasBIDS(owner=self, exclude_list=['func', 'func_mask', 'anat', 'anat_mask'])

        self.func_bids = BIDSInputExternalSearch(self,
                                                 'func',
                                                 entities_to_overwrite={'subject': CMDLINE_VALUE,
                                                                        'session': CMDLINE_VALUE,
                                                                        'run': CMDLINE_VALUE,
                                                                        'extension': ['.nii', '.nii.gz'],
                                                                        },
                                                 output_derivatives={
                                                     'label_signals_mat': 'LabelSignals',
                                                     'label_signals_pkl': 'LabelSignals',
                                                     'corr_mtx_pkl': 'CorrelationMatrix',
                                                     'corr_mtx_mat': 'CorrelationMatrix',
                                                     'corr_mtx_png': 'CorrelationMatrix',
                                                     'corr_mtx_shift_png': 'CorrelationShiftMatrix',
                                                 }
                                                 )

        self.func_mask_bids = BIDSInputExternalSearch(self,
                                                      'func_mask',
                                                      dependent_iterable=self.func_bids,
                                                      dependent_entities=['subject', 'session', 'run'],
                                                      create_base_bids_string=False,
                                                      entities_to_overwrite={
                                                          'desc': CMDLINE_VALUE,
                                                          'extension': ['.nii', '.nii.gz'],
                                                      },
                                                      )
        self._modify_parameter('func_mask_desc', 'default', "'ManualBrainMask'")

        self.anat_bids = BIDSInputExternalSearch(self,
                                                 'anat',
                                                 dependent_iterable=self.func_bids,
                                                 dependent_entities=['subject','session'],
                                                 entities_to_overwrite={
                                                     'run': CMDLINE_VALUE,
                                                     'extension': ['.nii', '.nii.gz'],
                                                     'scope': 'self',
                                                 },
                                                 )

        self.anat_mask_bids = BIDSInputExternalSearch(self,
                                                      'anat_mask',
                                                      dependent_iterable=self.anat_bids,
                                                      dependent_entities=['subject', 'session', 'run'],
                                                      create_base_bids_string=False,
                                                      entities_to_overwrite={
                                                          'desc': CMDLINE_VALUE,
                                                          'extension': ['.nii', '.nii.gz'],
                                                      },
                                                      )
        self._modify_parameter('anat_mask_desc', 'default', "'ManualBrainMask'")


    def create_workflow(self):
        wf = super().create_workflow()
        self.add_bids_to_workflow(wf)

        inputnode = wf.get_node('inputnode')
        wf.connect([
            (inputnode, self.func2atlas.workflow, [
                ('func_original_file','inputnode.func_original_file'),
                ('func_mask_original_file', 'inputnode.func_mask_original_file'),
                ('anat_original_file', 'inputnode.anat_original_file'),
                ('anat_mask_original_file', 'inputnode.anat_mask_original_file'),
            ])
        ])

        return wf


if __name__ == "__main__":
    bids_args = [
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids'",
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives'",
        "'participant'",
        '--input_derivatives_dirs',
        "['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives']",

        '--func_base_bids_string', "'task-rs_bold.nii.gz'",
        '--func_subject', "'Nl311f9'",
        '--func_session', "'2020021001'",
        '--func_run', "'05'",

        '--anat_base_bids_string', "'acq-TurboRARE_T2w.nii.gz'",
        '--anat_run', "'01'",

        '--reg_func_antsarg_float',
        '--reg_func_preproc_be4d_brain_extract_method', 'NO_BRAIN_EXTRACTION',
        '--reg_func_preproc_skip_mc',

        '--reg_anat_antsarg_float',
        '--reg_anat_be_brain_extract_method', 'REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK',
        '--reg_anat_be_ants_be_template',
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/TemplatesAndProbabilityMasks/sub-AnatTemplate_acq-TurboRARE_desc-0p15x0p15x0p55mm20200804_T2w.nii.gz'",
        '--reg_anat_be_ants_be_template_probability_mask',
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/TemplatesAndProbabilityMasks/sub-AnatTemplateProbabilityMask_acq-TurboRARE_desc-0p15x0p15x0p55mm20200804_T2w.nii.gz'",


        '--label_mapping',
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/DownsampleAtlasBIDS/label_mapping_host_0p3x0p3x0p55mm.txt'",
        '--reg_atlas',
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/Atlases/sub-AMBMCc57bl6_desc-ModelDownsampled.nii.gz'",
        '--reg_atlas_mask',
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/Atlases/sub-AMBMCc57bl6_desc-ModelDownsampledBrainMask.nii.gz'",
        '--reg_downsample',

        '--nipype_processing_dir', "'./func_corrmtx_test'",
        '--keep_unnecessary_outputs',
    ]

    tmp = MouseCorrelationMatrixBIDS()
    tmp.run_bids(bids_args)