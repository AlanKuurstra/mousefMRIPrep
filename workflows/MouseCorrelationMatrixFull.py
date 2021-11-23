from cfmm.workflow import Workflow
from workflows.MouseFuncToAtlasFull import MouseFuncToAtlas, MouseFuncToAtlasBIDS
from nipype_interfaces.ComputeCorrelationMatrix import CFMMComputeCorrelationMatrix
from nipype_interfaces.ExtractLabels import ExractLabelMeans, get_node_read_label_mapping_file
import nipype.pipeline.engine as pe
from cfmm.bids_parameters import BIDSWorkflowMixin
class MouseCorrelationMatrix(Workflow):
    group_name = 'Correlation Matrix'
    flag_prefix = 'corr_'

    def _add_parameters(self):
        # how can we get the same help as the children?
        self._add_parameter('func',
                            help='Explicitly specify location of the input functional for correlation matrix processing.',
                            iterable=True)
        self._add_parameter('func_mask',
                            help='Explicitly specify location of the input functional mask for atlas registration.',
                            iterable=True)
        self._add_parameter('anat',
                            help='Explicitly specify location of the anatomical image used for intermediate registration.',
                            iterable=True)
        self._add_parameter('anat_mask',
                            help='Explicitly specify location of the anatomical mask used for intermediate registration.',
                            iterable=True)
        self._add_parameter('label_mapping',
                            help='Location of text file mapping label names to integer value and label image.',
                            required=True,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func2atlas = MouseFuncToAtlas(owner=self, exclude_list=['func', 'func_mask', 'anat', 'anat_mask'])
        self.corr_mtx = CFMMComputeCorrelationMatrix(owner=self)
        # should probably add outputs from anat
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
            self.populate_user_value(arg_dict)
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


from cfmm.bids_parameters import CMDLINE_VALUE, BIDSInputExternalSearch


class MouseCorrelationMatrixBIDS(MouseCorrelationMatrix,BIDSWorkflowMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_bids_parameter_group()
        self.bids._modify_parameter('analysis_level', 'choices', ['participant'])

        self._remove_subcomponent_attribute('func2atlas')
        self.func2atlas = MouseFuncToAtlasBIDS(owner=self, exclude_list=['func', 'func_mask', 'anat', 'anat_mask'])

        self.func_bids = BIDSInputExternalSearch(self,
                                                 'func',
                                                 entities_to_overwrite={'subject': CMDLINE_VALUE,
                                                                        'session': CMDLINE_VALUE,
                                                                        'run': CMDLINE_VALUE,
                                                                        'extension': ['.nii', '.nii.gz'],
                                                                        },
                                                 output_derivatives={
                                                     'label_signals_mat': 'LabelSignalsMat',
                                                     'label_signals_pkl': 'LabelSignalsPkl',
                                                     'corr_mtx_pkl': 'CorrelationMatrixPkl',
                                                     'corr_mtx_mat': 'CorrelationMatrixMat',
                                                     'corr_mtx_png': 'CorrelationMatrixPng',
                                                     'corr_mtx_shift_png': 'CorrelationShiftMatrix',
                                                 }
                                                 )

        self.func_mask_bids = BIDSInputExternalSearch(self,
                                                      'func_mask',
                                                      dependent_search=self.func_bids,
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
                                                 dependent_search=self.func_bids,
                                                 dependent_entities=['subject','session'],
                                                 entities_to_overwrite={
                                                     'run': CMDLINE_VALUE,
                                                     'extension': ['.nii', '.nii.gz'],
                                                     'scope': 'self',
                                                 },
                                                 )

        self.anat_mask_bids = BIDSInputExternalSearch(self,
                                                      'anat_mask',
                                                      dependent_search=self.anat_bids,
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
        # bids stuff
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids'",
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives'",
        "'participant'",
        '--input_derivatives_dirs',"['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives']",

        # find functional
        '--func_base_bids_string', "'task-rs_bold.nii.gz'",
        '--func_subject', "'Nl311f9'",
        '--func_session', "'2020021001'",
        '--func_run', "'05'",

        # find anatomical
        '--anat_base_bids_string', "'acq-TurboRARE_T2w.nii.gz'",
        '--anat_run', "'01'",

        # functional preprocessing
        '--reg_func_antsarg_float',
        '--reg_func_preproc_be4d_brain_extract_method', 'BRAINSUITE',
        '--reg_func_preproc_smooth_fwhm', '0.5',
        '--reg_func_preproc_tf_highpass_fwhm', '100',

        # anatomical preprocessing
        '--reg_anat_antsarg_float',
        '--reg_anat_be_brain_extract_method', 'REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK',
        '--reg_anat_be_ants_be_template',"'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/TemplatesAndProbabilityMasks/sub-AnatTemplate_acq-TurboRARE_desc-0p15x0p15x0p55mm20200804_T2w.nii.gz'",
        '--reg_anat_be_ants_be_template_probability_mask',"'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/TemplatesAndProbabilityMasks/sub-AnatTemplateProbabilityMask_acq-TurboRARE_desc-0p15x0p15x0p55mm20200804_T2w.nii.gz'",

        # registration atlas, high res and downsampled
        '--reg_atlas',"'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/Atlases/sub-AMBMCc57bl6_desc-ModelHalfRes.nii.gz'",
        '--reg_atlas_mask',"'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/Atlases/sub-AMBMCc57bl6_desc-ModelHalfResBrainMask.nii.gz'",
        '--reg_downsample',
        '--reg_downsampled_atlas',"'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/DownsampleAtlasBIDS/sub-AMBMCc57bl6/sub-AMBMCc57bl6_desc-ModelBinDownsampled.nii.gz'",
        '--reg_downsample_shift_transformation',"'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/DownsampleAtlasBIDS/sub-AMBMCc57bl6/sub-AMBMCc57bl6_desc-ModelBinDownsampleShift.mat'",

        # correlation matrix options
        '--label_mapping',
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/DownsampleAtlasBIDS/label_mapping_host.txt'",

        # nipype stuff
        '--nipype_processing_dir', "'/storage/akuurstr/Esmin_mouse_registration/test_full_pipeline2/'",
        '--plugin',"'MultiProc'",
        '--plugin_args',"{'n_procs' : 16, 'memory_gb' : 50}",

        '--write_config_file'
    ]

    config_args = [
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids'",
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives'",
        "'participant'",
        '--config_file',"'/softdev/akuurstr/python/modules/mousefMRIPrep/workflows/config.txt'",
    ]

    tmp = MouseCorrelationMatrixBIDS()
    tmp.run_bids(config_args)