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
            (extract_label_means, compute_corr_mtx, [('output_file_pkl', 'label_signals_pkl')]),
        ])

        inputnode, outputnode, wf = self.get_io_and_workflow()

        return wf



from workflows.CFMMCommon import delistify
from workflows.CFMMBIDS import CMDLINE_VALUE, BIDSAppArguments, BIDSIterable


class MouseCorrelationMatrixBIDS(MouseCorrelationMatrix,CFMMBIDSWorkflowMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_bids_parameter_group()
        self.bids._modify_parameter('analysis_level', 'choices', ['participant'])

        self._remove_subcomponent('func2atlas')
        self.func2atlas = MouseFuncToAtlasBIDS(owner=self, exclude_list=['func', 'func_mask', 'anat', 'anat_mask'])

        self.func_bids = BIDSIterable(self,
                                      'func',
                                      entities_to_overwrite={'subject': CMDLINE_VALUE,
                                                             'session': CMDLINE_VALUE,
                                                             'run': CMDLINE_VALUE,
                                                             'extension': ['.nii', '.nii.gz'],
                                                             },
                                      )

        self.func_mask_bids = BIDSIterable(self,
                                           'func_mask',
                                           dependent_entities=['subject', 'session', 'run'],
                                           create_base_bids_string=False,
                                           entities_to_overwrite={
                                               'desc': CMDLINE_VALUE,
                                               'extension': ['.nii', '.nii.gz'],
                                           },
                                           )

        self.anat_bids = BIDSIterable(self,
                                      'anat',
                                      dependent_entities=['subject'],
                                      entities_to_overwrite={
                                          'session': CMDLINE_VALUE,
                                          'run': CMDLINE_VALUE,
                                          'extension': ['.nii', '.nii.gz'],
                                          'scope': 'self',
                                      },
                                      )

        self.anat_mask_bids = BIDSIterable(self,
                                           'anat_mask',
                                           dependent_entities=['subject', 'session', 'run'],
                                           create_base_bids_string=False,
                                           entities_to_overwrite={
                                               'desc': CMDLINE_VALUE,
                                               'extension': ['.nii', '.nii.gz'],
                                           },
                                           )

        if 'func_mask' not in self.exclude_list:
            self._modify_parameter('func_mask_desc', 'default', "'ManualBrainMask'")
        if 'anat_mask' not in self.exclude_list:
            self._modify_parameter('anat_mask_desc', 'default', "'ManualBrainMask'")


    def create_workflow(self):
        wf = super().create_workflow()

        inputnode = wf.get_node('inputnode')
        wf.connect(inputnode, 'func_original_file', self.func2atlas.workflow, 'inputnode.func_original_file')
        wf.connect(inputnode, 'func_mask_original_file', self.func2atlas.workflow,'inputnode.func_mask_original_file')
        wf.connect(inputnode, 'anat_original_file', self.func2atlas.workflow, 'inputnode.anat_original_file')
        wf.connect(inputnode, 'anat_mask_original_file', self.func2atlas.workflow, 'inputnode.anat_mask_original_file')

        # user creates iteration list
        iteration_list = []
        for func in self.func_bids.search():
            self.anat_bids.dependent_file = func
            anat_list = self.anat_bids.search()
            if len(anat_list)<1:
                logger.warning(f'Could not find an anatomical image for {func}. Skipping.')
            for anat in anat_list:
                self.func_mask_bids.dependent_file = func
                func_mask = delistify(self.func_mask_bids.search())
                self.anat_mask_bids.dependent_file = anat
                anat_mask = delistify(self.anat_mask_bids.search())
                #print(anat.split('/')[-1], func.split('/')[-1])
                inputnode_dict = {}
                inputnode_dict['func'] = func
                inputnode_dict['func_original_file'] = func
                inputnode_dict['func_mask'] = func_mask
                inputnode_dict['func_mask_original_file'] = func_mask
                inputnode_dict['anat'] = anat
                inputnode_dict['anat_original_file'] = anat
                inputnode_dict['anat_mask'] = anat_mask
                inputnode_dict['anat_mask_original_file'] = anat_mask
                iteration_list.append(inputnode_dict)

        iterable_wf = self.iterable_inputnode(wf, iteration_list)

        return iterable_wf


if __name__ == "__main__":
    cmd_args = [
        '--func',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/sub-Nl311f9/ses-2020021001/func/sub-Nl311f9_ses-2020021001_task-rs_run-02_bold.nii.gz',
        '--anat',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/sub-Nl311f9/ses-2020021001/anat/sub-Nl311f9_ses-2020021001_acq-TurboRARE_run-01_T2w.nii.gz',
        '--label_mapping',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/DownsampleAtlasBIDS/label_mapping_host_0p3x0p3x0p55mm.txt',

        '--reg_atlas', '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/atlases/AMBMC_model_downsampled.nii.gz',
        '--reg_atlas_mask',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/atlases/AMBMC_model_downsampled_mask.nii.gz',
        '--reg_downsampled_atlas',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/DownsampleAtlasBIDS/sub-AMBMCc57bl6_desc-ModelDownsampledDownsampled0p3x0p3x0p55mm.nii.gz',
        '--reg_downsample_shift_transformation',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/DownsampleAtlasBIDS/sub-AMBMCc57bl6_desc-ModelDownsampledDownsampled0p3x0p3x0p55mmShiftTransformation.mat',

        '--reg_func_antsarg_float',
        '--reg_func_preproc_be4d_brain_extract_method', 'BRAINSUITE',
        '--reg_func_preproc_skip_mc',
        '--reg_func_preproc_tr', '1.5',
        '--reg_func_preproc_slice_timing',
        '[0.012, 0.1087741935483871, 0.2055483870967742, 0.30232258064516127, 0.3990967741935484, 0.4958709677419355, 0.5926451612903225, 0.6894193548387096, 0.7861935483870969, 0.8829677419354839, 0.979741935483871, 1.076516129032258, 1.173290322580645, 1.2700645161290323, 1.3668387096774193, 1.4636129032258063, 0.06038709677419355, 0.15716129032258064, 0.25393548387096776, 0.3507096774193548, 0.44748387096774195, 0.544258064516129, 0.6410322580645161, 0.7378064516129031, 0.8345806451612904, 0.9313548387096774, 1.0281290322580643, 1.1249032258064517, 1.2216774193548388, 1.3184516129032258, 1.415225806451613]',

        '--reg_anat_antsarg_float',
        '--reg_anat_be_brain_extract_method', 'REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK',
        '--reg_anat_be_ants_be_template',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/TemplatesAndProbabilityMasks/sub-AnatTemplate_acq-TurboRARE_desc-0p15x0p15x0p55mm20200804_T2w.nii.gz',
        '--reg_anat_be_ants_be_template_probability_mask',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/TemplatesAndProbabilityMasks/sub-AnatTemplateProbabilityMask_acq-TurboRARE_desc-0p15x0p15x0p55mm20200804_T2w.nii.gz',

        '--nipype_processing_dir', './func_corrmtx_test',
        '--keep_unnecessary_outputs',
    ]

    cmd_args2 = [
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives',
        'participant',
        '--input_derivatives_dirs',
        "['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives']",
        '--bids_layout_db', './func_corrmtx_test/bids_database',

        '--anat_base_bids_string', 'acq-TurboRARE_T2w.nii.gz',
        '--anat_session', "'2020021001'",
        '--anat_run', "'01'",

        '--func_base_bids_string', 'task-rs_bold.nii.gz',
        '--func_subject', "'Nl311f9'",
        '--func_session', "'2020021001'",
        '--func_run', "'02'",

        '--label_mapping','/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/DownsampleAtlasBIDS/label_mapping_host_0p3x0p3x0p55mm.txt',
        '--reg_atlas', '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/Atlases/sub-AMBMCc57bl6_desc-ModelDownsampled.nii.gz',
        '--reg_atlas_mask','/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/Atlases/sub-AMBMCc57bl6_desc-ModelDownsampledBrainMask.nii.gz',
        '--reg_downsample',

        '--reg_func_antsarg_float',
        '--reg_func_preproc_be4d_brain_extract_method', 'NO_BRAIN_EXTRACTION',
        '--reg_func_preproc_skip_mc',

        '--reg_anat_antsarg_float',
        '--reg_anat_be_brain_extract_method', 'REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK',
        '--reg_anat_be_ants_be_template', '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/TemplatesAndProbabilityMasks/sub-AnatTemplate_acq-TurboRARE_desc-0p15x0p15x0p55mm20200804_T2w.nii.gz',
        '--reg_anat_be_ants_be_template_probability_mask','/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/TemplatesAndProbabilityMasks/sub-AnatTemplateProbabilityMask_acq-TurboRARE_desc-0p15x0p15x0p55mm20200804_T2w.nii.gz',

        '--nipype_processing_dir', './func_corrmtx_test',
        '--keep_unnecessary_outputs',
    ]

    tmp = MouseCorrelationMatrixBIDS()
    tmp.run_bids(cmd_args2)
