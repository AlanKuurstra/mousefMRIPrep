from workflows.CFMMWorkflow import CFMMWorkflow
from workflows.MouseFuncToAnat import MouseFuncToAnat, MouseFuncToAnatBIDS
from workflows.MouseAnatToAtlas import MouseAnatToAtlas
from workflows.CFMMAnts import get_node_ants_transform_concat_list
from nipype.interfaces.ants import ApplyTransforms
from nipype.pipeline import engine as pe
from workflows.CFMMBIDS import CFMMBIDSWorkflowMixin, BIDSInputExternalSearch, CMDLINE_VALUE, BIDSInputWorkflow, \
    BIDSDerivativesInputWorkflow
from workflows.CFMMCommon import delistify
from workflows.CFMMLogging import NipypeLogger as logger


class MouseFuncToAtlas(CFMMWorkflow):
    group_name = 'Functional to Atlas Registration'
    flag_prefix = 'reg_'

    def validate_parameters(self):
        # if downsample atlas is enabled make sure either
        # label_images is given
        # or downsampled_atlas and downsampled_label_map are given (defined together)

        # ensure downsampled_atlas, downsampled_labels_map, and func have the same resolution
        pass

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
        self._add_parameter('atlas',
                            help='')
        self._add_parameter('atlas_mask',
                            help='')
        self._add_parameter('skip_func2anat_reg',
                            action='store_true',
                            help=f'Assume functional and anatomical are already aligned. Do not apply functional to anatomical registration.',
                            )
        self._add_parameter('downsample',
                            action='store_true',
                            help=f'Use a downampled atlas as the reference for the final transformation of the '
                                 f'functional image.  The full resolution atlas is used to obtain 3D registration '
                                 f'transformations, but the downsampled atlas will be used as the reference when '
                                 f'applying those transformations to the 4D functional image.',
                            )

        self._add_parameter('downsampled_atlas',
                            help=f'This image is used as the reference for the final transformation of the '
                                 f'functional image.  The full resolution atlas is used to obtain 3D registration '
                                 f'transformations, but the downsampled atlas will be used as the reference when '
                                 f'applying those transformations to the 4D functional image.')

        self._add_parameter('downsample_shift_transformation',
                            help=f'If the downsampled atlas is shifted with respect to the original atlas due to the '
                                 f'algorithm used, then the user can provide a transformation file to apply the '
                                 f'corresponding shift to the registered functional image.')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func2anat = MouseFuncToAnat(owner=self, exclude_list=['in_file', 'in_file_mask', 'anat', 'anat_mask'])
        self.anat2atlas = MouseAnatToAtlas(owner=self, exclude_list=['in_file', 'in_file_mask', 'atlas', 'atlas_mask'])
        self.outputs = ['func_to_atlas', 'func_to_atlas_composite_transform']

    def create_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        func2anat_wf = self.func2anat.create_workflow()
        anat2atlas_wf = self.anat2atlas.create_workflow()

        concat_list_func_to_atlas = get_node_ants_transform_concat_list(name='concat_list_func_to_atlas')

        concat_transforms_func_to_atlas = pe.Node(interface=ApplyTransforms(), name='concat_transforms_func_to_atlas', )
        concat_transforms_func_to_atlas.inputs.dimension = 3
        # concat_transforms_func_to_atlas.inputs.float = reduce_to_float_precision
        # if gzip_large_images:
        #     concat_transforms_func_to_atlas.inputs.output_image = 'func_to_atlas_transform.nii.gz'
        # else:
        #     concat_transforms_func_to_atlas.inputs.output_image = 'func_to_atlas_transform.nii'

        # FOR SOME REASON THE H5 CONCATENATION DOESN'T WORK :(
        # concat_transforms_func_to_atlas.inputs.output_image = 'func_to_atlas_transform.h5'
        concat_transforms_func_to_atlas.inputs.output_image = 'func_to_atlas_transform.nii.gz'
        concat_transforms_func_to_atlas.inputs.print_out_composite_warp_file = True
        concat_transforms_func_to_atlas.inputs.float = True

        register_func_to_atlas = pe.Node(interface=ApplyTransforms(), name='register_func_to_atlas', )
        # if gzip_large_images:
        #     register_func_to_atlas.inputs.output_image = 'warped.nii.gz'
        # else:
        #     register_func_to_atlas.inputs.output_image = 'warped.nii'
        # register_func_to_atlas.inputs.float = reduce_to_float_precision
        # register_func_to_atlas.inputs.interpolation = interpolation
        register_func_to_atlas.inputs.dimension = 3
        register_func_to_atlas.inputs.input_image_type = 3

        inputnode, outputnode, wf = self.get_io_and_workflow()

        wf.connect([
            (inputnode, func2anat_wf, [('func', 'inputnode.in_file')]),
            (inputnode, func2anat_wf, [('func_mask', 'inputnode.in_file_mask')]),
            (inputnode, func2anat_wf, [('anat', 'inputnode.anat')]),
            (anat2atlas_wf, func2anat_wf, [('outputnode.brain_mask', 'inputnode.anat_mask')]),

            (inputnode, anat2atlas_wf, [('anat', 'inputnode.in_file')]),
            (inputnode, anat2atlas_wf, [('anat_mask', 'inputnode.in_file_mask')]),
            (inputnode, anat2atlas_wf, [('atlas', 'inputnode.atlas')]),
            (inputnode, anat2atlas_wf, [('atlas_mask', 'inputnode.atlas_mask')]),

            (anat2atlas_wf, concat_list_func_to_atlas,
             [('outputnode.anat_to_atlas_composite_transform', 'apply_second')]),

            (concat_list_func_to_atlas, concat_transforms_func_to_atlas, [('transforms', 'transforms')]),
            (inputnode, concat_transforms_func_to_atlas, [('atlas', 'reference_image')]),
            (func2anat_wf, concat_transforms_func_to_atlas, [('outputnode.avg', 'input_image')]),

            (func2anat_wf, register_func_to_atlas, [('outputnode.preprocessed', 'input_image')]),
            (concat_transforms_func_to_atlas, register_func_to_atlas, [('output_image', 'transforms')]),

            (register_func_to_atlas, outputnode, [('output_image', 'func_to_atlas')]),
            (concat_transforms_func_to_atlas, outputnode, [('output_image', 'func_to_atlas_composite_transform')]),
        ])

        if not self.get_parameter('skip_func2anat_reg').user_value:
            wf.connect([
                (func2anat_wf, concat_list_func_to_atlas,
                 [('outputnode.func_to_anat_composite_transform', 'apply_first')]),
            ])

        if self.get_parameter('downsample').user_value:
            wf.connect([
                (inputnode, concat_list_func_to_atlas, [('downsample_shift_transformation', 'apply_third')]),
                (inputnode, register_func_to_atlas, [('downsampled_atlas', 'reference_image')]),
            ])
        else:
            wf.connect([
                (inputnode, register_func_to_atlas, [('atlas', 'reference_image')]),
            ])
        return wf


from workflows.DownsampleAtlas import get_node_dynamic_res_desc


class MouseFuncToAtlasBIDS(MouseFuncToAtlas, CFMMBIDSWorkflowMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._remove_subcomponent('func2anat')
        self.func2anat = MouseFuncToAnatBIDS(owner=self, exclude_list=['in_file', 'in_file_mask', 'anat', 'anat_mask'])

        self.add_bids_parameter_group()
        self.bids._modify_parameter('analysis_level', 'choices', ['participant'])

        self.func_bids = BIDSInputExternalSearch(self,
                                                 'func',
                                                 entities_to_overwrite={'subject': CMDLINE_VALUE,
                                                                        'session': CMDLINE_VALUE,
                                                                        'run': CMDLINE_VALUE,
                                                                        'extension': ['.nii', '.nii.gz'],
                                                                        },
                                                 output_derivatives={
                                                     'func_to_atlas': 'FuncToAtlas',
                                                     'func_to_atlas_composite_transform': 'FuncToAtlasTransform',
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
                                                 dependent_entities=['subject', 'session'],
                                                 entities_to_overwrite={
                                                     'session': CMDLINE_VALUE,
                                                     'run': CMDLINE_VALUE,
                                                     'extension': ['.nii', '.nii.gz'],
                                                     'scope': 'self',
                                                 },
                                                 )

        # although we could make anat_mask a derivative input like anat_to_atlas_composite_transform below, we want
        # it's parameters to be in the same place of the program help as anat - and so we bring it up the nested pipelines
        # with anat.
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

        self.atlas_bids = BIDSInputWorkflow(self,
                                            'atlas',
                                            )
        self.atlas_mask_bids = BIDSInputWorkflow(self,
                                                 'atlas_mask',
                                                 base_input=('atlas', ['subject', 'session', 'run', 'desc']),
                                                 entities_to_overwrite={'extension': ['.nii', '.nii.gz']},
                                                 entities_to_extend=[('desc', CMDLINE_VALUE)],
                                                 )

        self.res_desc_node = get_node_dynamic_res_desc('res_desc')
        self.downsample_atlas_bids = BIDSDerivativesInputWorkflow(self,
                                                                  'downsampled_atlas',
                                                                  base_input='atlas',
                                                                  base_input_derivative_desc=self.res_desc_node,
                                                                  base_input_derivative_extension=['.nii', '.nii.gz'],
                                                                  )
        self.downsample_atlas_shift_bids = BIDSDerivativesInputWorkflow(self,
                                                                        'downsample_shift_transformation',
                                                                        base_input='downsampled_atlas',
                                                                        base_input_derivative_desc='ShiftTransformation',
                                                                        base_input_derivative_extension=['.mat'],
                                                                        )

    def create_workflow(self):
        wf = super().create_workflow()
        self.add_bids_to_workflow(wf)

        inputnode = wf.get_node('inputnode')
        wf.connect([
            (inputnode, self.func2anat.workflow, [
                ('func_original_file','inputnode.in_file_original_file'),
                ('func_mask_original_file', 'inputnode.in_file_mask_original_file'),
                ('anat_original_file', 'inputnode.anat_original_file'),
                ('anat_mask_original_file', 'inputnode.anat_mask_original_file'),
            ])
        ])
        bids_input_searches = wf.get_node('BIDSInputSearches')
        wf.connect(inputnode, 'func', bids_input_searches, f'{self.res_desc_node.name}.reference')
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

        '--func_antsarg_float',
        '--func_preproc_be4d_brain_extract_method', 'BRAINSUITE',
        '--func_preproc_skip_mc',

        '--anat_antsarg_float',
        '--anat_be_brain_extract_method', 'REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK',
        '--anat_be_ants_be_template',
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/TemplatesAndProbabilityMasks/sub-AnatTemplate_acq-TurboRARE_desc-0p15x0p15x0p55mm20200804_T2w.nii.gz'",
        '--anat_be_ants_be_template_probability_mask',
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/TemplatesAndProbabilityMasks/sub-AnatTemplateProbabilityMask_acq-TurboRARE_desc-0p15x0p15x0p55mm20200804_T2w.nii.gz'",

        '--atlas',
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/Atlases/sub-AMBMCc57bl6_desc-ModelDownsampled.nii.gz'",
        '--atlas_mask',
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/Atlases/sub-AMBMCc57bl6_desc-ModelDownsampledBrainMask.nii.gz'",
        '--downsample',
        '--nipype_processing_dir', "'./func_reg_test'",
        '--keep_unnecessary_outputs',
    ]


    tmp = MouseFuncToAtlasBIDS()
    tmp.run_bids(bids_args)
