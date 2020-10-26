from workflows.CFMMWorkflow import CFMMWorkflow
from workflows.MouseFuncToAnat import MouseFuncToAnat, MouseFuncToAnatBIDS
from workflows.MouseAnatToAtlas import MouseAnatToAtlas
from workflows.CFMMAnts import get_node_ants_transform_concat_list
from nipype.interfaces.ants import ApplyTransforms
from nipype.pipeline import engine as pe
from workflows.CFMMBIDS import CFMMBIDSWorkflowMixer, BIDSInputExternalSearch, CMDLINE_VALUE, BIDSInputWorkflow, \
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
        self._add_parameter('anat_to_atlas_composite_transform',
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
        self.outputs = ['func_to_atlas', 'func_to_atlas_composite_transform']

    def create_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        func2anat_wf = self.func2anat.create_workflow()

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
            (inputnode, func2anat_wf, [('anat_mask', 'inputnode.anat_mask')]),
            (inputnode, concat_list_func_to_atlas, [('anat_to_atlas_composite_transform', 'apply_second')]),

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


class MouseFuncToAtlasBIDS(MouseFuncToAtlas, CFMMBIDSWorkflowMixer):
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
                                          'func_to_atlas':'FuncToAtlas',
                                          'func_to_atlas_composite_transform':'FuncToAtlasTransform',
                                      }
                                      )

        self.func_mask_bids = BIDSInputExternalSearch(self,
                                           'func_mask',
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
                                      dependent_entities=['subject'],
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
                                                      dependent_iterable=self.anat_bids,
                                                      dependent_entities=['subject', 'session', 'run'],
                                                      create_base_bids_string=False,
                                                      entities_to_overwrite={
                                                          'desc': CMDLINE_VALUE,
                                                          'extension': ['.nii', '.nii.gz'],
                                                      },
                                                      )
        self._modify_parameter('anat_mask_desc', 'default', "'ManualBrainMask'")

        # Having the bids search as nodes inside the pipeline (using BIDSInputWorkflow or BIDSDerivativesInputWorkflow)
        # allows for a bids search that is dependent on an iterable without having to bring the iterable all the way
        # to the top level. The iterable will arrive on the inputnode inside the pipeline and the search can still
        # be done inside the pipeline.
        self.anat_to_atlas_composite_transform_bids = BIDSDerivativesInputWorkflow(self,
                                                                                   'anat_to_atlas_composite_transform',
                                                                                   base_input='anat',
                                                                                   base_input_derivative_desc='AnatToAtlasTransform',
                                                                                   base_input_derivative_extension=[
                                                                                       '.h5'],
                                                                                   )

        # want to do a bids search for downsampled atlas
        # first create the parameter that the bids search will be around
        self._add_parameter('atlas',
                            help='')
        # create the bids search subcomponent related to the atlas parameter
        self.atlas_bids = BIDSInputWorkflow(self,
                                            'atlas',
                                            )
        self.res_desc_node = get_node_dynamic_res_desc('res_desc')
        # create the derivative search subcomponent that relies on the atlas bids search.
        self.downsample_atlas_bids = BIDSDerivativesInputWorkflow(self,
                                                                  'downsampled_atlas',
                                                                  base_input='atlas',
                                                                  base_input_derivative_desc=self.res_desc_node,
                                                                  base_input_derivative_extension=['.nii', '.nii.gz'],
                                                                  )
        # create the derivative search subcomponent that relies on the downsampled atlas.
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
        # the func,func_mask,anat,anat_mask connectinos are done in super
        # the bids_original file connections are not.
        wf.connect([
            (inputnode, self.func2anat.workflow, [('func_original_file', 'inputnode.in_file_original_file'),
                                                ('func_mask_original_file', 'inputnode.in_file_mask_original_file'),
                                                  ('anat_original_file', 'inputnode.anat_original_file'),
                                                  ('anat_mask_original_file', 'inputnode.anat_mask_original_file'),
                                                ])
            ])

        bids_input_searches = wf.get_node('BIDSInputSearches')
        wf.connect(inputnode, 'func', bids_input_searches, f'{self.res_desc_node.name}.reference')

        return wf



if __name__ == "__main__":
    cmd_args = [
        # bidsapp
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives',
        'participant',
        '--input_derivatives_dirs',
        "['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives']",
        '--bids_layout_db', './func_reg_test/bids_database',

        '--anat_base_bids_string', 'acq-TurboRARE_T2w.nii.gz',

        '--func_base_bids_string', 'task-rs_bold.nii.gz',
        '--func_subject', "'Nl311f9'",
        '--func_session', "'2020021001'",
        '--func_run', "'01'",

        '--func_antsarg_float',
        '--func_preproc_be4d_brain_extract_method', 'NO_BRAIN_EXTRACTION',
        '--func_preproc_skip_mc',

        '--atlas',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/Atlases/sub-AMBMCc57bl6_desc-ModelDownsampled.nii.gz',
        '--downsample',
        '--nipype_processing_dir', './func_reg_test',
        '--keep_unnecessary_outputs',
    ]

    cmd_args2 = [
        '--func',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/sub-Nl311f9/ses-2020021001/func/sub-Nl311f9_ses-2020021001_task-rs_run-02_bold.nii.gz',
        '--anat',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/sub-Nl311f9/ses-2020021001/anat/sub-Nl311f9_ses-2020021001_acq-TurboRARE_run-01_T2w.nii.gz',
        '--atlas', '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/atlases/AMBMC_model_downsampled.nii.gz',
        '--atlas_mask',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/atlases/AMBMC_model_downsampled_mask.nii.gz',
        '--downsampled_atlas',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/DownsampleAtlasBIDS/sub-AMBMCc57bl6_desc-ModelDownsampledDownsampled0p3x0p3x0p55mm.nii.gz',
        '--downsample_shift_transformation',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/DownsampleAtlasBIDS/sub-AMBMCc57bl6_desc-ModelDownsampledDownsampled0p3x0p3x0p55mmShiftTransformation.mat',

        '--func_antsarg_float',
        '--func_preproc_be4d_brain_extract_method', 'BRAINSUITE',
        '--func_preproc_skip_mc',
        '--func_preproc_tr', '1.5',
        '--func_preproc_slice_timing',
        '[0.012, 0.1087741935483871, 0.2055483870967742, 0.30232258064516127, 0.3990967741935484, 0.4958709677419355, 0.5926451612903225, 0.6894193548387096, 0.7861935483870969, 0.8829677419354839, 0.979741935483871, 1.076516129032258, 1.173290322580645, 1.2700645161290323, 1.3668387096774193, 1.4636129032258063, 0.06038709677419355, 0.15716129032258064, 0.25393548387096776, 0.3507096774193548, 0.44748387096774195, 0.544258064516129, 0.6410322580645161, 0.7378064516129031, 0.8345806451612904, 0.9313548387096774, 1.0281290322580643, 1.1249032258064517, 1.2216774193548388, 1.3184516129032258, 1.415225806451613]',

        '--anat_antsarg_float',
        '--anat_be_brain_extract_method', 'REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK',
        '--anat_be_ants_be_template',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/TemplatesAndProbabilityMasks/sub-AnatTemplate_acq-TurboRARE_desc-0p15x0p15x0p55mm20200804_T2w.nii.gz',
        '--anat_be_ants_be_template_probability_mask',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/TemplatesAndProbabilityMasks/sub-AnatTemplateProbabilityMask_acq-TurboRARE_desc-0p15x0p15x0p55mm20200804_T2w.nii.gz',

        '--nipype_processing_dir', './func_reg_test',
        '--keep_unnecessary_outputs',
    ]

    tmp = MouseFuncToAtlasBIDS()
    tmp.run(cmd_args)
