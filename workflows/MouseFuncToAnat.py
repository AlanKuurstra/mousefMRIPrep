import argparse
from workflows.CFMMWorkflow import CFMMWorkflow
from workflows.CFMMBIDS import CFMMBIDSWorkflowMixin, CMDLINE_VALUE, BIDSInputExternalSearch
from workflows.CFMMAnts import AntsDefaultArguments, CFMMAntsRegistration
from workflows.MouseBrainExtraction import MouseBrainExtractionBIDS
from workflows.CFMMCommon import NipypeWorkflowArguments, delistify
from nipype.pipeline import engine as pe
from nipype.interfaces.utility import Function
from workflows.MouseFuncPreprocessing import MouseFuncPreprocessing, MouseFuncPreprocessingBIDS
from workflows.CFMMLogging import NipypeLogger as logger


class MouseFuncToAnatANTs(CFMMAntsRegistration):
    """
    Wrapper class for CFMMAntsRegistration with default parameter values suitable for mouse brains.
    """

    def _add_parameters(self):
        """
        Wrapper class for CFMMBse with default parameter values suitable for mouse brains.
        """
        super()._add_parameters()
        # note: the type conversion function you provided to argparse is only called on string defaults
        # therefore a default of 3 will set the argument to 3 (both integers)
        # a default of '3' will go through the convert function and in our case convert_argparse_using_eval.convert()'s
        # eval() function will convert the string to integer 3
        # it is important to to include two sets of quotes if the default value is supposed to be a string
        # so that after the eval function, it will still be a string
        self._modify_parameter('output_transform_prefix', 'default', "'output_'")
        self._modify_parameter('dimension', 'default', 3)
        self._modify_parameter('transforms', 'default', "['Affine', 'SyN']")
        # transform_parameters:
        # gradient step
        # updateFieldVarianceInVoxelSpace - smooth the deformation computed on the "updated" gradient field before this is added to previous deformations to form the "total" gradient field
        # totalFieldVarianceInVoxelSpace - smooth the deformation computed on the "total" gradient field
        self._modify_parameter('transform_parameters', 'default', "[(0.005,), (0.005, 3.0, 0.0)]")

        # transform for each stage vs composite for entire warp
        self._modify_parameter('write_composite_transform', 'default', "True")
        # combines adjacent transforms when possible
        self._modify_parameter('collapse_output_transforms', 'default', "False")

        self._modify_parameter('metric', 'default', "[['MI'],['MI', 'CC']]")
        self._modify_parameter('number_of_iterations', 'default', "[[10000, 10000, 10000], [200, 200, 500]]")
        # weight used if you do multimodal registration. Default is 1 (value ignored currently by ANTs)
        self._modify_parameter('metric_weight', 'default', "[[1],[0.5, 0.5]]")
        # radius for CC between 2-5
        self._modify_parameter('radius_or_number_of_bins', 'default', "[[32], [32, 4]]")
        self._modify_parameter('use_histogram_matching', 'default', "[False, True]")

        # use a negative number if you want to do all iterations and never exit
        self._modify_parameter('convergence_threshold', 'default', "[1.e-10] * 2")
        # if the cost hasn't changed by convergence threshold in the last window size iterations, exit loop
        self._modify_parameter('convergence_window_size', 'default', "[15]*2")
        self._modify_parameter('smoothing_sigmas', 'default', "[[0.3, 0.2, 0]] * 2")
        # we use mm instead of vox because we don't have isotropic voxels
        self._modify_parameter('sigma_units', 'default', "['mm'] * 2  ")
        self._modify_parameter('shrink_factors', 'default', "[[1, 1, 1]]*2")
        # estimate the learning rate step size only at the beginning of each level. Does this override the value chosen in transform_parameters?
        self._modify_parameter('use_estimate_learning_rate_once', 'default', "[True] * 2")

        self._modify_parameter('output_warped_image', 'default', "'output_warped_image.nii.gz'")


class MouseFuncToAnat(CFMMWorkflow):
    group_name = 'Functional Preprocessing and Registration'
    flag_prefix = 'func_'

    def _add_parameters(self):
        self._add_parameter('in_file',
                            help='Explicitly specify location of the input functional for anatomical registration.')
        self._add_parameter('in_file_mask',
                            help='Explicitly specify location of the input functional mask for anatomical registration.')
        self._add_parameter('anat',
                            help='Explicitly specify location of the anatomical to be registered to.')
        self._add_parameter('anat_mask',
                            help='Explicitly specify location of the anatomical mask.')
        self._add_parameter('no_mask_func2anat',
                            action='store_true',
                            help="Don't use masks during the functional to anatomical registration.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nipype = NipypeWorkflowArguments(owner=self, exclude_list=['nthreads_mapnode', 'mem_gb_mapnode'])
        self.ants_args = AntsDefaultArguments(owner=self)
        self.preproc = MouseFuncPreprocessing(owner=self, exclude_list=['in_file', 'in_file_mask'])
        self.ants_reg = MouseFuncToAnatANTs(owner=self)

        self.ants_reg.get_parameter('float').default_provider = self.ants_args.get_parameter('float')
        self.ants_reg.get_parameter('interpolation').default_provider = self.ants_args.get_parameter('interpolation')
        self.ants_reg.get_parameter('num_threads').default_provider = self.nipype.get_parameter('nthreads_node')

        # motion correction happens before slice timing correction (same as fMRIPrep)
        # so if we're doing stc, then we can't concatenate MC transform with func2anat
        self.outputs = ['func_avg_to_anat', 'func_to_anat_composite_transform', 'preprocessed', 'avg', 'brain_mask']

    def create_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        preproc_wf = self.preproc.create_workflow()
        ants_reg = self.ants_reg.get_node()

        inputnode, outputnode, wf = self.get_io_and_workflow()

        wf.connect([
            (inputnode, preproc_wf, [('in_file', 'inputnode.in_file')]),
            (inputnode, preproc_wf, [('in_file_mask', 'inputnode.in_file_mask')]),
            (inputnode, ants_reg, [('anat', 'fixed_image')]),
            (preproc_wf, outputnode, [('outputnode.mc_transform', 'mc_transform'),
                                      ('outputnode.preprocessed', 'preprocessed'),
                                      ('outputnode.avg', 'avg')]),
            (preproc_wf, ants_reg, [('outputnode.avg', 'moving_image')]),
            (ants_reg, outputnode, [('warped_image', 'func_avg_to_anat'),
                                    ('composite_transform', 'func_to_anat_composite_transform'), ]),

        ])
        if self.preproc.outputnode_field_connected('brain_mask'):
            wf.connect([(preproc_wf, outputnode, [('outputnode.brain_mask', 'brain_mask'),])])

        if not self.get_parameter('no_mask_func2anat').user_value:
            wf.connect([
                (inputnode, ants_reg, [('anat_mask', 'fixed_image_mask')]),
                (preproc_wf, ants_reg, [('outputnode.brain_mask', 'moving_image_mask')]),
            ])

        # if using motion correction, but not use slice timing correction, then the motion correction should really
        # be concatenated with the atlas registration transformations for a single interpolation. Then, the
        # full transformation = [MC transformation, func2anat transformation, anat2atlas transformation]
        # however, since stc corrects values in k-space (not in image space), it can't be concatenated with spatial
        # displacements. In this case one would have to do multiple interpolations. First MC, then stc, then the
        # concatenated transformation from func2atlas.

        # since CFMM rarely ignores stc, the logic to concatenate MC with func2atlas when skipping stc has not
        # yet been implemented

        # if self.preproc.get_parameter('skip_stc').user_value:
        #     wf.connect([
        #         (preproc_wf, concat_list_func_to_anat,[('outputnode.mc_transform', 'apply_first')]),
        #         (ants_reg, concat_list_func_to_anat,[('composite_transform', 'apply_second')]),
        #         (preproc_wf, outputnode[('outputnode.preprocessed_before_mc', 'preprocessed')]),
        #         (concat_list_func_to_anat, outputnode[('concatenated_transform', 'func_to_anat_composite_transform')]),
        #     ])

        return wf


class MouseFuncToAnatBIDS(MouseFuncToAnat, CFMMBIDSWorkflowMixin):
    def __init__(self, *args, **kwargs):
        self.exclude_parameters(['slice_timing', 'tr', 'slice_encoding_direction', ])
        super().__init__(*args, **kwargs)
        self.add_bids_parameter_group()
        self.bids._modify_parameter('analysis_level', 'choices', ['participant'])

        # easier to use MouseFuncPreprocessing existing BIDS component because it takes care of removing
        # tr, slice encoding direction, and slice timings with information from the sidecar file
        # this is unusual behaviour
        self._remove_subcomponent('preproc')
        self.preproc = MouseFuncPreprocessingBIDS(owner=self,
                                                  exclude_list=['in_file', 'in_file_mask'],
                                                  )
        self.preproc.in_file_bids.disable_derivatives = True

        self.in_file_bids = BIDSInputExternalSearch(self,
                                         'in_file',
                                         entities_to_overwrite={'subject': CMDLINE_VALUE,
                                                                'session': CMDLINE_VALUE,
                                                                'run': CMDLINE_VALUE,
                                                                'extension': ['.nii', '.nii.gz'],
                                                                },
                                         output_derivatives={
                                             'func_avg_to_anat': 'FuncAvgToAnat',
                                             'func_to_anat_composite_transform': 'FuncToAnatTransform',
                                             'preprocessed': 'Preproc',
                                             'avg': 'Avg',
                                             'brain_mask': 'BrainMask',
                                         })

        self.in_file_mask_bids = BIDSInputExternalSearch(self,
                                              'in_file_mask',
                                                         dependent_search=self.in_file_bids,
                                              dependent_entities=['subject', 'session', 'run'],
                                              create_base_bids_string=False,
                                              entities_to_overwrite={'desc': CMDLINE_VALUE,
                                                                     'extension': ['.nii', '.nii.gz']},
                                              )

        self.anat_bids = BIDSInputExternalSearch(self,
                                      'anat',
                                                 dependent_search=self.in_file_bids,
                                      dependent_entities=['subject'],
                                      entities_to_overwrite={
                                          'session': CMDLINE_VALUE,
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
        self._modify_parameter('in_file_mask_desc', 'default', "'ManualBrainMask'")
        self._modify_parameter('anat_mask_desc', 'default', "'ManualBrainMask'")

    def create_workflow(self):
        wf = super().create_workflow()
        self.add_bids_to_workflow(wf)
        inputnode = self.inputnode

        # when replacing a normal subcomponent with its bid version, (which is an unusual activity)
        # we must manually provide the original bids file
        wf.connect([
            (inputnode, self.preproc.workflow, [('in_file_original_file', 'inputnode.in_file_original_file'),
                                                ('in_file_mask_original_file', 'inputnode.in_file_mask_original_file')
                                                ])
            ])

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

        '--anat_base_bids_string', "'acq-TurboRARE_T2w.nii.gz'",
        '--anat_session', "'2020021001'",
        '--anat_run', "'01'",

        '--preproc_be4d_ants_be_antsarg_float',
        '--preproc_be4d_brain_extract_method', 'NO_BRAIN_EXTRACTION',
        '--nipype_processing_dir', "'./func_preprocessing_test'",
        '--keep_unnecessary_outputs',

        '--preproc_smooth_fwhm', '0.6',
        '--preproc_smooth_brightness_threshold', '20.0',
        '--preproc_tf_highpass_sigma', '33',
        '--preproc_skip_mc',
    ]


    tmp = MouseFuncToAnatBIDS()
    tmp.run_bids(bids_args)
