from workflows.CFMMAnts import CFMMN4BiasFieldCorrection, CFMMAntsRegistration
from workflows.CFMMBrainSuite import CFMMBse
from workflows.BrainExtraction import BrainSuiteBrainExtraction, AntsBrainExtraction, BrainExtraction, \
    BrainExtraction4D, BrainSuiteBrainExtractionBIDS, AntsBrainExtractionBIDS, BrainExtractionBIDS#BrainExtraction4DBIDS
class MouseN4BiasFieldCorrection(CFMMN4BiasFieldCorrection):
    """
    Wrapper class for CFMMN4BiasFieldCorrection with default parameter values suitable for mouse brains.
    """
    def _add_parameters(self):
        """
        Modifies default parameter values.
        """
        super()._add_parameters()
        self._modify_parameter('bspline_fitting_distance', 'default', 20.0)
        self._modify_parameter('dimension', 'default', 3)
        self._modify_parameter('save_bias', 'default', False)
        self._modify_parameter('copy_header', 'default', True)
        self._modify_parameter('n_iterations', 'default', [50] * 4)
        self._modify_parameter('convergence_threshold', 'default', 1e-7)
        self._modify_parameter('shrink_factor', 'default', 4)

class MouseAntsRegistrationBE(CFMMAntsRegistration):
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
        self._modify_parameter('initial_moving_transform_com', 'default', "1")
        self._modify_parameter('transforms', 'default', "['Affine', 'SyN']")
        # transform_parameters:
        # gradient step
        # updateFieldVarianceInVoxelSpace - smooth the deformation computed on the "updated" gradient field before this is added to previous deformations to form the "total" gradient field
        # totalFieldVarianceInVoxelSpace - smooth the deformation computed on the "total" gradient field
        self._modify_parameter('transform_parameters', 'default', "[(0.1,), (0.1, 3.0, 0.0)]")
        """*********************************************************************************************************************************"""
        self._modify_parameter('number_of_iterations', 'default', "[[1, 0, 0], [0, 0, 0]]")
        """*********************************************************************************************************************************"""
        # transform for each stage vs composite for entire warp
        self._modify_parameter('write_composite_transform', 'default', "True")
        # combines adjacent transforms when possible
        self._modify_parameter('collapse_output_transforms', 'default', "False")
        # ants_reg.inputs.initialize_transforms_per_stage = False #seems to be for initializing linear transforms only
        # using CC when atlas was made using same protocol
        self._modify_parameter('metric', 'default', "['CC'] * 2")
        # weight used if you do multimodal registration. Default is 1 (value ignored currently by ANTs)
        self._modify_parameter('metric_weight', 'default', "[1] * 2")
        # radius for CC between 2-5
        self._modify_parameter('radius_or_number_of_bins', 'default', "[5] * 2")
        # not entirely sure why we don't need to specify sampling strategy and percentage for non-linear syn registration
        # but I'm just following ANTs examples
        self._modify_parameter('sampling_strategy', 'default', "['Regular', None]")
        self._modify_parameter('sampling_percentage', 'default', "[0.5, None]")
        # use a negative number if you want to do all iterations and never exit
        self._modify_parameter('convergence_threshold', 'default', "[1.e-8,1.e-9]")
        # if the cost hasn't changed by convergence threshold in the last window size iterations, exit loop
        self._modify_parameter('convergence_window_size', 'default', "[10] * 2")
        self._modify_parameter('smoothing_sigmas', 'default', "[[0.3, 0.15, 0], [0.3, 0.15, 0]]")
        # we use mm instead of vox because we don't have isotropic voxels
        self._modify_parameter('sigma_units', 'default', "['mm'] * 2  ")
        self._modify_parameter('shrink_factors', 'default', "[[3, 2, 1], [3, 2, 1]]")
        # estimate the learning rate step size only at the beginning of each level. Does this override the value chosen in transform_parameters?
        self._modify_parameter('use_estimate_learning_rate_once', 'default', "[True,True]")
        self._modify_parameter('use_histogram_matching', 'default', "[True, True]")
        self._modify_parameter('output_warped_image', 'default', "'output_warped_image.nii.gz'")

class MouseBse(CFMMBse):
    """
    Wrapper class for CFMMBse with default parameter values suitable for mouse brains.
    """
    def _add_parameters(self):
        """
        Modifies default parameter values.
        """
        super()._add_parameters()
        self._modify_parameter('diffusionConstant', 'default', 30.0)
        self._modify_parameter('diffusionIterations', 'default', 3)
        self._modify_parameter('edgeDetectionConstant', 'default', 0.55)
        self._modify_parameter('radius', 'default', 2)
        self._modify_parameter('dilateFinalMask', 'default', True)
        self._modify_parameter('trim', 'default', False)
        self._modify_parameter('noRotate', 'default', True)


class MouseBrainSuiteBrainExtraction(BrainSuiteBrainExtraction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._remove_subcomponent('bse')
        self.bse = MouseBse(owner=self)

class MouseBrainSuiteBrainExtractionBIDS(BrainSuiteBrainExtractionBIDS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._remove_subcomponent('bse')
        self.bse = MouseBse(owner=self)

class MouseAntsBrainExtraction(AntsBrainExtraction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._remove_subcomponent('ants_reg')
        self.ants_reg = MouseAntsRegistrationBE(owner=self)

class MouseAntsBrainExtractionBIDS(AntsBrainExtractionBIDS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._remove_subcomponent('ants_reg')
        self.ants_reg = MouseAntsRegistrationBE(owner=self)

class MouseBrainExtraction(BrainExtraction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._remove_subcomponent('n4')
        self._remove_subcomponent('bse')
        self._remove_subcomponent('ants')
        self.n4 = MouseN4BiasFieldCorrection(owner=self)
        self.bse = MouseBrainSuiteBrainExtraction(owner=self, exclude_list=['in_file'])
        self.ants = MouseAntsBrainExtraction(owner=self,
                                             exclude_list=['in_file', 'in_file_mask'],
                                             replaced_parameters={
                                                 'brain_extract_method': self.get_parameter('brain_extract_method')
                                             })

class MouseBrainExtractionBIDS(BrainExtractionBIDS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._remove_subcomponent('n4')
        self._remove_subcomponent('bse')
        self._remove_subcomponent('ants')
        self.n4 = MouseN4BiasFieldCorrection(owner=self)
        self.bse = MouseBrainSuiteBrainExtractionBIDS(
            owner=self,
            exclude_list=['in_file'],
            save_derivatives=False
        )

        self.ants = MouseAntsBrainExtractionBIDS(
            owner=self,
            exclude_list=['in_file', 'in_file_mask'],
            replaced_parameters={
                'brain_extract_method': self.get_parameter('brain_extract_method')
            },
            save_derivatives=False
        )
class MouseBrainExtraction4D(BrainExtraction4D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._remove_subcomponent('n4')
        self._remove_subcomponent('bse')
        self._remove_subcomponent('ants')
        self.n4 = MouseN4BiasFieldCorrection(owner=self)

        self.bse = MouseBrainSuiteBrainExtraction(owner=self,exclude_list=['in_file'])
        self.ants = MouseAntsBrainExtraction(owner=self,
                                             exclude_list=['in_file','in_file_mask','brain_extract_method'],
                                             replaced_parameters={
                                                 'brain_extract_method': self.get_parameter('brain_extract_method'),
                                             }
                                             )

# class MouseBrainExtraction4DBIDS(BrainExtraction4DBIDS):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._remove_subcomponent('n4')
#         self._remove_subcomponent('bse')
#         self._remove_subcomponent('ants')
#         self.n4 = MouseN4BiasFieldCorrection(owner=self)
#         self.bse = MouseBrainSuiteBrainExtraction(owner=self, exclude_list=['in_file'],save_derivatives=False)
#         self.ants = MouseAntsBrainExtraction(owner=self,
#                                              exclude_list=['in_file','in_file_mask','brain_extract_method'],
#                                              replaced_parameters={
#                                                  'brain_extract_method': self.get_parameter('brain_extract_method'),
#                                              },
#                                              save_derivatives=False
#                                              )
#
# def print_component(component):
#     print(component)
#     for k, v in component._parameters.items():
#         print(f'  {k}, {id(v)}, {v.flagname}, {v.groupname}, {v.user_value}')
#     if hasattr(component, 'subcomponents'):
#         for subcomponent in component.subcomponents:
#             print_component(subcomponent)


if __name__ == "__main__":
    from workflows.CFMMCommon import NipypeRunArguments
    from workflows.CFMMConfigFile import CFMMConfig
    from workflows.CFMMLogging import NipypeLogger as logger
    from workflows.CFMMBase import CFMMParserGroups
    import configargparse

    cmd_args = [
        # bidsapp
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives',
        'participant',
        '--input_derivatives_dirs',
        "['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives']",
        '--bids_layout_db', './brain_extract_test/bids_database',
        '--participant_label', 'Nl311f9',
        '--session_labels', '2020021001',
        '--run_labels', '01',
        '--in_file','/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/sub-Nl311f9/ses-2020021001/func/sub-Nl311f9_ses-2020021001_task-rs_run-02_bold.nii.gz',
        '--in_file_entities_labels_string', 'acq-TurboRARE_T2w.nii.gz',
        #'--in_file_entities_labels_string', 'task-rs_bold.nii.gz',
        '--ants_be_template_subject', 'AnatTemplate',
        '--ants_be_template_probability_mask_subject', 'AnatTemplateProbabilityMask',
        '--ants_be_template_desc', '0p15x0p15x0p55mm20200804',
        '--ants_be_template_probability_mask_desc', '0p15x0p15x0p55mm20200804',
        '--brain_extract_method', 'REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK',
        '--nipype_processing_dir', './brain_extract_test',
        '--keep_unnecessary_outputs',
    ]

    parser_groups = CFMMParserGroups(configargparse.ArgumentParser())

    config_file_obj = CFMMConfig()
    config_file_obj.populate_parser_groups(parser_groups)

    nipype_run_arguments = NipypeRunArguments()
    nipype_run_arguments.populate_parser_groups(parser_groups)

    tmp = MouseBrainExtraction4DBIDS()
    #print_component(tmp)
    tmp.populate_parser_groups(parser_groups)
    parser_groups.parser.print_help()



    parsed_namespace = config_file_obj.parse_args(parser_groups, cmd_args)
    parsed_dict = vars(parsed_namespace)

    nipype_run_arguments.populate_parameters(parsed_dict)
    tmp.populate_parameters(parsed_dict)
    tmp.validate_parameters()

    # print_component(tmp)

    wf = tmp.create_workflow()
    #wf.write_graph(graph2use='flat')

    logger.info('Starting Program!')

    nipype_run_arguments.run_workflow(wf)

    logger.info('Finished Program!')







