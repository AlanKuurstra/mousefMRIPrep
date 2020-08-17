from workflows.CFMMBase import CFMMParserArguments, CFMMInterface
from nipype.interfaces.ants import Registration, N4BiasFieldCorrection, ApplyTransforms, Atropos
from niworkflows.interfaces.ants import ThresholdImage


class AntsArguments(CFMMParserArguments):
    group_name="ANTs Arguments"
    flag_prefix = 'antsarg_'
    def add_parser_arguments(self):
        # add parser arguments
        self.add_parser_argument('interpolation',
                                 default='Linear',
                                 help="Interpolation method in antsRegistration and antsApplyTransforms.")
        self.add_parser_argument('high_precision',
                                 action='store_true',
                                 help="Use double precision instead of float in antsRegistration antsApplyTransform.")
        self.add_parser_argument('gzip_large_images',
                                 action='store_true',
                                 help="If true, gzip large images. Gzip saves space but I/O operations take longer.")


class CFMMN4BiasFieldCorrection(CFMMInterface):
    group_name = 'N4 Correction'
    flag_prefix = 'n4_'
    def __init__(self, *args, **kwargs):
        super().__init__(N4BiasFieldCorrection, *args, **kwargs)

    def add_parser_arguments(self):
        super().add_parser_arguments()
        self.modify_parser_argument('bspline_fitting_distance',
                                    'help',
                                    'Distance between spline knots. Should be appropriate for object size.')


class CFMMAntsRegistration(CFMMInterface):
    group_name = 'ANTs Registration'
    flag_prefix = 'antsreg_'
    def __init__(self, *args, **kwargs):
        super().__init__(Registration, *args, **kwargs)

    def add_parser_arguments(self):
        super().add_parser_arguments()


class CFMMApplyTransforms(CFMMInterface):
    group_name='Apply Transforms'
    flag_prefix = 'applytf_'
    def __init__(self, *args, **kwargs):
        super().__init__(ApplyTransforms, *args, **kwargs)

    def add_parser_arguments(self):
        super().add_parser_arguments()
        self.modify_parser_argument('dimension', 'default', "3")
        self.modify_parser_argument('output_image', 'default', "'deformed_mask.nii'")
        self.modify_parser_argument('interpolation', 'default', "'Linear'")
        self.modify_parser_argument('default_value', 'default', "0")
        # apply_transform.inputs.transforms = ['ants_Warp.nii.gz', 'trans.mat']
        # apply_transform.inputs.invert_transform_flags = [False, False]
        # apply_transform.inputs.transforms = composite_transform


class CFMMThresholdImage(CFMMInterface):
    group_name = "Threshold Image"
    flag_prefix = 'thresh_'
    def __init__(self, *args, **kwargs):
        super().__init__(ThresholdImage, *args, **kwargs)

    def add_parser_arguments(self):
        super().add_parser_arguments()
        self.modify_parser_argument('dimension', 'default', "3")
        self.modify_parser_argument('th_low', 'default', "0.5")
        self.modify_parser_argument('th_high', 'default', "1.0")
        self.modify_parser_argument('inside_value', 'default', "1")
        self.modify_parser_argument('outside_value', 'default', "0")

class MouseN4BiasFieldCorrection(CFMMN4BiasFieldCorrection):
    """
    Wrapper class for CFMMN4BiasFieldCorrection with default parameter values suitable for mouse brains.
    """
    def add_parser_arguments(self):
        """
        Modifies default parameter values.
        """
        super().add_parser_arguments()
        self.modify_parser_argument('bspline_fitting_distance', 'default', 20.0)
        self.modify_parser_argument('dimension', 'default', 3)
        self.modify_parser_argument('save_bias', 'default', False)
        self.modify_parser_argument('copy_header', 'default', True)
        self.modify_parser_argument('n_iterations', 'default', [50] * 4)
        self.modify_parser_argument('convergence_threshold', 'default', 1e-7)
        self.modify_parser_argument('shrink_factor', 'default', 4)

class MouseAntsRegistrationBE(CFMMAntsRegistration):
    """
    Wrapper class for CFMMAntsRegistration with default parameter values suitable for mouse brains.
    """
    def add_parser_arguments(self):
        """
        Wrapper class for CFMMBse with default parameter values suitable for mouse brains.
        """
        super().add_parser_arguments()
        # note: the type conversion function you provided to argparse is only called on string defaults
        # therefore a default of 3 will set the argument to 3 (both integers)
        # a default of '3' will go through the convert function and in our case convert_argparse_using_eval.convert()'s
        # eval() function will convert the string to integer 3
        # it is important to to include two sets of quotes if the default value is supposed to be a string
        # so that after the eval function, it will still be a string
        self.modify_parser_argument('output_transform_prefix', 'default', "'output_'")
        self.modify_parser_argument('dimension', 'default', 3)
        self.modify_parser_argument('initial_moving_transform_com', 'default', "1")
        self.modify_parser_argument('transforms', 'default', "['Affine', 'SyN']")
        # transform_parameters:
        # gradient step
        # updateFieldVarianceInVoxelSpace - smooth the deformation computed on the "updated" gradient field before this is added to previous deformations to form the "total" gradient field
        # totalFieldVarianceInVoxelSpace - smooth the deformation computed on the "total" gradient field
        self.modify_parser_argument('transform_parameters', 'default', "[(0.1,), (0.1, 3.0, 0.0)]")
        self.modify_parser_argument('number_of_iterations', 'default', "[[10, 5, 3], [10, 5, 3]]")
        # transform for each stage vs composite for entire warp
        self.modify_parser_argument('write_composite_transform', 'default', "True")
        # combines adjacent transforms when possible
        self.modify_parser_argument('collapse_output_transforms', 'default', "False")
        # ants_reg.inputs.initialize_transforms_per_stage = False #seems to be for initializing linear transforms only
        # using CC when atlas was made using same protocol
        self.modify_parser_argument('metric', 'default', "['CC'] * 2")
        # weight used if you do multimodal registration. Default is 1 (value ignored currently by ANTs)
        self.modify_parser_argument('metric_weight', 'default', "[1] * 2")
        # radius for CC between 2-5
        self.modify_parser_argument('radius_or_number_of_bins', 'default', "[5] * 2")
        # not entirely sure why we don't need to specify sampling strategy and percentage for non-linear syn registration
        # but I'm just following ANTs examples
        self.modify_parser_argument('sampling_strategy', 'default', "['Regular', None]")
        self.modify_parser_argument('sampling_percentage', 'default', "[0.5, None]")
        # use a negative number if you want to do all iterations and never exit
        self.modify_parser_argument('convergence_threshold', 'default', "[1.e-8,1.e-9]")
        # if the cost hasn't changed by convergence threshold in the last window size iterations, exit loop
        self.modify_parser_argument('convergence_window_size', 'default', "[10] * 2")
        self.modify_parser_argument('smoothing_sigmas', 'default', "[[0.3, 0.15, 0], [0.3, 0.15, 0]]")
        # we use mm instead of vox because we don't have isotropic voxels
        self.modify_parser_argument('sigma_units', 'default', "['mm'] * 2  ")
        self.modify_parser_argument('shrink_factors', 'default', "[[3, 2, 1], [3, 2, 1]]")
        # estimate the learning rate step size only at the beginning of each level. Does this override the value chosen in transform_parameters?
        self.modify_parser_argument('use_estimate_learning_rate_once', 'default', "[True,True]")
        self.modify_parser_argument('use_histogram_matching', 'default', "[True, True]")
        self.modify_parser_argument('output_warped_image', 'default', "'output_warped_image.nii.gz'")
