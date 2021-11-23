from workflows.CFMMAnts import CFMMAntsRegistration
from workflows.MotionCorrection import MotionCorrection

class MouseAntsRegistrationMC(CFMMAntsRegistration):
    """
    Wrapper class for CFMMAntsRegistration with default parameter values suitable for mouse brains.
    """
    def _add_parameters(self):
        """
        Wrapper class for CFMMBse with default parameter values suitable for mouse brains.
        """
        # follow https://github.com/ANTsX/ANTs/blob/master/Scripts/antsMotionCorrExample
        # could not use GaussianDisplacementField for non-linear because nipype has a bug:
        # GaussianDisplacementField does output an inverse transform, but nipype expects one and produces
        # ERROR:
        # traits.trait_errors.TraitError: Each element of the 'reverse_transforms' trait of a RegistrationOutputSpec
        # instance must be a pathlike object or string representing an existing file, but a value of
        # '.../mc_ants_reg/mapflow/_mc_ants_reg356/output_1InverseWarp.nii.gz' <class 'str'> was specified.
        # SyN does not expect an inverse transform and can be used instead.

        super()._add_parameters()
        # note: the type conversion function you provided to argparse is only called on string defaults
        # therefore a default of 3 will set the argument to 3 (both integers)
        # a default of '3' will go through the convert function and in our case convert_argparse_using_eval.convert()'s
        # eval() function will convert the string to integer 3
        # it is important to to include two sets of quotes if the default value is supposed to be a string
        # so that after the eval function, it will still be a string


        self._modify_parameter('dimension', 'default', 3)

        # transform_parameters:
        # gradient step
        # updateFieldVarianceInVoxelSpace - smooth the deformation computed on the "updated" gradient field before this is added to previous deformations to form the "total" gradient field
        # totalFieldVarianceInVoxelSpace - smooth the deformation computed on the "total" gradient field
        # self._modify_parameter('transforms', 'default', "['Affine', 'GaussianDisplacementField']")
        # self._modify_parameter('transform_parameters', 'default', "[(0.005,),(0.5, 3.0, 0.5)]")
        self._modify_parameter('transforms', 'default', "['Affine', 'SyN']")
        self._modify_parameter('transform_parameters', 'default', "[(0.005,),(0.5, 3.0, 0.5)]")


        # transform for each stage vs composite for entire warp
        self._modify_parameter('write_composite_transform', 'default', "False")

        self._modify_parameter('metric', 'default', "['GC','CC']")

        #self._modify_parameter('number_of_iterations', 'default', "[[20],[20]]")
        self._modify_parameter('number_of_iterations', 'default', "[[20],[5]]")


        # weight used if you do multimodal registration. Default is 1 (value ignored currently by ANTs)
        self._modify_parameter('metric_weight', 'default', "[1,1]")
        # radius for CC between 2-5
        self._modify_parameter('radius_or_number_of_bins', 'default', "[2] * 2")
        # not entirely sure why we don't need to specify sampling strategy and percentage for non-linear syn registration
        # but I'm just following ANTs examples
        self._modify_parameter('sampling_strategy', 'default', "['Regular',None]")
        self._modify_parameter('sampling_percentage', 'default', "[0.2, None]")
        self._modify_parameter('use_histogram_matching', 'default', "[True]*2")

        # use a negative number if you want to do all iterations and never exit
        self._modify_parameter('convergence_threshold', 'default', "[1.e-9] * 2")
        # if the cost hasn't changed by convergence threshold in the last window size iterations, exit loop
        self._modify_parameter('convergence_window_size', 'default', "[5,5]")
        self._modify_parameter('smoothing_sigmas', 'default', "[[0],[0]]")

        # we use mm instead of vox because we don't have isotropic voxels
        self._modify_parameter('sigma_units', 'default', "['mm'] * 2  ")
        self._modify_parameter('shrink_factors', 'default', "[[1],[1]]")
        # estimate the learning rate step size only at the beginning of each level. Does this override the value chosen in transform_parameters?
        self._modify_parameter('use_estimate_learning_rate_once', 'default', "[False] * 2")
        self._modify_parameter('output_transform_prefix', 'default', "'output_'")
        self._modify_parameter('output_warped_image', 'default', "'output_warped_image.nii.gz'")
        self._modify_parameter('verbose', 'default', "True")

class MouseMotionCorrection(MotionCorrection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._remove_subcomponent_attribute('mc_ants_reg')
        self.mc_ants_reg = MouseAntsRegistrationMC(owner=self)
        self.mc_ants_reg.get_parameter('float').default_provider = self.ants_args.get_parameter('float')
        self.mc_ants_reg.get_parameter('interpolation').default_provider = self.ants_args.get_parameter('interpolation')

