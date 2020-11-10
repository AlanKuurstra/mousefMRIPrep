import argparse
from workflows.CFMMWorkflow import CFMMWorkflow
from workflows.CFMMBIDS import CFMMBIDSWorkflowMixin, BIDSInputExternalSearch, CMDLINE_VALUE
from workflows.CFMMAnts import AntsDefaultArguments, CFMMAntsRegistration
from workflows.MouseBrainExtraction import MouseBrainExtraction
from workflows.CFMMCommon import NipypeWorkflowArguments, delistify
from nipype.pipeline import engine as pe
from nipype.interfaces.utility import Function


class MouseAnatToAtlasANTs(CFMMAntsRegistration):
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
        self._modify_parameter('transforms', 'default', "['Translation', 'Rigid', 'Affine', 'SyN']")
        # transform_parameters:
        # gradient step
        # updateFieldVarianceInVoxelSpace - smooth the deformation computed on the "updated" gradient field before this is added to previous deformations to form the "total" gradient field
        # totalFieldVarianceInVoxelSpace - smooth the deformation computed on the "total" gradient field
        self._modify_parameter('transform_parameters', 'default', "[(0.1,)] * 3 + [(0.1, 3.0, 0.0)]")

        # transform for each stage vs composite for entire warp
        self._modify_parameter('write_composite_transform', 'default', "True")
        # combines adjacent transforms when possible
        self._modify_parameter('collapse_output_transforms', 'default', "False")
        self._modify_parameter('initialize_transforms_per_stage', 'default',
                               "True")  # seems to be for initializing linear transforms only

        self._modify_parameter('metric', 'default', "['MI'] * 3 + [['MI', 'CC']]")
        self._modify_parameter('number_of_iterations', 'default', "[[10000, 10000, 10000]] * 3 + [[100, 100, 100, 300]]")
        # weight used if you do multimodal registration. Default is 1 (value ignored currently by ANTs)
        self._modify_parameter('metric_weight', 'default', "[1] * 3 + [[0.5, 0.5]]")
        # radius for CC between 2-5
        self._modify_parameter('radius_or_number_of_bins', 'default', "[32] * 3 + [[32, 4]]")
        # not entirely sure why we don't need to specify sampling strategy and percentage for non-linear syn registration
        # but I'm just following ANTs examples
        self._modify_parameter('sampling_strategy', 'default', "['Regular'] * 2 + [None, [None, None]]")
        # self._modify_parameter('sampling_percentage', 'default', "")
        self._modify_parameter('use_histogram_matching', 'default', "[False] * 3 + [True]")

        # use a negative number if you want to do all iterations and never exit
        self._modify_parameter('convergence_threshold', 'default', "[1.e-9] * 4")
        # if the cost hasn't changed by convergence threshold in the last window size iterations, exit loop
        self._modify_parameter('convergence_window_size', 'default', "[20] * 3 + [5]")
        self._modify_parameter('smoothing_sigmas', 'default', "[[0.35, 0.2, 0.03]] * 3 + [[0.39, 0.3, 0.1, 0.03]]")
        # we use mm instead of vox because we don't have isotropic voxels
        self._modify_parameter('sigma_units', 'default', "['mm'] * 4  ")
        self._modify_parameter('shrink_factors', 'default', "[[3, 2, 1]] * 3 + [[3, 2, 1, 1]]")
        # estimate the learning rate step size only at the beginning of each level. Does this override the value chosen in transform_parameters?
        self._modify_parameter('use_estimate_learning_rate_once', 'default', "[True] * 4")

        self._modify_parameter('output_warped_image', 'default', "'output_warped_image.nii.gz'")


def reduce_smoothing_for_large_atlas(smoothing_sigmas, smallest_dim_size):
    # antsRegistration fails during smoothing step for high resolution atlases
    # reduce smoothign sigma if it will create a kernel that is too large in pixel units
    import numpy as np
    largest_num_pxls_ants_can_handle = 26
    corrected_smoothing_sigmas = [np.array(x) for x in smoothing_sigmas]
    corrected_smoothing_sigmas_pxls = [np.array(x) / smallest_dim_size for x in smoothing_sigmas]
    for x, y in zip(corrected_smoothing_sigmas_pxls, corrected_smoothing_sigmas):
        y[x > largest_num_pxls_ants_can_handle] = largest_num_pxls_ants_can_handle * smallest_dim_size
    corrected_smoothing_sigmas = [list(x) for x in corrected_smoothing_sigmas]
    return corrected_smoothing_sigmas, smoothing_sigmas


def get_shrink_factors(smoothing_sigmas, smallest_dim_size):
    # automatically calculate shrink factors that are 2.5x the smoothing sigma
    import numpy as np
    return [list((np.array(sigmas) / smallest_dim_size * 1.5 + 1).astype(int)) for sigmas in smoothing_sigmas]


def get_atlas_smallest_dim_spacing(atlas_file_location):
    # return the smallest spacing (in mm) of any given dimension
    # useful for determining the largest number of pixels a physical distance will be
    import numpy as np
    import nibabel as nib
    return np.array(nib.load(atlas_file_location).header['pixdim'][1:4]).min()


class MouseAnatToAtlas(CFMMWorkflow):
    group_name = 'Anat Preprocessing and Registration'
    flag_prefix = 'anat_'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nipype = NipypeWorkflowArguments(owner=self, exclude_list=['nthreads_mapnode', 'mem_gb_mapnode'])
        self.ants_args = AntsDefaultArguments(owner=self)
        self.be = MouseBrainExtraction(owner=self, exclude_list=['in_file', 'in_file_mask'])
        # to help the end user, this pipeline will calculate reasonable shrink factors based on the user's chosen
        # smoothing sigmas.
        self.ants_reg = MouseAnatToAtlasANTs(owner=self, exclude_list=['shrink_factors'])
        self.ants_reg.get_parameter('float').default_provider = self.ants_args.get_parameter('float')
        self.ants_reg.get_parameter('interpolation').default_provider = self.ants_args.get_parameter('interpolation')
        self.ants_reg.get_parameter('num_threads').default_provider = self.nipype.get_parameter('nthreads_node')

        self.outputs = ['anat_to_atlas', 'anat_to_atlas_composite_transform', 'brain_mask']

    def _add_parameters(self):
        self._add_parameter('in_file',
                            help='Explicitly specify location of the input anatomical for atlas registration.',
                            iterable=True)
        self._add_parameter('in_file_mask',
                            help='Explicitly specify location of the input anatomical mask for atlas registration.',
                            iterable=True)
        self._add_parameter('atlas',
                            help='Explicitly specify location of the atlas to be registered to.')
        self._add_parameter('atlas_mask',
                            help='Explicitly specify location of the atlas mask.')
        self._add_parameter('no_mask_anat2atlas',
                            action='store_true',
                            help="Don't use masks during the anatomical to atlas registration.")

    def create_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        be_workflow = self.be.create_workflow()
        ants_reg = self.ants_reg.get_node('ants_reg')

        atlas_smallest_dim = pe.Node(
            Function(input_names=["atlas_file_location"], output_names=["smallest_dim_size"],
                     function=get_atlas_smallest_dim_spacing),
            name="atlas_smallest_dim")

        correct_smooth_factors = pe.Node(
            Function(input_names=["smoothing_sigmas", "smallest_dim_size"],
                     output_names=["corrected_smoothing_sigmas", "original_smoothing_sigmas"],
                     function=reduce_smoothing_for_large_atlas), name="correct_smooth_factors")
        correct_smooth_factors.inputs.smoothing_sigmas = self.ants_reg.get_parameter('smoothing_sigmas').user_value

        calc_shrink_factors = pe.Node(
            Function(input_names=["smoothing_sigmas", "smallest_dim_size"], output_names=["shrink_factors"],
                     function=get_shrink_factors), name="shrink_factors")

        inputnode, outputnode, wf = self.get_io_and_workflow()

        wf.connect([
            (inputnode, be_workflow, [('in_file', 'inputnode.in_file')]),
            (inputnode, be_workflow, [('in_file_mask', 'inputnode.in_file_mask')]),
            (be_workflow, ants_reg, [('outputnode.out_file_n4_corrected', 'moving_image')]),
            (inputnode, ants_reg, [('atlas', 'fixed_image')]),

            (inputnode, atlas_smallest_dim, [('atlas', 'atlas_file_location')]),
            (atlas_smallest_dim, correct_smooth_factors, [('smallest_dim_size', 'smallest_dim_size')]),
            (correct_smooth_factors, calc_shrink_factors, [('original_smoothing_sigmas', 'smoothing_sigmas')]),
            (atlas_smallest_dim, calc_shrink_factors, [('smallest_dim_size', 'smallest_dim_size')]),
            (correct_smooth_factors, ants_reg, [('corrected_smoothing_sigmas', 'smoothing_sigmas')]),
            (calc_shrink_factors, ants_reg, [('shrink_factors', 'shrink_factors')]),
            (ants_reg, outputnode, [('warped_image', 'anat_to_atlas'),
                                    ('composite_transform', 'anat_to_atlas_composite_transform'), ]),
            (be_workflow, outputnode, [('outputnode.out_file_mask', 'brain_mask')]),

        ])

        if not self.get_parameter('no_mask_anat2atlas').user_value:
            wf.connect([
                (inputnode, ants_reg, [('atlas_mask', 'fixed_image_mask')]),
                (be_workflow, ants_reg, [('outputnode.out_file_mask', 'moving_image_mask')]),
            ])

        return wf


class MouseAnatToAtlasBIDS(MouseAnatToAtlas, CFMMBIDSWorkflowMixin):
    def __init__(self, *args, **kwargs):
        # can this be a function in bids mixer?
        super().__init__(*args, **kwargs)

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
                                                        'anat_to_atlas': 'AnatToAtlas',
                                                        'anat_to_atlas_composite_transform': 'AnatToAtlasTransform',
                                                        'brain_mask': 'BrainMask',
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
        # should the exclude check happen in _modify_parameters??
        # doing that would hide potential mistakes from the user. if they make a typo in the parameter, it would just
        # be ignored - the modification to the parameter would not be made and the user might not notice
        if 'in_file_mask' not in self.exclude_list:
            self._modify_parameter('in_file_mask_desc', 'default', "'ManualBrainMask'")

    def create_workflow(self):
        wf = super().create_workflow()
        self.add_bids_to_workflow(wf)
        return wf


if __name__ == "__main__":
    bids_args = [
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids'",
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives'",
        "'participant'",
        '--input_derivatives_dirs',
        "['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives']",

        '--in_file_base_bids_string', "'acq-TurboRARE_T2w.nii.gz'",
        '--in_file_subject', "'Nl311f9'",
        '--in_file_session', "'2020021001'",

        # for masking in_file through registration of template
        '--be_ants_be_template',
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/TemplatesAndProbabilityMasks/sub-AnatTemplate_acq-TurboRARE_desc-0p15x0p15x0p55mm20200804_T2w.nii.gz'",
        '--be_ants_be_template_probability_mask',
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/TemplatesAndProbabilityMasks/sub-AnatTemplateProbabilityMask_acq-TurboRARE_desc-0p15x0p15x0p55mm20200804_T2w.nii.gz'",
        # atlas to register using mask created by template
        '--atlas',
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/Atlases/sub-AMBMCc57bl6_desc-ModelDownsampled.nii.gz'",
        '--atlas_mask',
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/Atlases/sub-AMBMCc57bl6_desc-ModelDownsampledBrainMask.nii.gz'",
        '--antsarg_float',
        '--be_brain_extract_method', 'REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK',
        #'--be_brain_extract_method', 'BRAINSUITE',
    ]
    tmp = MouseAnatToAtlasBIDS()
    tmp.run_bids(bids_args)
