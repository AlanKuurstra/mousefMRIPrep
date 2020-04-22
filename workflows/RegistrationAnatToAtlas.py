from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from multiprocessing import cpu_count
from nipype.interfaces.ants import Registration, N4BiasFieldCorrection, ApplyTransforms, Atropos
from niworkflows.interfaces.ants import ThresholdImage
from nipype.interfaces.fsl import ApplyMask
import nibabel as nib
from nipype.interfaces.utility import Function


def reduce_smoothing_for_large_atlas(smoothing_sigmas,smallest_dim_size):
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

def get_shrink_factors(smoothing_sigmas,smallest_dim_size):
    # automatically calculate shrink factors that are 2.5x the smoothing sigma
    import numpy as np
    return [list((np.array(sigmas) / smallest_dim_size * 1.5 + 1).astype(int)) for sigmas in smoothing_sigmas]

def get_atlas_smallest_dim_spacing(atlas_file_location):
    # return the smallest spacing in any given dimension
    # useful for determining the largest number of pixels an physical distance will be
    import numpy as np
    import nibabel as nib
    return np.array(nib.load(atlas_file_location).header['pixdim'][1:4]).min()
    
def init_anat_to_atlas_registration(
        name = 'register_anat_to_atlas',
        mask = True,
        reduce_to_float_precision=False,
        interpolation='Linear',
        omp_nthreads = None,
        mem_gb = 3.0,
):

    wf = pe.Workflow(name)

    if omp_nthreads is None or omp_nthreads < 1:
        omp_nthreads = cpu_count()

    inputnode = pe.Node(niu.IdentityInterface(fields=['anat', 'anat_mask', 'atlas', 'atlas_mask']), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(fields=['anat_to_atlas','anat_to_atlas_composite_transform']),name='outputnode')

    atlas_smallest_dim = pe.Node(
        Function(input_names=["atlas_file_location"], output_names=["smallest_dim_size"], function=get_atlas_smallest_dim_spacing),
        name="atlas_smallest_dim")

    correct_smooth_factors = pe.Node(
        Function(input_names=["smoothing_sigmas", "smallest_dim_size"], output_names=["corrected_smoothing_sigmas","original_smoothing_sigmas"],
                 function=reduce_smoothing_for_large_atlas), name="correct_smooth_factors")
    correct_smooth_factors.inputs.smoothing_sigmas = [[0.35, 0.25, 0.1]] * 3 + [[0.39, 0.3, 0.2, 0.1]]
    
    calc_shrink_factors = pe.Node(Function(input_names=["smoothing_sigmas","smallest_dim_size"], output_names=["shrink_factors"], function=get_shrink_factors), name="shrink_factors")

    ants_method = 'Mix'  # 'Mix', 'MI', 'CC'
    ants_reg = pe.Node(interface=Registration(), name='antsRegistration',n_procs=omp_nthreads,mem_gb=mem_gb)
    ants_reg.inputs.output_transform_prefix = "output_"
    ants_reg.inputs.initial_moving_transform_com = 1  # seems to be necessary. initial translation alignment by geometric center of the images (=0), the image intensities (=1), or the origin of the images (=2)
    ants_reg.inputs.dimension = 3
    ants_reg.inputs.float = reduce_to_float_precision
    ants_reg.inputs.interpolation = interpolation
    ants_reg.inputs.transforms = ['Translation', 'Rigid', 'Affine', 'SyN']
    ants_reg.inputs.transform_parameters = [(0.1,)] * 3 + [(0.1, 3.0, 0.0)]
    # gradient step
    # updateFieldVarianceInVoxelSpace - smooth the deformation computed on the "updated" gradient field before this is added to previous deformations to form the "total" gradient field
    # totalFieldVarianceInVoxelSpace - smooth the deformation computed on the "total" gradient field
    ants_reg.inputs.write_composite_transform = True  # transform for each stage vs composite for entire warp
    ants_reg.inputs.collapse_output_transforms = False  # combines adjacent transforms when possible
    ants_reg.inputs.initialize_transforms_per_stage = True  # seems to be for initializing linear transforms only

    if ants_method == 'MI':
        ants_reg.inputs.number_of_iterations = [[10000, 10000, 10000]] * 3 + [[50, 25, 10]]
        ants_reg.inputs.metric = ['MI'] * 4  # MI same as Mattes
        ants_reg.inputs.radius_or_number_of_bins = [32] * 4  # histogram bins for MI
        ants_reg.inputs.sampling_strategy = ['Regular'] * 2 + [None, None]
        ants_reg.inputs.use_histogram_matching = [False] * 4
    elif ants_method == 'CC':
        ants_reg.inputs.number_of_iterations = [[10, 5, 3]] * 3 + [[10, 5, 3]]
        ants_reg.inputs.metric = ['CC'] * 4
        ants_reg.inputs.radius_or_number_of_bins = [4] * 4  # radius for CC between 2-5
        ants_reg.inputs.sampling_strategy = ['Regular'] * 2 + [None, None]
        ants_reg.inputs.use_histogram_matching = [True] * 4
    elif ants_method == 'Mix':
        ants_reg.inputs.number_of_iterations = [[10000, 10000, 10000]] * 3 + [[100, 100, 100, 300]]
        ants_reg.inputs.metric = ['MI'] * 3 + [['MI', 'CC']]
        ants_reg.inputs.metric_weight = [1] * 3 + [[0.5, 0.5]]  # weight used if you do multimodal registration
        ants_reg.inputs.radius_or_number_of_bins = [32] * 3 + [[32, 4]]
        ants_reg.inputs.sampling_strategy = ['Regular'] * 2 + [None, [None, None]]
        ants_reg.inputs.use_histogram_matching = [False] * 3 + [True]  # True is default

    ants_reg.inputs.convergence_threshold = [1.e-8] * 4  # use a negative number if you want to do all iterations and never exit
    ants_reg.inputs.convergence_window_size = [20] * 3 + [5]  # if the cost hasn't changed by convergence threshold in the last window size iterations, exit loop
    ants_reg.inputs.sigma_units = ['mm'] * 4  # we use mm instead of vox because we don't have isotropic voxels
    ants_reg.inputs.use_estimate_learning_rate_once = [True] * 4  # estimate the learning rate step size only at the beginning of each level. Does this override the value chosen in transform_parameters?
    ants_reg.inputs.output_warped_image = 'output_warped_image.nii.gz'
    ants_reg.n_procs = omp_nthreads

    # note, in neurodocker's precompiled version of ants, the -v option gives version instead of making output verbose
    # this is different than apt's ants version and nipype's presumed behaviour and causes failures
    # ants_reg.inputs.verbose = True

    wf.connect([
        (inputnode, ants_reg, [('atlas', 'fixed_image')]),
        (inputnode, ants_reg, [('anat', 'moving_image')]),
        (inputnode, atlas_smallest_dim, [('atlas', 'atlas_file_location')]),

        (atlas_smallest_dim, correct_smooth_factors, [('smallest_dim_size', 'smallest_dim_size')]),

        (correct_smooth_factors, calc_shrink_factors, [('original_smoothing_sigmas', 'smoothing_sigmas')]),
        (atlas_smallest_dim, calc_shrink_factors, [('smallest_dim_size', 'smallest_dim_size')]),

        (correct_smooth_factors, ants_reg, [('corrected_smoothing_sigmas', 'smoothing_sigmas')]),
        (calc_shrink_factors, ants_reg, [('shrink_factors', 'shrink_factors')]),
        
        (ants_reg, outputnode, [('composite_transform', 'anat_to_atlas_composite_transform')]),
        (ants_reg, outputnode, [('warped_image', 'anat_to_atlas')]),
    ])
    if mask:
        wf.connect([
            (inputnode, ants_reg, [('atlas_mask', 'fixed_image_mask')]),
            (inputnode, ants_reg, [('anat_mask', 'moving_image_mask')]),
        ])

    return wf

if __name__=='__main__':
    wf = init_anat_to_atlas_registration()
    wf.inputs.inputnode.anat = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/sub-NL311F9/ses-2020021001/anat/sub-NL311F9_ses-2020021001_acq-TurboRARE_run-01_T2w.nii.gz'
    wf.inputs.inputnode.atlas = '/home/akuurstr/Desktop/Esmin_mouse_registration/test/AMBMC_model.nii.gz'
    wf.inputs.inputnode.atlas_mask = '/home/akuurstr/Desktop/Esmin_mouse_registration/test/AMBMC_model_mask.nii.gz'
    wf.base_dir = 'atlas_registration_wf'
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run()


    # GaussianOperator (0x55d5bb2f9548): Kernel size has exceeded the specified maximum width of 32 and has been truncated to 33 elements.  You can raise the maximum width using the SetMaximumKernelWidth method.
    # https://github.com/ANTsX/ANTs/issues/187
