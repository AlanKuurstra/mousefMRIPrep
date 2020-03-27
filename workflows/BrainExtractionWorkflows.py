from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from multiprocessing import cpu_count
from nipype.interfaces.ants import Registration, N4BiasFieldCorrection, ApplyTransforms, Atropos
from niworkflows.interfaces.ants import ThresholdImage
import nipype.interfaces.brainsuite as bs
from nipype.interfaces.fsl import ImageMaths, CopyGeom, ApplyMask
from enum import Enum

class BrainExtractMethod(Enum):
    BRAINSUITE = 1
    REGISTRATION_WITH_MASK = 2
    REGISTRATION_NO_MASK = 3
    NO_BRAIN_EXTRACTION = 4

def init_ants_brain_extraction_wf(
        name='ants_brain_extraction_wf',
        omp_nthreads=None,
        mem_gb=3.0,
        n4_bspline_fitting_distance=20
):
    wf = pe.Workflow(name)

    if omp_nthreads is None or omp_nthreads < 1:
        omp_nthreads = cpu_count()

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'in_file_mask', 'template', 'template_probability_mask']),
                        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file_n4_corrected_brain_extracted','out_file_n4_corrected', 'out_file_mask']),
        name='outputnode')

    inu_n4 = pe.Node(
        N4BiasFieldCorrection(
            dimension=3, save_bias=False, copy_header=True,
            n_iterations=[50] * 4, convergence_threshold=1e-7, shrink_factor=4,
            bspline_fitting_distance=n4_bspline_fitting_distance),
        n_procs=omp_nthreads, name='inu_n4', iterfield=['input_image'])

    ants_reg = pe.Node(interface=Registration(), name='antsRegistration')
    ants_reg.inputs.output_transform_prefix = "output_"
    # ants_reg.inputs.initial_moving_transform = 'trans.mat'
    ants_reg.inputs.dimension = 3
    ants_reg.inputs.transforms = ['Affine', 'SyN']
    ants_reg.inputs.transform_parameters = [(0.1,), (0.1, 3.0, 0.0)]
    # gradient step
    # updateFieldVarianceInVoxelSpace - smooth the deformation computed on the "updated" gradient field before this is added to previous deformations to form the "total" gradient field
    # totalFieldVarianceInVoxelSpace - smooth the deformation computed on the "total" gradient field
    # ants_reg.inputs.number_of_iterations = [[500, 500, 500], [50, 25, 10]] #500 for Mattes/MI and 50 for CC, also dependent on gradient step
    ants_reg.inputs.number_of_iterations = [[10, 5, 3], [10, 5, 3]]
    ants_reg.inputs.write_composite_transform = True  # transform for each stage vs composite for entire warp
    ants_reg.inputs.collapse_output_transforms = False  # combines adjacent transforms when possible
    # ants_reg.inputs.initialize_transforms_per_stage = False #seems to be for initializing linear transforms only
    # ants_reg.inputs.metric = ['Mattes']*2
    ants_reg.inputs.metric = ['CC'] * 2  # using CC because atlas was made from same protocol
    ants_reg.inputs.metric_weight = [
                                        1] * 2  # weight used if you do multimodal registration. Default is 1 (value ignored currently by ANTs)
    # ants_reg.inputs.radius_or_number_of_bins = [32]*2 # histogram bins for MI
    ants_reg.inputs.radius_or_number_of_bins = [5] * 2  # radius for CC between 2-5
    ants_reg.inputs.sampling_strategy = ['Regular', None]  # Random vs Regular
    ants_reg.inputs.sampling_percentage = [0.3, None]
    # not entirely sure why we don't need to specify sampling strategy and percentage for non-linear syn registration
    ants_reg.inputs.convergence_threshold = [1.e-8,
                                             1.e-9]  # use a negative number if you want to do all iterations and never exit
    ants_reg.inputs.convergence_window_size = [
                                                  10] * 2  # if the cost hasn't changed by convergence threshold in the last window size iterations, exit loop
    ants_reg.inputs.smoothing_sigmas = [[0.3, 0.15, 0], [0.3, 0.15, 0]]
    ants_reg.inputs.sigma_units = ['mm'] * 2  # we use mm instead of vox because we don't have isotropic voxels
    ants_reg.inputs.shrink_factors = [[3, 2, 1], [3, 2, 1]]
    ants_reg.inputs.use_estimate_learning_rate_once = [True,
                                                       True]  # estimate the learning rate step size only at the beginning of each level. Does this override the value chosen in transform_parameters?
    ants_reg.inputs.use_histogram_matching = [True, True]  # This is the default
    ants_reg.inputs.output_warped_image = 'output_warped_image.nii.gz'
    ants_reg.n_procs = omp_nthreads

    apply_transform = pe.Node(interface=ApplyTransforms(), name='antsApplyTransforms')
    apply_transform.inputs.dimension = 3
    apply_transform.inputs.output_image = 'deformed_mask.nii'
    apply_transform.inputs.interpolation = 'Linear'
    apply_transform.inputs.default_value = 0
    # apply_transform.inputs.transforms = ['ants_Warp.nii.gz', 'trans.mat']
    # apply_transform.inputs.invert_transform_flags = [False, False]
    # apply_transform.inputs.transforms = composite_transform

    thr_brainmask = pe.Node(ThresholdImage(
        dimension=3, th_low=0.5, th_high=1.0, inside_value=1,
        outside_value=0), name='thr_brainmask')

    # USE ATROPOS TO CLEAN UP??

    apply_mask = pe.Node(ApplyMask(), iterfield=['in_file'], name='apply_mask')

    # atropos doesn't do so well on T2w mouse data
    atropos = pe.Node(Atropos(
        dimension=3,
        initialization='KMeans',
        number_of_tissue_classes=3,
        n_iterations=3,
        convergence_threshold=0.0,
        mrf_radius=[1, 1, 1],
        mrf_smoothing_factor=0.1,
        likelihood_model='Gaussian',
        use_random_seed=True),
        name='01_atropos', n_procs=omp_nthreads, mem_gb=3)

    wf.connect([
        (inputnode, inu_n4, [('in_file', 'input_image')]),

        (inu_n4, ants_reg, [('output_image', 'fixed_image')]),
        (inputnode, ants_reg, [('in_file_mask', 'fixed_image_mask')]),
        (inputnode, ants_reg, [('template', 'moving_image')]),

        (inputnode, apply_transform, [('template_probability_mask', 'input_image')]),
        (ants_reg, apply_transform, [('composite_transform', 'transforms')]),
        (inu_n4, apply_transform, [('output_image', 'reference_image')]),

        (apply_transform, thr_brainmask, [('output_image', 'input_image')]),

        (thr_brainmask, apply_mask, [('output_image', 'mask_file')]),
        (inu_n4, apply_mask, [('output_image', 'in_file')]),

        (apply_mask, outputnode, [('out_file', 'out_file_n4_corrected_brain_extracted')]),
        (inu_n4, outputnode, [('output_image', 'out_file_n4_corrected')]),
        (thr_brainmask, outputnode, [('output_image', 'out_file_mask')]),


        # (thr_brainmask, atropos, [('output_image', 'mask_image')]),
        # (apply_mask, atropos, [('out_file', 'intensity_images')]),
    ])
    return wf


def init_brainsuite_brain_extraction_wf(
        name='brainsuite_brain_extraction_wf',
        diffusionConstant=30,
        diffusionIterations=3,
        edgeDetectionConstant=0.55,
        radius=2,
        dilateFinalMask=True,
        omp_nthreads=None,
        mem_gb=3.0,
        n4_bspline_fitting_distance=20
):
    wf = pe.Workflow(name)

    if omp_nthreads is None or omp_nthreads < 1:
        omp_nthreads = cpu_count()

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', ]),
                        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file_n4_corrected_brain_extracted','out_file_n4_corrected','out_file_mask']),
        name='outputnode')

    inu_n4 = pe.Node(
        N4BiasFieldCorrection(
            dimension=3, save_bias=False, copy_header=True,
            n_iterations=[50] * 4, convergence_threshold=1e-7, shrink_factor=4,
            bspline_fitting_distance=n4_bspline_fitting_distance),
        n_procs=omp_nthreads, name='inu_n4', iterfield=['input_image'])

    bse = pe.Node(interface=bs.Bse(), name='BSE')
    bse.inputs.diffusionConstant = diffusionConstant
    bse.inputs.diffusionIterations = diffusionIterations
    bse.inputs.edgeDetectionConstant = edgeDetectionConstant
    bse.inputs.radius = radius
    bse.inputs.dilateFinalMask = dilateFinalMask
    bse.inputs.trim = False
    bse.inputs.noRotate = True

    # default behaviour of brainsuite is to rotate to LPI orientation
    # this can be overridden by using the noRotate option, however this option will create a nifti with inconsistent
    # qform and sform values.  To fix this, copy the header information from the original image to the mask using fsl.
    fix_bse_orientation = pe.Node(interface=CopyGeom(), name='fixBSEOrientation')

    # brainsuite outputs mask value as 255
    fix_bse_value = pe.Node(interface=ImageMaths(), name='fixBSEValue')
    fix_bse_value.inputs.op_string = '-div 255'

    apply_mask = pe.Node(ApplyMask(), iterfield=['in_file'], name='apply_mask')

    wf.connect([
        (inputnode, inu_n4, [('in_file', 'input_image')]),
        (inu_n4, bse, [('output_image', 'inputMRIFile')]),

        (inu_n4, fix_bse_orientation, [('output_image', 'in_file')]),
        (bse, fix_bse_orientation, [('outputMaskFile', 'dest_file')]),

        (fix_bse_orientation, fix_bse_value, [('out_file', 'in_file')]),

        (fix_bse_value, apply_mask, [('out_file', 'mask_file')]),
        (inu_n4, apply_mask, [('output_image', 'in_file')]),

        (apply_mask, outputnode, [('out_file', 'out_file_n4_corrected_brain_extracted')]),
        (inu_n4, outputnode, [('output_image', 'out_file_n4_corrected')]),
        (fix_bse_value, outputnode, [('out_file', 'out_file_mask')]),
    ])
    return wf

def init_n4_bias_and_brain_extraction_wf(
        brain_extraction_method,
        name='n4_bias_and_brain_extraction_wf',
        omp_nthreads=None,
        mem_gb=3.0,
        n4_bspline_fitting_distance=20,
        diffusionConstant=30,
        diffusionIterations=3,
        edgeDetectionConstant=0.55,
        radius=2,
        dilateFinalMask=True,
):
    wf = pe.Workflow(name)

    if omp_nthreads is None or omp_nthreads < 1:
        omp_nthreads = cpu_count()

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'in_file_mask', 'template', 'template_probability_mask']),
                        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file_n4_corrected_brain_extracted','out_file_n4_corrected', 'out_file_mask']),
        name='outputnode')

    if brain_extraction_method == BrainExtractMethod.BRAINSUITE:
        brain_extraction =init_brainsuite_brain_extraction_wf(
                                                            omp_nthreads=omp_nthreads,
                                                            mem_gb=mem_gb,
                                                            n4_bspline_fitting_distance=n4_bspline_fitting_distance,
                                                            diffusionConstant=diffusionConstant,
                                                            diffusionIterations=diffusionIterations,
                                                            edgeDetectionConstant=edgeDetectionConstant,
                                                            radius=radius,
                                                            dilateFinalMask=dilateFinalMask,
                                                            )
        wf.connect([
            (inputnode, brain_extraction, [('in_file', 'inputnode.in_file')]),
            (brain_extraction, outputnode, [('outputnode.out_file_n4_corrected_brain_extracted', 'out_file_n4_corrected_brain_extracted')]),
            (brain_extraction, outputnode, [('outputnode.out_file_n4_corrected', 'out_file_n4_corrected')]),
            (brain_extraction, outputnode, [('outputnode.out_file_mask', 'out_file_mask')]),
        ])

    elif brain_extraction_method == BrainExtractMethod.REGISTRATION_WITH_MASK:
        brain_extraction = init_ants_brain_extraction_wf(
            omp_nthreads=omp_nthreads,
            mem_gb=mem_gb,
            n4_bspline_fitting_distance=n4_bspline_fitting_distance,
        )
        wf.connect([
            (inputnode, brain_extraction, [('in_file', 'inputnode.in_file')]),
            (inputnode, brain_extraction, [('in_file_mask', 'inputnode.in_file_mask')]),
            (inputnode, brain_extraction, [('template', 'inputnode.template')]),
            (inputnode, brain_extraction, [('template_probability_mask', 'inputnode.template_probability_mask')]),
            (brain_extraction, outputnode, [('outputnode.out_file_n4_corrected_brain_extracted', 'out_file_n4_corrected_brain_extracted')]),
            (brain_extraction, outputnode, [('outputnode.out_file_n4_corrected', 'out_file_n4_corrected')]),
            (brain_extraction, outputnode, [('outputnode.out_file_mask', 'out_file_mask')]),
        ])
    elif brain_extraction_method == BrainExtractMethod.NO_BRAIN_EXTRACTION:
        inu_n4 = pe.Node(
            N4BiasFieldCorrection(
                dimension=3, save_bias=False, copy_header=True,
                n_iterations=[50] * 4, convergence_threshold=1e-7, shrink_factor=4,
                bspline_fitting_distance=n4_bspline_fitting_distance),
            n_procs=omp_nthreads, name='inu_n4', iterfield=['input_image'])
        wf.connect([
            (inputnode, inu_n4, [('in_file', 'input_image')]),
            (inu_n4, outputnode, [('output_image', 'out_file_n4_corrected')]),

        ])
    return wf

if __name__ == '__main__':
    in_file = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/sub-NL311F9/ses-2020021001/anat/sub-NL311F9_ses-2020021001_acq-TurboRARE_run-01_T2w.nii.gz'
    in_mask = 'doesnotexist'
    template = '/softdev/akuurstr/python/modules/mouse_resting_state/mouse_model/commontemplate0_orientation_corrected.nii.gz'
    mask_probability = '/softdev/akuurstr/python/modules/mouse_resting_state/mouse_model/model_mask.nii.gz'

    both_workflows = pe.Workflow(name='wf2')

    brainsuite_method = init_n4_bias_and_brain_extraction_wf(BrainExtractMethod.BRAINSUITE)
    brainsuite_method.inputs.inputnode.in_file = str(in_file)

    template_method = init_n4_bias_and_brain_extraction_wf(BrainExtractMethod.REGISTRATION_WITH_MASK, name='template_method')
    template_method.inputs.inputnode.in_file = str(in_file)
    template_method.inputs.inputnode.template = str(template)
    template_method.inputs.inputnode.template_probability_mask = str(mask_probability)
    both_workflows.connect([
        (brainsuite_method, template_method, [('outputnode.out_file_mask', 'inputnode.in_file_mask')]),
    ])

    both_workflows.base_dir = 'brain_extraction_wf'
    both_workflows.config['execution']['remove_unnecessary_outputs'] = False
    both_workflows.run()

    # what if no input mask for structural??


