from workflows.BrainExtractionWorkflows import init_n4_bias_and_brain_extraction_wf, BrainExtractMethod
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from multiprocessing import cpu_count
from nipype_interfaces.ants_preprocess import MotionCorr


def init_bold_reference(
        name='bold_reference_wf',
        omp_nthreads=None,
        mem_gb=3.0,
        n4_bspline_fitting_distance=20,
        diffusionConstant=30,
        diffusionIterations=3,
        edgeDetectionConstant=0.55,
        radius=2,
        dilateFinalMask=True,
        perform_motion_correction=True,
        brain_extract_method=BrainExtractMethod.REGISTRATION_WITH_MASK
):
    wf = pe.Workflow(name)

    if omp_nthreads is None or omp_nthreads < 1:
        omp_nthreads = cpu_count()

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_file', 'bold_avg_mask', 'bold_template', 'bold_template_probability_mask', ]), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_avg', 'bold_avg_n4_corrected', 'bold_avg_mask', 'motion_correction_transform', 'bold_motion_corrected']),
        name='outputnode')

    ants_mc_initial_average = pe.Node(MotionCorr(), name='initial_average')

    ants_mc = pe.Node(MotionCorr(), name='motion_correct')
    ants_mc.inputs.metric_type = 'GC'
    ants_mc.inputs.metric_weight = 1
    ants_mc.inputs.radius_or_bins = 1
    ants_mc.inputs.sampling_strategy = "Random"
    ants_mc.inputs.sampling_percentage = 0.05
    ants_mc.inputs.iterations = [10, 3]
    ants_mc.inputs.smoothing_sigmas = [0, 0]
    ants_mc.inputs.shrink_factors = [1, 1]
    ants_mc.inputs.n_images = 15
    ants_mc.inputs.use_fixed_reference_image = True
    ants_mc.inputs.use_scales_estimator = True
    ants_mc.inputs.output_average_image = 'output_average.nii.gz'
    ants_mc.inputs.output_warped_image = 'warped.nii.gz'
    ants_mc.inputs.output_transform_prefix = 'motcorr'
    ants_mc.inputs.transformation_model = 'Affine'
    ants_mc.inputs.gradient_step_length = 0.005
    # the composite tranform is stored in csv. no examples on how to concat that transform or apply it to an input image.
    # it's probably more used for antsMotionCorrStats
    # however there are examples on how to use the transform as a displacement field (https://stnava.github.io/fMRIANTs/)
    ants_mc.inputs.write_displacement = True

    func_brain_extraction_wf = init_n4_bias_and_brain_extraction_wf(brain_extract_method,
                                                                      name='func_brain_extraction_wf',
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
        (inputnode, ants_mc_initial_average, [('bold_file', 'average_image')]),
        (func_brain_extraction_wf, outputnode, [('outputnode.out_file_n4_corrected', 'bold_avg_n4_corrected')]),
    ])

    if perform_motion_correction:
        wf.connect([
            (ants_mc_initial_average, ants_mc, [('average_image', 'fixed_image')]),
            (inputnode, ants_mc, [('bold_file', 'moving_image')]),

            (ants_mc, outputnode, [('average_image', 'bold_avg')]),
            (ants_mc, func_brain_extraction_wf, [('average_image', 'inputnode.in_file')]),
            (ants_mc, outputnode, [('displacement_field', 'motion_correction_transform')]),
            (ants_mc, outputnode, [('warped_image', 'bold_motion_corrected')]),
        ])
    else:
        wf.connect([
            (ants_mc_initial_average, outputnode, [('average_image', 'bold_avg')]),
            (ants_mc_initial_average, func_brain_extraction_wf, [('average_image', 'inputnode.in_file')]),
        ])

    if not brain_extract_method == BrainExtractMethod.NO_BRAIN_EXTRACTION:
        wf.connect([
            (func_brain_extraction_wf, outputnode, [('outputnode.out_file_mask', 'bold_avg_mask')]),
        ])

    if brain_extract_method in (BrainExtractMethod.REGISTRATION_WITH_MASK, BrainExtractMethod.REGISTRATION_NO_MASK):
        wf.connect([
            (inputnode, func_brain_extraction_wf, [('bold_template', 'inputnode.template')]),
            (inputnode, func_brain_extraction_wf,
             [('template_probability_mask', 'inputnode.template_probability_mask')]),
        ])
        if brain_extract_method == BrainExtractMethod.REGISTRATION_WITH_MASK:
            wf.connect([
                (inputnode, func_brain_extraction_wf, [('bold_avg_mask', 'inputnode.in_file_mask')]),
            ])
    return wf
