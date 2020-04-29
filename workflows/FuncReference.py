from workflows.BrainExtraction import init_n4_bias_and_brain_extraction_wf, BrainExtractMethod
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from multiprocessing import cpu_count
from nipype.interfaces.fsl import Split, Merge
from nipype.interfaces.ants import Registration, ApplyTransforms, AverageImages
from nipype.interfaces.utility import Function
from nipype.interfaces.base.traits_extension import isdefined
from nipype_interfaces.DisplacementManip import MergeDisplacement


def select_images(image_list, images_slice):
    return image_list[images_slice]


def get_avg_selection_node(name='select_imgs_to_avg'):
    node = pe.Node(interface=Function(input_names=['image_list', 'images_slice'], output_names=['sliced_list'],
                                                    function=select_images), name=name)
    return node

def get_average_image_wf(name='avg_img',images_slice=slice(None,None),omp_nthreads=None,mem_gb=3.0):
    wf=pe.Workflow(name=name)
    if omp_nthreads is None or omp_nthreads < 1:
        omp_nthreads = cpu_count()
    inputnode = pe.Node(interface=niu.IdentityInterface(fields=['images', 'normalize', 'output_average_image','images_slice']),
                        name='inputnode')
    inputnode.inputs.images_slice=images_slice
    outputnode = pe.Node(interface=niu.IdentityInterface(fields=['output_average_image']),name='outputnode')

    imgs_to_avg = get_avg_selection_node(name='imgs_to_avg')
    avg_img = pe.Node(interface=AverageImages(), name='avg_img',n_procs=omp_nthreads,mem_gb=mem_gb)
    avg_img.inputs.output_average_image = 'average.nii.gz'
    avg_img.inputs.dimension = 3

    wf.connect([
        (inputnode, imgs_to_avg, [('images', 'image_list')]),
        (inputnode, imgs_to_avg, [('images_slice', 'images_slice')]),
        (imgs_to_avg, avg_img, [('sliced_list', 'images')]),
        (inputnode, avg_img, [('normalize', 'normalize')]),
        (avg_img, outputnode, [('output_average_image', 'output_average_image')]),
    ])

    if isdefined(inputnode.inputs.output_average_image):
        wf.connect([
        (inputnode, avg_img, [('output_average_image', 'output_average_image')]),
        ])
    return wf

def init_func_reference(
        name='func_reference_wf',
        func_metadata=None,
        perform_motion_correction=True,
        brain_extract_method=BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK,
        write_merged_files=True,

        # for BrainExtractMethod.BRAINSUITE only
        n4_bspline_fitting_distance=20,
        diffusionConstant=30,
        diffusionIterations=3,
        edgeDetectionConstant=0.55,
        radius=2,
        dilateFinalMask=True,

        # for nipype nodes
        reduce_to_float_precision=False,
        interpolation='Linear',
        nthreads_node=None,
        mem_gb_node=3.0,
        nthreads_mapnode=None,
        mem_gb_mapnode=3,
):
    wf = pe.Workflow(name)

    if nthreads_node is None or nthreads_node < 1:
        nthreads_node = cpu_count()

    tr=None
    if func_metadata is not None:
        tr=func_metadata.get('RepetitionTime')
    if tr is None:
        tr=1

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['func_file', 'func_avg_mask', 'func_template', 'func_template_probability_mask', ]), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['func_avg',
                'func_avg_n4_corrected',
                'func_avg_mask',
                'func_motion_corrected',
                'motion_correction_transform',
                'func_split',
                'func_motion_corrected_split',
                'motion_correction_transform_split',
                ]),name='outputnode')
    output_avg = pe.Node(niu.IdentityInterface(fields=['func_avg']),name='output_avg')


    # custom inteface to antsMotionCorr does not work with non-linear registration
    # do our own motion correction with a mapnode and antsRegistration
    split_func = pe.Node(interface=Split(), name='split_func', n_procs=nthreads_node, mem_gb=mem_gb_node)
    split_func.inputs.dimension = 't'

    initial_avg = get_average_image_wf(name='initial_avg', images_slice=slice(0,50), omp_nthreads=nthreads_node, mem_gb=mem_gb_node)
    initial_avg.inputs.inputnode.normalize = False

    # if we wanted to average the single 4d func file instead of the split 3d volumes we could use:
    # fslroi input output tmin tmax
    # fslmaths input -Tmean output

    mc_register_func_to_avg = pe.MapNode(interface=Registration(), name='mc_register_func_to_avg', iterfield=['moving_image',], n_procs=nthreads_mapnode, mem_gb=mem_gb_mapnode)
    mc_register_func_to_avg.inputs.dimension = 3
    mc_register_func_to_avg.inputs.float = reduce_to_float_precision
    mc_register_func_to_avg.inputs.interpolation = interpolation
    mc_register_func_to_avg.inputs.output_transform_prefix = "output_"
    mc_register_func_to_avg.inputs.transforms = ['Affine', 'SyN']
    mc_register_func_to_avg.inputs.transform_parameters = [(0.005,),(0.005, 0.0, 0.0)]
    mc_register_func_to_avg.inputs.write_composite_transform = False  # transform for each stage vs composite for entire warp

    mc_register_func_to_avg.inputs.number_of_iterations = [[20],[20]]
    mc_register_func_to_avg.inputs.metric = ['CC'] * 2
    mc_register_func_to_avg.inputs.metric_weight = [1,1]
    mc_register_func_to_avg.inputs.radius_or_number_of_bins = [2] * 2  # radius for CC between 2-5
    mc_register_func_to_avg.inputs.sampling_strategy = ['Regular',None]
    mc_register_func_to_avg.inputs.sampling_percentage = [0.2,None]
    mc_register_func_to_avg.inputs.smoothing_sigmas = [[0],[0]]
    mc_register_func_to_avg.inputs.shrink_factors = [[1],[1]]
    mc_register_func_to_avg.inputs.use_histogram_matching = [True] * 2

    #mc_register_func_to_avg.inputs.convergence_threshold = [1.e-8] * 2
    #mc_register_func_to_avg.inputs.convergence_window_size = [5,5]
    mc_register_func_to_avg.inputs.sigma_units = ['mm'] * 2  # we use mm instead of vox because we don't have isotropic voxels
    mc_register_func_to_avg.inputs.use_estimate_learning_rate_once = [False] * 2  # estimate the learning rate step size only at the beginning of each level. Does this override the value chosen in transform_parameters?
    mc_register_func_to_avg.inputs.output_warped_image = 'motion_corrected.nii.gz'
    mc_register_func_to_avg.n_procs = nthreads_node
    mc_register_func_to_avg.inputs.verbose = True

    refined_avg = get_average_image_wf(name='refined_avg', images_slice=slice(None, None), omp_nthreads=nthreads_node, mem_gb=mem_gb_node)
    refined_avg.inputs.inputnode.normalize = False

    from nipype.interfaces.utility import Function
    def reverse_list(forward_list):
        reverselist = forward_list.copy()
        reverselist.reverse()
        return reverselist
    reverse_transform_list = pe.MapNode(
        Function(input_names=["forward_list"], output_names=["transforms_reversed"],
                 function=reverse_list),name='reverse_transform_list', iterfield=['forward_list'])

    combine_mc_displacements=pe.MapNode(interface=ApplyTransforms(), name='combine_mc_displacements', iterfield=['transforms','reference_image','input_image'], n_procs=nthreads_mapnode, mem_gb=mem_gb_mapnode)
    combine_mc_displacements.inputs.dimension = 3
    combine_mc_displacements.inputs.output_image = 'motion_corr_transform.nii.gz'
    combine_mc_displacements.inputs.print_out_composite_warp_file = True

    create_4d_mc_func = pe.Node(interface=Merge(), name='create_4d_mc_func', n_procs=nthreads_node, mem_gb=mem_gb_node)
    create_4d_mc_func.inputs.dimension = 't'
    create_4d_mc_func.inputs.tr = tr


    create_4d_mc_displacement = pe.Node(interface=MergeDisplacement(), name='create_4d_mc_displacement', n_procs=nthreads_node, mem_gb=mem_gb_node)
    create_4d_mc_displacement.inputs.tr = tr


    func_brain_extraction_wf = init_n4_bias_and_brain_extraction_wf(brain_extract_method,
                                                                    name='func_brain_extraction_wf',
                                                                    omp_nthreads=nthreads_node,
                                                                    mem_gb=mem_gb_node,
                                                                    n4_bspline_fitting_distance=n4_bspline_fitting_distance,
                                                                    diffusionConstant=diffusionConstant,
                                                                    diffusionIterations=diffusionIterations,
                                                                    edgeDetectionConstant=edgeDetectionConstant,
                                                                    radius=radius,
                                                                    dilateFinalMask=dilateFinalMask,
                                                                    )

    wf.connect([
        (inputnode, split_func, [('func_file', 'in_file')]),
        (split_func, outputnode, [('out_files', 'func_split')]),
        (split_func, initial_avg, [('out_files', 'inputnode.images')]),
        (func_brain_extraction_wf, outputnode, [('outputnode.out_file_n4_corrected', 'func_avg_n4_corrected')]),
        (output_avg, outputnode, [('func_avg', 'func_avg')]),
    ])

    if perform_motion_correction:
        wf.connect([
            (split_func, mc_register_func_to_avg, [('out_files', 'moving_image')]),
            (initial_avg, mc_register_func_to_avg, [('outputnode.output_average_image', 'fixed_image')]),

            (mc_register_func_to_avg, refined_avg, [('warped_image', 'inputnode.images')]),
            (refined_avg, func_brain_extraction_wf, [('outputnode.output_average_image', 'inputnode.in_file')]),

            (mc_register_func_to_avg, reverse_transform_list, [('forward_transforms', 'forward_list')]),
            (reverse_transform_list, combine_mc_displacements, [('transforms_reversed', 'transforms')]),
            (mc_register_func_to_avg, combine_mc_displacements, [('warped_image', 'reference_image')]),
            (mc_register_func_to_avg, combine_mc_displacements, [('warped_image', 'input_image')]),

            (refined_avg, output_avg, [('outputnode.output_average_image', 'func_avg')]),
            (mc_register_func_to_avg, outputnode, [('warped_image', 'func_motion_corrected_split')]),
            (combine_mc_displacements, outputnode, [('output_image', 'motion_correction_transform_split')]),
        ])

        if write_merged_files:
            wf.connect([
            (mc_register_func_to_avg, create_4d_mc_func, [('warped_image', 'in_files')]),
            (create_4d_mc_func, outputnode, [('merged_file', 'func_motion_corrected')]),
            (combine_mc_displacements, create_4d_mc_displacement, [('output_image', 'displacement_imgs')]),
            (create_4d_mc_displacement, outputnode, [('output_file', 'motion_correction_transform')]),
            ])
    else:
        wf.connect([
            (initial_avg, func_brain_extraction_wf, [('outputnode.output_average_image', 'inputnode.in_file')]),
            (initial_avg, output_avg, [('outputnode.output_average_image', 'func_avg')]),
        ])

    if not brain_extract_method == BrainExtractMethod.NO_BRAIN_EXTRACTION:
        wf.connect([
            (func_brain_extraction_wf, outputnode, [('outputnode.out_file_mask', 'func_avg_mask')]),
        ])

    # if brain_extract_method in (BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK, BrainExtractMethod.REGISTRATION_NO_INITIAL_MASK):
    #     wf.connect([
    #         (inputnode, func_brain_extraction_wf, [('func_template', 'inputnode.template')]),
    #         (inputnode, func_brain_extraction_wf,
    #          [('template_probability_mask', 'inputnode.template_probability_mask')]),
    #     ])
    #     if brain_extract_method == BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK:
    #         wf.connect([
    #             (inputnode, func_brain_extraction_wf, [('func_avg_mask', 'inputnode.in_file_mask')]),
    #         ])
    # return wf


    if brain_extract_method in (BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK, BrainExtractMethod.REGISTRATION_NO_INITIAL_MASK, BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK):
        wf.connect([
            (inputnode, func_brain_extraction_wf, [('func_template', 'inputnode.template')]),
            (inputnode, func_brain_extraction_wf,
             [('func_template_probability_mask', 'inputnode.template_probability_mask')]),
        ])
    if brain_extract_method in (BrainExtractMethod.USER_PROVIDED_MASK,BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK):
            wf.connect([
                (inputnode, func_brain_extraction_wf, [('func_avg_mask', 'inputnode.in_file_mask')]),
            ])
    return wf