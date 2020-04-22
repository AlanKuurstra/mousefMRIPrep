from workflows.BrainExtraction import init_n4_bias_and_brain_extraction_wf, BrainExtractMethod
from workflows.RegistrationAnatToAtlas import init_anat_to_atlas_registration
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from multiprocessing import cpu_count
from workflows.FuncReference import init_func_reference
from workflows.RegistrationFuncToAnat import init_func_to_anat_registration
from nipype.interfaces.ants import ApplyTransforms
from nipype_interfaces.DisplacementManip import ReplicateDisplacement, ReplicateImage
from nipype.interfaces.utility import Function
from nipype.interfaces.fsl import Split, Merge
from enum import Enum
from fmriprep.workflows.bold.stc import init_bold_stc_wf
import json
import nibabel as nib
from nipype.interfaces.base.traits_extension import isdefined
from nipype_interfaces.DerivativesDatasink import init_derivatives_datasink


class MotionCorrectionTransform(Enum):
    CONCAT = 1
    DOUBLE_INTERPOLATION = 2
    NO_MC = 3


def get_transform_concat_list(apply_first=False, apply_second=False, apply_third=False):
    # ApplyTransforms wants transforms listed in reverse order of application
    transforms_reverse_order = []
    if apply_third:
        transforms_reverse_order.append(apply_third)
    if apply_second:
        transforms_reverse_order.append(apply_second)
    if apply_first:
        transforms_reverse_order.append(apply_first)
    return transforms_reverse_order


def get_node_transform_concat_list(name='transform_concat_list', mapnode_iterfield=None):
    if mapnode_iterfield is None:
        node = pe.Node(
            Function(input_names=["apply_first", "apply_second", "apply_third"], output_names=["transforms"],
                     function=get_transform_concat_list), name=name)
    else:
        node = pe.MapNode(
            interface=Function(input_names=["apply_first", "apply_second", "apply_third"], output_names=["transforms"],
                               function=get_transform_concat_list), name=name, iterfield=mapnode_iterfield)
    return node


def init_func_processing(
        # pipeline input parameters
        name='func_processing',
        input_func_file=None,
        atlas=None,
        atlas_mask=None,
        label_mapping=None,

        # motion correction (usually not necessary because of earbars)
        mc_transform_method=MotionCorrectionTransform.NO_MC,

        # slice timing correction
        perform_stc=None,

        # func brain extraction (only necessary if performing func to anat registration and using masks)
        # how to extract a mask for functional image
        func_brain_extract_method=BrainExtractMethod.NO_BRAIN_EXTRACTION,
        # if using REGISTRATION_WITH_INITIAL_MASK or USER_PROVIDED_MASK
        func_file_mask=None,
        # if using registration brain extraction
        func_brain_extract_template=None,
        func_brain_extract_probability_mask=None,

        # func to anat options (usually not necessary because of earbars)
        perform_func_to_anat_registration=False,
        # use extracted masks in func to anat registration?
        use_masks_func_to_anat_registration=False,

        # node resources
        omp_nthreads=None,
        mem_gb=50,

        # registration processing time
        interpolation="Linear",
        reduce_to_float_precision=True,
        # gzip func_to_atlas transform and final registered functional image? reduces image size, increases processing time
        gzip_large_images=True,

        # for brainsuite brain extraction
        n4_bspline_fitting_distance=20,
        diffusionConstant=30,
        diffusionIterations=3,
        edgeDetectionConstant=0.55,
        radius=2,
        dilateFinalMask=True,

        # correlation matrix
        correlation_shift_interval_s=0.375,
        correlation_max_shift_s=1.5,
        correlation_search_for_neg_corr=False,

        # for datasink
        derivatives_collection_dir=None,
        derivatives_pipeline_name='MousefMRIPrep',
):
    # set other parameters

    # initial development with lower resolution atlas allowed for transformation of 4d functional file
    # there are memory problems when attempting a 4d transformation to a high resolution atlas
    # force the pipeline to split the functional in the temporal domain
    # we force the split because the correlation matrix nodes currently only work when split along temporal
    # will be easy to create additional correlation matrix interfaces for 4d files if necessary
    SPLIT_FUNC_INTO_SEPARATE_VOLUMES = True

    metadata = {}
    with open(input_func_file.split(".nii")[0] + ".json", 'r') as f:
        metadata = json.load(f)

    if perform_stc == None:
        perform_stc = bool(metadata.get("SliceTiming"))

    if (mc_transform_method == MotionCorrectionTransform.CONCAT) and perform_stc:
        print(
            "Warning: Motion Correction Transform switched to DOUBLE_INTERPOLATION due to enabling of slice timing correction")
        mc_transform_method = MotionCorrectionTransform.DOUBLE_INTERPOLATION

    if use_masks_func_to_anat_registration and (func_brain_extract_method == BrainExtractMethod.NO_BRAIN_EXTRACTION):
        print("Warning: Not using masks in func to anat registration since func_brain_extract_method == BrainExtractMethod.NO_BRAIN_EXTRACTION")
        use_masks_func_to_anat_registration = False


    if omp_nthreads is None or omp_nthreads < 1:
        omp_nthreads = cpu_count()

    perform_motion_correction = (mc_transform_method != MotionCorrectionTransform.NO_MC)

    func_header = nib.load(input_func_file).header
    tr = None
    if metadata is not None:
        tr = metadata.get('RepetitionTime')
    if tr is None:
        tr = func_header['pixdim'][4]
        metadata['RepetitionTime'] = tr
    nvolumes = func_header['dim'][4]

    # create workflow
    wf = pe.Workflow(name)

    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'func_file',
        'func_file_mask',
        'func_template',
        'func_template_probability_mask',
        'anat_file',
        'anat_file_mask',
        'anat_to_atlas_composite_transform',
        'atlas',
        'atlas_mask',
        'label_mapping',
    ]), name='inputnode')

    # NOTE: best practice is to set these inputnode values using parameters in init_func_to_atlas_registration
    # the isdefined() method is used during pipeline creation to determine which nodes to create and connect
    # if the inputnode is set after the pipeline is created, necessary nodes may not be created.

    # for example if you don't set the label_mapping in the function call:
    #   wf = init_func_to_atlas_registration(...)
    # and then try to set it later using
    #   wf.inputs.inputnode.func_file = input_func_file
    # it won't work because the correlation matrix nodes were not created during the execution of init_func_to_atlas_registration

    if input_func_file is not None:
        inputnode.inputs.func_file = input_func_file
    if func_file_mask is not None:
        inputnode.inputs.func_file_mask = func_file_mask
    if atlas is not None:
        inputnode.inputs.atlas = atlas
    if atlas_mask is not None:
        inputnode.inputs.atlas_mask = atlas_mask
    if label_mapping is not None:
        inputnode.inputs.label_mapping = label_mapping

    if func_brain_extract_template is not None:
        inputnode.inputs.func_template = func_brain_extract_template
    if func_brain_extract_probability_mask is not None:
        inputnode.inputs.func_template_probability_mask = func_brain_extract_probability_mask

    preprocess_func_wf = init_func_reference(
        name='preprocess_func',
        func_metadata=metadata,
        perform_motion_correction=perform_motion_correction,
        brain_extract_method=func_brain_extract_method,
        write_merged_files=True,
        # for BrainExtractMethod.BRAINSUITE only
        n4_bspline_fitting_distance=n4_bspline_fitting_distance,
        diffusionConstant=diffusionConstant,
        diffusionIterations=diffusionIterations,
        edgeDetectionConstant=edgeDetectionConstant,
        radius=radius,
        dilateFinalMask=dilateFinalMask,

        reduce_to_float_precision=reduce_to_float_precision,
        interpolation=interpolation,

        omp_nthreads=omp_nthreads,
        mem_gb=mem_gb,
    )

    wf.connect([
        (inputnode, preprocess_func_wf, [('func_file', 'inputnode.func_file')]),
    ])

    if func_brain_extract_method in (
    BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK, BrainExtractMethod.REGISTRATION_NO_INITIAL_MASK, BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK):
        wf.connect([
            (inputnode, preprocess_func_wf, [('func_template', 'inputnode.func_template')]),
            (inputnode, preprocess_func_wf,
             [('func_template_probability_mask', 'inputnode.func_template_probability_mask')]),
        ])
    if func_brain_extract_method in (BrainExtractMethod.USER_PROVIDED_MASK,BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK):
        wf.connect([
            (inputnode, preprocess_func_wf, [('func_file_mask', 'inputnode.func_avg_mask')]),
        ])

    concat_list_func_to_atlas = get_node_transform_concat_list(name='concat_list_func_to_atlas')

    if perform_func_to_anat_registration:
        func_to_anat = init_func_to_anat_registration(
            mask=use_masks_func_to_anat_registration,
            reduce_to_float_precision=reduce_to_float_precision,
            interpolation=interpolation,
            omp_nthreads=omp_nthreads,
            mem_gb=mem_gb,
        )
        wf.connect([
            (preprocess_func_wf, func_to_anat, [('outputnode.func_avg_n4_corrected', 'inputnode.func_reference')]),
            (
                inputnode, func_to_anat, [('anat_file', 'inputnode.anat')]),
            (func_to_anat, concat_list_func_to_atlas,
             [('outputnode.func_to_anat_composite_transform', 'apply_first')]),
        ])
        if use_masks_func_to_anat_registration:
            wf.connect([
                (inputnode, func_to_anat,
                 [('anat_file_mask', 'inputnode.anat_mask')]),
                (preprocess_func_wf, func_to_anat, [('outputnode.func_avg_mask', 'inputnode.func_reference_mask')]),
            ])

    concat_transforms_func_to_atlas = pe.Node(interface=ApplyTransforms(), name='concat_transforms_func_to_atlas',
                                              n_procs=omp_nthreads, mem_gb=mem_gb)
    concat_transforms_func_to_atlas.inputs.dimension = 3
    concat_transforms_func_to_atlas.inputs.float = reduce_to_float_precision
    # concat_transforms_func_to_atlas.inputs.output_image = 'func_to_atlas_transform.h5'
    if gzip_large_images:
        concat_transforms_func_to_atlas.inputs.output_image = 'func_to_atlas_transform.nii.gz'
    else:
        concat_transforms_func_to_atlas.inputs.output_image = 'func_to_atlas_transform.nii'
    concat_transforms_func_to_atlas.inputs.print_out_composite_warp_file = True

    wf.connect([
        # (func_to_anat, concat_list_func_to_atlas, [('outputnode.func_to_anat_composite_transform', 'apply_first')]),
        (inputnode, concat_list_func_to_atlas, [('anat_to_atlas_composite_transform', 'apply_second')]),
        (concat_list_func_to_atlas, concat_transforms_func_to_atlas, [('transforms', 'transforms')]),
        (inputnode, concat_transforms_func_to_atlas, [('atlas', 'reference_image')]),
        (preprocess_func_wf, concat_transforms_func_to_atlas, [('outputnode.func_avg', 'input_image')]),
    ])

    func_to_warp_to_atlas = pe.Node(niu.IdentityInterface(fields=['func_file']), name='func_to_warp_to_atlas')

    if not perform_stc:
        if (mc_transform_method in (MotionCorrectionTransform.NO_MC, MotionCorrectionTransform.CONCAT)):
            if SPLIT_FUNC_INTO_SEPARATE_VOLUMES:
                wf.connect([(preprocess_func_wf, func_to_warp_to_atlas, [('outputnode.func_split', 'func_file')]), ])
            else:
                wf.connect([(inputnode, func_to_warp_to_atlas, [('func_file', 'func_file')]), ])

        elif mc_transform_method == MotionCorrectionTransform.DOUBLE_INTERPOLATION:
            if SPLIT_FUNC_INTO_SEPARATE_VOLUMES:
                wf.connect([(preprocess_func_wf, func_to_warp_to_atlas,
                             [('outputnode.func_motion_corrected_split', 'func_file')]), ])
            else:
                wf.connect([(preprocess_func_wf, func_to_warp_to_atlas,
                             [('outputnode.func_motion_corrected', 'func_file')]), ])
    elif perform_stc:
        func_stc_wf = init_bold_stc_wf(name='preprocess_func_stc', metadata=metadata)
        split_stc = pe.Node(interface=Split(), name='preprocess_func_stc_split', n_procs=omp_nthreads, mem_gb=mem_gb)
        split_stc.inputs.dimension = 't'
        if mc_transform_method == MotionCorrectionTransform.NO_MC:
            if SPLIT_FUNC_INTO_SEPARATE_VOLUMES:
                wf.connect([
                    (inputnode, func_stc_wf, [('func_file', 'inputnode.bold_file')]),
                    (func_stc_wf, split_stc, [('outputnode.stc_file', 'in_file')]),
                    (split_stc, func_to_warp_to_atlas, [('out_files', 'func_file')]),
                ])
            else:
                wf.connect([
                    (inputnode, func_stc_wf, [('func_file', 'inputnode.bold_file')]),
                    (func_stc_wf, func_to_warp_to_atlas, [('outputnode.stc_file', 'func_file')]),
                ])
        elif mc_transform_method == MotionCorrectionTransform.DOUBLE_INTERPOLATION:
            if SPLIT_FUNC_INTO_SEPARATE_VOLUMES:
                wf.connect([
                    (preprocess_func_wf, func_stc_wf, [('outputnode.func_motion_corrected', 'inputnode.bold_file')]),
                    (func_stc_wf, split_stc, [('outputnode.stc_file', 'in_file')]),
                    (split_stc, func_to_warp_to_atlas, [('out_files', 'func_file')]),
                ])
            else:
                wf.connect([
                    (preprocess_func_wf, func_stc_wf, [('outputnode.func_motion_corrected', 'inputnode.bold_file')]),
                    (func_stc_wf, func_to_warp_to_atlas, [('outputnode.stc_file', 'func_file')]),
                ])
        elif mc_transform_method == MotionCorrectionTransform.CONCAT:
            assert ("ERROR: CANNOT CONCAT MOTION CORRECTION TRANSFORM WHEN PERFORMING SLICE TIMING CORRECTION")

    if not SPLIT_FUNC_INTO_SEPARATE_VOLUMES:
        register_func_to_atlas = pe.Node(interface=ApplyTransforms(), name='register_func_to_atlas',
                                         n_procs=omp_nthreads, mem_gb=mem_gb)
        if gzip_large_images:
            register_func_to_atlas.inputs.output_image = 'warped.nii.gz'
        else:
            register_func_to_atlas.inputs.output_image = 'warped.nii'
        register_func_to_atlas.inputs.float = reduce_to_float_precision
        register_func_to_atlas.inputs.interpolation = interpolation

        if mc_transform_method in (MotionCorrectionTransform.NO_MC, MotionCorrectionTransform.DOUBLE_INTERPOLATION):
            # mice are in ear bars, so if motion correction is deemed unnecessary, processing time can be reduced
            register_func_to_atlas.inputs.dimension = 3
            register_func_to_atlas.inputs.input_image_type = 3
            wf.connect([
                (concat_transforms_func_to_atlas, register_func_to_atlas, [('output_image', 'transforms')]),
                (func_to_warp_to_atlas, register_func_to_atlas, [('func_file', 'input_image')]),
                (inputnode, register_func_to_atlas, [('atlas', 'reference_image')]),

            ])

        elif mc_transform_method == MotionCorrectionTransform.CONCAT:
            concat_list_mc_to_func_to_atlas = get_node_transform_concat_list(name='concat_list_mc_to_func_to_atlas')

            # if atlas is low res, there's enough memory to do use ReplicateImage and ReplicateDisplacement
            replicate_atlas = pe.Node(interface=ReplicateImage(), name='replicate_atlas', n_procs=omp_nthreads,
                                      mem_gb=mem_gb)
            replicate_atlas.inputs.tr = tr
            replicate_atlas.inputs.reps = nvolumes

            replicate_displacement = pe.Node(interface=ReplicateDisplacement(), name='replicate_displacement',
                                             n_procs=omp_nthreads, mem_gb=mem_gb)
            replicate_displacement.inputs.tr = tr
            replicate_displacement.inputs.reps = nvolumes

            register_func_to_atlas.inputs.dimension = 4
            wf.connect([
                (inputnode, replicate_atlas, [('atlas', 'input_3d_image')]),
                (concat_transforms_func_to_atlas, replicate_displacement, [('output_image', 'input_transform')]),
                (preprocess_func_wf, concat_list_mc_to_func_to_atlas,
                 [('outputnode.motion_correction_transform', 'apply_first')]),
                (replicate_displacement, concat_list_mc_to_func_to_atlas, [('output_file', 'apply_second')]),
                (concat_list_mc_to_func_to_atlas, register_func_to_atlas, [('transforms', 'transforms')]),
                (func_to_warp_to_atlas, register_func_to_atlas, [('func_file', 'input_image')]),
                (replicate_atlas, register_func_to_atlas, [('output_file', 'reference_image')]),
            ])



    elif SPLIT_FUNC_INTO_SEPARATE_VOLUMES:
        concat_list_mc_to_func_to_atlas = get_node_transform_concat_list(name='concat_list_mc_to_func_to_atlas',
                                                                         mapnode_iterfield=['apply_first'])

        register_func_to_atlas = pe.MapNode(interface=ApplyTransforms(), name='register_func_to_atlas',
                                            iterfield=['input_image', 'transforms'], n_procs=omp_nthreads,
                                            mem_gb=mem_gb)
        register_func_to_atlas.inputs.dimension = 3

        if gzip_large_images:
            register_func_to_atlas.inputs.output_image = 'warped.nii.gz'
        else:
            register_func_to_atlas.inputs.output_image = 'warped.nii'
        register_func_to_atlas.inputs.float = reduce_to_float_precision
        register_func_to_atlas.inputs.interpolation = interpolation

        create_4d_image = pe.Node(interface=Merge(), name='create_4d_image', n_procs=omp_nthreads, mem_gb=mem_gb)
        create_4d_image.inputs.dimension = 't'
        create_4d_image.inputs.tr = tr

        wf.connect([
            (func_to_warp_to_atlas, register_func_to_atlas, [('func_file', 'input_image')]),
            (inputnode, register_func_to_atlas, [('atlas', 'reference_image')]),
            (concat_transforms_func_to_atlas, concat_list_mc_to_func_to_atlas, [('output_image', 'apply_second')]),
            (concat_list_mc_to_func_to_atlas, register_func_to_atlas, [('transforms', 'transforms')]),
            # (register_func_to_atlas, create_4d_image, [('output_image', 'in_files')]),
        ])

        if mc_transform_method in (MotionCorrectionTransform.NO_MC, MotionCorrectionTransform.DOUBLE_INTERPOLATION):
            # we need the concat_list_mc_to_func_to_atlas mapnode to expand our single 3d transform even though we aren't applying
            # motion correction transforms to input file
            concat_list_mc_to_func_to_atlas.inputs.apply_first = [None] * nvolumes
        elif mc_transform_method == MotionCorrectionTransform.CONCAT:
            wf.connect([
                (preprocess_func_wf, concat_list_mc_to_func_to_atlas,
                 [('outputnode.motion_correction_transform_split', 'apply_first')]),
            ])

        if isdefined(inputnode.inputs.label_mapping):
            from nipype_interfaces.CorrelationMatrix import ExractLabelMeans, ComputeCorrelationMatrix
            extract_label_means = pe.Node(interface=ExractLabelMeans(), name='extract_label_signal_means')
            compute_corr_mtx = pe.Node(interface=ComputeCorrelationMatrix(), name='compute_corr_mtx')
            compute_corr_mtx.inputs.shift_interval_s = correlation_shift_interval_s
            compute_corr_mtx.inputs.max_shift_s = correlation_max_shift_s
            compute_corr_mtx.inputs.tr = tr
            compute_corr_mtx.inputs.search_for_neg_corr = correlation_search_for_neg_corr

            wf.connect([
                (inputnode, extract_label_means, [('label_mapping', 'label_file')]),
                (register_func_to_atlas, extract_label_means, [('output_image', 'split_volumes_list')]),
                (extract_label_means, compute_corr_mtx, [('output_file_pkl', 'label_signals_pkl')]),
            ])

    # datasinks
    derivatives_func_avg_n4_corrected = init_derivatives_datasink('derivatives_func_avg_n4_corrected',
                                                                  bids_datatype='func',
                                                                  bids_description='FuncAvgN4Corrected',
                                                                  derivatives_collection_dir=derivatives_collection_dir,
                                                                  derivatives_pipeline_name=derivatives_pipeline_name)

    derivatives_func_to_atlas_transform = init_derivatives_datasink('derivatives_func_to_atlas_transform',
                                                                    bids_datatype='func',
                                                                    bids_description='FuncToAtlasTransform',
                                                                    derivatives_collection_dir=derivatives_collection_dir,
                                                                    derivatives_pipeline_name=derivatives_pipeline_name)

    wf.connect([
        (inputnode, derivatives_func_avg_n4_corrected, [('func_file', 'inputnode.original_bids_file')]),
        (preprocess_func_wf, derivatives_func_avg_n4_corrected,
         [('outputnode.func_avg_n4_corrected', 'inputnode.file_to_rename')]),

        (inputnode, derivatives_func_to_atlas_transform, [('func_file', 'inputnode.original_bids_file')]),
        (concat_transforms_func_to_atlas, derivatives_func_to_atlas_transform,
         [('output_image', 'inputnode.file_to_rename')]),
    ])

    # show user how well the functional registered to the atlas
    register_func_avg_to_atlas = pe.Node(interface=ApplyTransforms(), name='register_func_avg_to_atlas',
                                         n_procs=omp_nthreads,
                                         mem_gb=mem_gb)
    register_func_avg_to_atlas.inputs.dimension = 3
    register_func_avg_to_atlas.inputs.float = reduce_to_float_precision
    register_func_avg_to_atlas.inputs.interpolation = interpolation

    def first_element_if_list(mylist):
        if type(mylist) == list:
            return mylist[0]
        return mylist

    reduce_mapnode_if_necessary = pe.Node(
        interface=Function(input_names=['mylist'], output_names=['first_element'], function=first_element_if_list),
        name='register_func_avg_to_atlas_reduce_mapnode')
    derivatives_func_avg_to_atlas = init_derivatives_datasink('derivatives_func_avg_to_atlas', bids_datatype='func',
                                                              bids_description='FuncAvgToAtlas',
                                                              derivatives_collection_dir=derivatives_collection_dir,
                                                              derivatives_pipeline_name=derivatives_pipeline_name)
    wf.connect([
        (preprocess_func_wf, register_func_avg_to_atlas, [('outputnode.func_avg', 'input_image')]),
        (inputnode, register_func_avg_to_atlas, [('atlas', 'reference_image')]),
        (concat_list_mc_to_func_to_atlas, reduce_mapnode_if_necessary, [('transforms', 'mylist')]),
        (reduce_mapnode_if_necessary, register_func_avg_to_atlas, [('first_element', 'transforms')]),
        (inputnode, derivatives_func_avg_to_atlas, [('func_file', 'inputnode.original_bids_file')]),
        (register_func_avg_to_atlas, derivatives_func_avg_to_atlas, [('output_image', 'inputnode.file_to_rename')]),

    ])

    if func_brain_extract_method != BrainExtractMethod.NO_BRAIN_EXTRACTION:
        method = ''
        if func_brain_extract_method == BrainExtractMethod.BRAINSUITE:
            method = 'Brainsuite'
        elif func_brain_extract_method in (
        BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK, BrainExtractMethod.REGISTRATION_NO_INITIAL_MASK, BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK):
            method = 'TemplateExtracted'
        derivatives_func_brain_mask = init_derivatives_datasink('derivatives_func_brain_mask',
                                                                bids_datatype='func',
                                                                bids_description=f'Func{method}BrainMask',
                                                                derivatives_collection_dir=derivatives_collection_dir,
                                                                derivatives_pipeline_name=derivatives_pipeline_name)

        wf.connect([
            (inputnode, derivatives_func_brain_mask, [('func_file', 'inputnode.original_bids_file')]),
            (preprocess_func_wf, derivatives_func_brain_mask,
             [('outputnode.func_avg_mask', 'inputnode.file_to_rename')]),
        ])

    if perform_motion_correction:
        derivatives_func_motion_corrected = init_derivatives_datasink('derivatives_func_motion_corrected',
                                                                      bids_datatype='func',
                                                                      bids_description='FuncMotionCorrected',
                                                                      derivatives_collection_dir=derivatives_collection_dir,
                                                                      derivatives_pipeline_name=derivatives_pipeline_name)
        derivatives_func_motion_corrected_transform = init_derivatives_datasink(
            'derivatives_func_motion_corrected_transform',
            bids_datatype='func',
            bids_description='FuncMotionCorrectedTransform',
            derivatives_collection_dir=derivatives_collection_dir,
            derivatives_pipeline_name=derivatives_pipeline_name)
        wf.connect([
            (inputnode, derivatives_func_motion_corrected, [('func_file', 'inputnode.original_bids_file')]),
            (preprocess_func_wf, derivatives_func_motion_corrected, [('outputnode.func_motion_corrected', 'inputnode.file_to_rename')]),
            (inputnode, derivatives_func_motion_corrected_transform, [('func_file', 'inputnode.original_bids_file')]),
            (preprocess_func_wf, derivatives_func_motion_corrected_transform,
             [('outputnode.motion_correction_transform', 'inputnode.file_to_rename')]),
        ])

    if perform_func_to_anat_registration:
        derivatives_func_avg_to_anat = init_derivatives_datasink('derivatives_func_avg_to_anat', bids_datatype='func',
                                                                 bids_description='FuncAvgToAnat',
                                                                 derivatives_collection_dir=derivatives_collection_dir,
                                                                 derivatives_pipeline_name=derivatives_pipeline_name)
        derivatives_func_avg_to_anat_transform = init_derivatives_datasink('derivatives_func_avg_to_anat_transform',
                                                                           bids_datatype='func',
                                                                           bids_description='FuncToAnatTransform',
                                                                           derivatives_collection_dir=derivatives_collection_dir,
                                                                           derivatives_pipeline_name=derivatives_pipeline_name)
        wf.connect([
            (inputnode, derivatives_func_avg_to_anat, [('func_file', 'inputnode.original_bids_file')]),
            (func_to_anat, derivatives_func_avg_to_anat, [('outputnode.func_to_anat', 'inputnode.file_to_rename')]),
            (inputnode, derivatives_func_avg_to_anat_transform, [('func_file', 'inputnode.original_bids_file')]),
            (func_to_anat, derivatives_func_avg_to_anat_transform,
             [('outputnode.func_to_anat_composite_transform', 'inputnode.file_to_rename')]),
        ])

    # correlation matrix outputs
    if SPLIT_FUNC_INTO_SEPARATE_VOLUMES:
        if isdefined(inputnode.inputs.label_mapping):
            derivatives_label_signal_means_pkl = init_derivatives_datasink('derivatives_label_signal_means_pkl',
                                                                           bids_datatype='func',
                                                                           bids_description='LabelSignals',
                                                                           derivatives_collection_dir=derivatives_collection_dir,
                                                                           derivatives_pipeline_name=derivatives_pipeline_name)
            derivatives_label_signal_means_mat = init_derivatives_datasink('derivatives_label_signal_means_mat',
                                                                           bids_datatype='func',
                                                                           bids_description='LabelSignals',
                                                                           derivatives_collection_dir=derivatives_collection_dir,
                                                                           derivatives_pipeline_name=derivatives_pipeline_name)

            derivatives_corr_mtx_png = init_derivatives_datasink('derivatives_corr_mtx_png', bids_datatype='func',
                                                                 bids_description='CorrelationMatrix',
                                                                 derivatives_collection_dir=derivatives_collection_dir,
                                                                 derivatives_pipeline_name=derivatives_pipeline_name)
            derivatives_corr_mtx_pkl = init_derivatives_datasink('derivatives_corr_mtx_pkl', bids_datatype='func',
                                                                 bids_description='CorrelationMatrix',
                                                                 derivatives_collection_dir=derivatives_collection_dir,
                                                                 derivatives_pipeline_name=derivatives_pipeline_name)
            derivatives_corr_mtx_mat = init_derivatives_datasink('derivatives_corr_mtx_mat', bids_datatype='func',
                                                                 bids_description='CorrelationMatrix',
                                                                 derivatives_collection_dir=derivatives_collection_dir,
                                                                 derivatives_pipeline_name=derivatives_pipeline_name)
            derivatives_corr_mtx_shift = init_derivatives_datasink('derivatives_corr_mtx_shift', bids_datatype='func',
                                                                   bids_description='CorrelationMatrixShift',
                                                                   derivatives_collection_dir=derivatives_collection_dir,
                                                                   derivatives_pipeline_name=derivatives_pipeline_name)

            wf.connect([
                (inputnode, derivatives_label_signal_means_pkl, [('func_file', 'inputnode.original_bids_file')]),
                (extract_label_means, derivatives_label_signal_means_pkl,
                 [('output_file_pkl', 'inputnode.file_to_rename')]),
                (inputnode, derivatives_label_signal_means_mat, [('func_file', 'inputnode.original_bids_file')]),
                (extract_label_means, derivatives_label_signal_means_mat,
                 [('output_file_mat', 'inputnode.file_to_rename')]),
                (inputnode, derivatives_corr_mtx_png, [('func_file', 'inputnode.original_bids_file')]),
                (compute_corr_mtx, derivatives_corr_mtx_png, [('output_file_png', 'inputnode.file_to_rename')]),
                (inputnode, derivatives_corr_mtx_pkl, [('func_file', 'inputnode.original_bids_file')]),
                (compute_corr_mtx, derivatives_corr_mtx_pkl, [('output_file_pkl', 'inputnode.file_to_rename')]),
                (inputnode, derivatives_corr_mtx_mat, [('func_file', 'inputnode.original_bids_file')]),
                (compute_corr_mtx, derivatives_corr_mtx_mat, [('output_file_mat', 'inputnode.file_to_rename')]),
                (inputnode, derivatives_corr_mtx_shift, [('func_file', 'inputnode.original_bids_file')]),
                (compute_corr_mtx, derivatives_corr_mtx_shift, [('output_file_shift_png', 'inputnode.file_to_rename')]),
            ])

    return wf


if __name__ == "__main__":
    SPLIT_FUNC_INTO_SEPARATE_VOLUMES = True

    wf = init_func_processing(
        input_func_file='/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/sub-NL311F9/ses-2020021001/func/sub-NL311F9_ses-2020021001_task-rs_run-01_bold.nii.gz',
        # input_func_file = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/sub-NL311F9/ses-2020021001/func/sub-NL311F9_ses-2020021001_task-rs_run-01_bold_ORIGINAL.nii.gz',
        atlas='/home/akuurstr/Desktop/Esmin_mouse_registration/test/AMBMC_model.nii.gz',
        atlas_mask='/home/akuurstr/Desktop/Esmin_mouse_registration/test/AMBMC_model_mask.nii.gz',
        label_mapping='/softdev/akuurstr/python/modules/mousefMRIPrep/label_mapping.txt',

        derivatives_collection_dir='/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/derivatives',
        derivatives_pipeline_name='MousefMRIPrep',

        gzip_large_images=False,

        perform_func_to_anat_registration=False,
        use_masks_func_to_anat_registration=True,
        func_brain_extract_method=BrainExtractMethod.USER_PROVIDED_MASK,  # this might not work!
        func_brain_extract_template='/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/derivatives/BrainExtractionTemplatesAndProbabilityMasks/FuncTemplate_task-rs_desc-avg0p3x0p3x0p55mm20200402_bold.nii.gz',
        func_brain_extract_probability_mask='/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/derivatives/BrainExtractionTemplatesAndProbabilityMasks/FuncTemplateProbabilityMask_task-rs_desc-avg0p3x0p3x0p55mm20200402_bold.nii.gz',
        func_file_mask='/storage/akuurstr/mouse_results/april14/func_brainsuite_init/sub-NL311F9/ses-2020021001/func/brainsuite_init.nii.gz',

        mc_transform_method=MotionCorrectionTransform.DOUBLE_INTERPOLATION,
    )

    # #let's do the easy template for now
    # #wf.inputs.inputnode.atlas = '/softdev/akuurstr/python/modules/mouse_resting_state/mouse_model/commontemplate0_orientation_corrected.nii.gz'
    # #wf.inputs.inputnode.atlas_mask = '/softdev/akuurstr/python/modules/mouse_resting_state/mouse_model/model_mask.nii.gz'

    wf.inputs.inputnode.anat_file = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/derivatives/MousefMRIPrep/sub-NL311F9/ses-2020021001/anat/sub-NL311F9_ses-2020021001_acq-TurboRARE_run-1_desc-n4Corrected_T2w.nii.gz'
    wf.inputs.inputnode.anat_file_mask = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/derivatives/MousefMRIPrep/sub-NL311F9/ses-2020021001/anat/sub-NL311F9_ses-2020021001_acq-TurboRARE_run-1_desc-TemplateExtractedBrainMask_T2w.nii'
    wf.inputs.inputnode.anat_to_atlas_composite_transform = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/derivatives/MousefMRIPrep/sub-NL311F9/ses-2020021001/anat/sub-NL311F9_ses-2020021001_acq-TurboRARE_run-1_desc-AnatToAtlasTransform_T2w.h5'


    wf.base_dir = '/storage/akuurstr/mouse_pipepline_output'
    wf.config['execution']['remove_unnecessary_outputs'] = False
    # exec_graph = wf.run(updatehash=True)
    exec_graph = wf.run()

    if 0:
        # pick a node in exec_graph
        from tools.nipype_get_node_outputs import print_available_nodes, get_node

        print('available nodes:')
        print_available_nodes(exec_graph)
        print('')
        hierarchy_name = 'funcToAtlasRegistration.funcToanatRegistration.antsRegistration'
        node = get_node(exec_graph, hierarchy_name)
        print(f'{hierarchy_name} outputs:')
        print(node.outputs)
        print('')
        stop

    # investigate registrations

    from tools.nipype_get_node_outputs import get_node_output

    func_original_hierarchy_name = 'funcToAtlasRegistration.func_reference_wf.func_brain_extraction_wf.ants_brain_extraction_wf.inu_n4'
    func_original = get_node_output(exec_graph, func_original_hierarchy_name, 'output_image')

    func_warped_to_anat_hierarchy_name = 'funcToAtlasRegistration.funcToanatRegistration.antsRegistration'
    func_warped_to_anat = get_node_output(exec_graph, func_warped_to_anat_hierarchy_name, 'warped_image')

    anat_original_hiearachy_name = 'funcToAtlasRegistration.anat_brain_extraction_wf.ants_brain_extraction_wf.inu_n4'
    anat_original = get_node_output(exec_graph, anat_original_hiearachy_name, 'output_image')
    # anat_mask = get_node_output(exec_graph,'funcToAtlasRegistration.anat_brain_extraction_wf.brainsuite_brain_extraction_wf.fixBSEValue','out_file')
    anat_mask = get_node_output(exec_graph,
                                'funcToAtlasRegistration.anat_brain_extraction_wf.ants_brain_extraction_wf.thr_brainmask',
                                'output_image')

    anat_warped_to_atlas_hierarchy_name = 'funcToAtlasRegistration.anatToAtlasRegistration.antsRegistration'
    anat_warped_to_atlas = get_node_output(exec_graph, anat_warped_to_atlas_hierarchy_name, 'warped_image')

    if SPLIT_FUNC_INTO_SEPARATE_VOLUMES == True:
        func_warped_to_atlas_hierarchy_name = 'funcToAtlasRegistration.create_4d_image'
        func_warped_to_atlas = get_node_output(exec_graph, func_warped_to_atlas_hierarchy_name, 'merged_file')
    else:
        func_warped_to_atlas_hierarchy_name = 'funcToAtlasRegistration.register_func_to_atlas'
        func_warped_to_atlas = get_node_output(exec_graph, func_warped_to_atlas_hierarchy_name, 'output_image')

    atlas = wf.inputs.inputnode.atlas

    sotp
    import subprocess

    # func to anat
    try:
        subprocess.call(["itksnap", '-g', anat_original, '-o', func_warped_to_anat])
    except:
        pass

    # anat to atlast
    try:
        subprocess.call(["itksnap", '-g', atlas, '-o', anat_warped_to_atlas])
    except:
        pass

    # masked atlas
    try:
        subprocess.call(["itksnap", '-g', atlas, '-o', wf.inputs.inputnode.atlas_mask])
    except:
        pass

    # masked anat
    try:
        subprocess.call(["itksnap", '-g', anat_original, '-o', anat_mask])
    except:
        pass

    # func to atlas
    try:
        subprocess.call(["itksnap", '-g', atlas, '-o', func_warped_to_atlas])
    except:
        pass
