from workflows.BrainExtractionWorkflows import init_n4_bias_and_brain_extraction_wf, BrainExtractMethod
from workflows.StructuralToAtlasRegistration_wf import init_structural_to_atlas_registration
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from multiprocessing import cpu_count
from workflows.BoldReference_wf import init_bold_reference
from workflows.BoldToStructuralRegistration_wf import init_bold_to_structural_registration
from nipype.interfaces.ants import ApplyTransforms
#from nipype.interfaces.ants.utils import ImageMath # not yet out in nipype version 1.4, but doesn't even have replicateDisplacement operation
#from nipype_interfaces.ants_preprocess import ImageMath
from nipype_interfaces.ants_preprocess import ReplicateDisplacement, ReplicateImage
from nipype.interfaces.utility import Function
from nipype.interfaces.fsl import Split, Merge
from nipype_interfaces.SplitDisplacement import SplitDisplacement
from enum import Enum
from fmriprep.workflows.bold.stc import init_bold_stc_wf
class MotionCorrectionTransform(Enum):
    CONCAT_LOWRES = 1
    CONCAT_HIGHRES = 2
    DOUBLE_INTERPOLATION = 3
    NO_MC = 4

omp_nthreads = None
mem_gb = 3.0,
name = 'BoldToAtlasRegistration'

n4_bspline_fitting_distance = 20
diffusionConstant = 30
diffusionIterations = 3
edgeDetectionConstant = 0.55
radius = 2
dilateFinalMask = True

structural_brain_extract_method = BrainExtractMethod.BRAINSUITE
use_masks_structural_to_atlas_registration = True

bold_brain_extract_method = BrainExtractMethod.BRAINSUITE
use_masks_bold_to_structural_registration = True
mc_transform_method = MotionCorrectionTransform.CONCAT_HIGHRES
if mc_transform_method != MotionCorrectionTransform.NO_MC:
    perform_motion_correction = True

wf = pe.Workflow(name)

if omp_nthreads is None or omp_nthreads < 1:
    omp_nthreads = cpu_count()

inputnode = pe.Node(niu.IdentityInterface(fields=[
    'bold_file',
    'bold_file_mask',
    'bold_template',
    'bold_template_probability_mask',
    'structural',
    'structural_mask',
    'structural_template',
    'structural_template_probability_mask',
    'atlas',
    'atlas_mask']), name='inputnode')

struct_brain_extraction_wf = init_n4_bias_and_brain_extraction_wf(structural_brain_extract_method,
                                                                  name='struct_brain_extraction_wf',
                                                                  omp_nthreads=omp_nthreads,
                                                                  mem_gb=mem_gb,
                                                                  n4_bspline_fitting_distance=n4_bspline_fitting_distance,
                                                                  diffusionConstant=diffusionConstant,
                                                                  diffusionIterations=diffusionIterations,
                                                                  edgeDetectionConstant=edgeDetectionConstant,
                                                                  radius=radius,
                                                                  dilateFinalMask=dilateFinalMask,
                                                                  )
bold_reference = init_bold_reference(
    name='bold_reference_wf',
    omp_nthreads=omp_nthreads,
    mem_gb=mem_gb,
    n4_bspline_fitting_distance=n4_bspline_fitting_distance,
    diffusionConstant=diffusionConstant,
    diffusionIterations=diffusionIterations,
    edgeDetectionConstant=edgeDetectionConstant,
    radius=radius,
    dilateFinalMask=dilateFinalMask,
    perform_motion_correction=perform_motion_correction,
    brain_extract_method=bold_brain_extract_method
)

bold_to_structural = init_bold_to_structural_registration(
        omp_nthreads = omp_nthreads,
        mask = use_masks_bold_to_structural_registration
)

struct_to_atlas = init_structural_to_atlas_registration(
    omp_nthreads=omp_nthreads,
    mask=use_masks_structural_to_atlas_registration)

wf.connect([
    (inputnode, bold_reference, [('bold_file', 'inputnode.bold_file')]),
    (inputnode, struct_brain_extraction_wf, [('structural', 'inputnode.in_file')]),
    (bold_reference, bold_to_structural, [('outputnode.bold_avg_n4_corrected', 'inputnode.bold_reference')]),
    (struct_brain_extraction_wf, bold_to_structural, [('outputnode.out_file_n4_corrected', 'inputnode.structural')]),
    (inputnode, struct_to_atlas, [('atlas', 'inputnode.atlas')]),
    (struct_brain_extraction_wf, struct_to_atlas, [('outputnode.out_file_n4_corrected', 'inputnode.structural')]),
])

if bold_brain_extract_method in (BrainExtractMethod.REGISTRATION_WITH_MASK, BrainExtractMethod.REGISTRATION_NO_MASK):
    wf.connect([
        (inputnode, bold_reference, [('bold_template', 'inputnode.template')]),
        (inputnode, bold_reference,
         [('bold_template_probability_mask', 'inputnode.template_probability_mask')]),
    ])
    if bold_brain_extract_method == BrainExtractMethod.REGISTRATION_WITH_MASK:
        wf.connect([
            (inputnode, bold_reference, [('bold_file_mask', 'inputnode.in_file_mask')]),
        ])

if structural_brain_extract_method in (BrainExtractMethod.REGISTRATION_WITH_MASK, BrainExtractMethod.REGISTRATION_NO_MASK):
    wf.connect([
        (inputnode, struct_brain_extraction_wf, [('structural_template', 'inputnode.template')]),
        (inputnode, struct_brain_extraction_wf,
         [('structural_template_probability_mask', 'inputnode.template_probability_mask')]),
    ])
    if structural_brain_extract_method == BrainExtractMethod.REGISTRATION_WITH_MASK:
        wf.connect([
            (inputnode, struct_brain_extraction_wf, [('structural_mask', 'inputnode.in_file_mask')]),
        ])

if use_masks_bold_to_structural_registration:
    wf.connect([
        (struct_brain_extraction_wf, bold_to_structural, [('outputnode.out_file_mask', 'inputnode.structural_mask')]),
        (bold_reference, bold_to_structural, [('outputnode.bold_avg_mask', 'inputnode.bold_reference_mask')]),
    ])

if use_masks_structural_to_atlas_registration:
    wf.connect([
        (inputnode, struct_to_atlas, [('atlas_mask', 'inputnode.atlas_mask')]),
        (struct_brain_extraction_wf, struct_to_atlas, [('outputnode.out_file_mask', 'inputnode.structural_mask')]),
    ])


def concatenate_transform_files(apply_first = False, apply_second = False, apply_third=False):
    # ApplyTransforms wants transforms listed in reverse order of application
    transforms_reverse_order = []
    if apply_third:
        transforms_reverse_order.append(apply_third)
    if apply_second:
        transforms_reverse_order.append(apply_second)
    if apply_first:
        transforms_reverse_order.append(apply_first)
    return transforms_reverse_order



transforms_to_concatenate = pe.Node(
    Function(input_names=["apply_first", "apply_second", "apply_third"], output_names=["transforms"], function=concatenate_transform_files),
    name="transforms_to_concatenate")

concat_transforms = pe.Node(interface=ApplyTransforms(), name='concat_transforms')
concat_transforms.inputs.dimension = 3
#concat_transforms.inputs.output_image = 'bold_to_atlas_transform.h5'
concat_transforms.inputs.output_image = 'bold_to_atlas_transform.nii.gz'
concat_transforms.inputs.print_out_composite_warp_file = True
#concat_transforms.inputs.interpolation = 'Linear'

wf.connect([
    (bold_to_structural, transforms_to_concatenate, [('outputnode.bold_to_structural_composite_transform', 'apply_first')]),
    (struct_to_atlas, transforms_to_concatenate, [('outputnode.structural_to_atlas_composite_transform', 'apply_second')]),
    (transforms_to_concatenate, concat_transforms, [('transforms', 'transforms')]),
    (inputnode, concat_transforms, [('atlas', 'reference_image')]),
    (bold_reference, concat_transforms, [('outputnode.bold_avg', 'input_image')]),
])


if mc_transform_method == MotionCorrectionTransform.CONCAT_HIGHRES:
    #if we really want to do transform concatenation, we could fslsplit the bold and the mostioncorrection, and then
    # we could do a concat transform on a per volume bases and combine them back into a 4d file at the end

    split_bold = pe.Node(interface=Split(),name='split_bold')
    split_bold.inputs.dimension = 't'
    split_motion_displacement = pe.Node(interface=SplitDisplacement(),name='split_motion_displacement')

    transforms_to_concatenate2 = pe.MapNode(
        Function(input_names=["apply_first", "apply_second", "apply_third"], output_names=["transforms"], function=concatenate_transform_files),
        name="transforms_to_concatenate2", iterfield=['apply_first'])

    #might need a join node on transforms_to_concatenate2 before sending to applytransforms
    register_bold_to_atlas = pe.MapNode(interface=ApplyTransforms(),name='register_bold_to_atlas',iterfield=['input_image','transforms'])
    register_bold_to_atlas.inputs.dimension = 3

    #create_4d_image = pe.JoinNode(interface=Merge(),joinsource="register_bold_to_atlas",joinfield=["in_files"],name='create_4d_image')
    create_4d_image = pe.Node(interface=Merge(),name='create_4d_image')
    create_4d_image.inputs.dimension = 't'

    #create_4d_image.inputs.tr = 't'


    # register_bold_to_atlas.inputs.dimension = 4
    wf.connect([
        (inputnode, split_bold, [('bold_file', 'in_file')]),
        (bold_reference, split_motion_displacement, [('outputnode.motion_correction_transform', 'displacement_img')]),
        (split_motion_displacement, transforms_to_concatenate2, [('output_files', 'apply_first')]),
        (concat_transforms, transforms_to_concatenate2, [('output_image', 'apply_second')]),
        (transforms_to_concatenate2, register_bold_to_atlas, [('transforms', 'transforms')]),
        (split_bold, register_bold_to_atlas, [('out_files', 'input_image')]),
        (inputnode, register_bold_to_atlas, [('atlas', 'reference_image')]),
        (register_bold_to_atlas, create_4d_image, [('output_image', 'in_files')]),
    ])
else:
    transforms_to_concatenate2 = pe.Node(
        Function(input_names=["apply_first", "apply_second", "apply_third"], output_names=["transforms"],
                 function=concatenate_transform_files),
        name="transforms_to_concatenate2")

    register_bold_to_atlas = pe.Node(interface=ApplyTransforms(), name='register_bold_to_atlas', mem_gb=85)
    register_bold_to_atlas.inputs.output_image = 'warped.nii.gz'
    register_bold_to_atlas.inputs.interpolation = 'HammingWindowedSinc'


if mc_transform_method == MotionCorrectionTransform.CONCAT_LOWRES:
    # if atlas is low res, there's enough memory to do use ReplicateImage and ReplicateDisplacement

    replicate_atlas = pe.Node(interface=ReplicateImage(), name='replicate_atlas', mem_gb=85)
    # replicate_atlas.mem_gb = 60

    replicate_displacement = pe.Node(interface=ReplicateDisplacement(), name='replicate_displacement', mem_gb=85)
    # replicate_displacement.mem_gb = 60

    register_bold_to_atlas.inputs.dimension = 4
    wf.connect([
        (inputnode, replicate_atlas, [('atlas', 'input_3d_image')]),
        (inputnode, replicate_atlas, [('bold_file', 'bold_file')]),

        (concat_transforms, replicate_displacement, [('output_image', 'input_transform')]),
        (inputnode, replicate_displacement, [('bold_file', 'bold_file')]),

        (bold_reference, transforms_to_concatenate2, [('outputnode.motion_correction_transform', 'apply_first')]),
        (concat_transforms, transforms_to_concatenate2, [('output_image', 'apply_second')]),
        (transforms_to_concatenate2, register_bold_to_atlas, [('transforms', 'transforms')]),
        (inputnode, register_bold_to_atlas, [('bold_file', 'input_image')]),
        (replicate_atlas, register_bold_to_atlas, [('output_file', 'reference_image')]),

    ])

if mc_transform_method == MotionCorrectionTransform.DOUBLE_INTERPOLATION:
    # motion correct, but don't concat transforms. two interpolations but uses less memory
    # can still be a memory problem for very large atlases
    register_bold_to_atlas.inputs.dimension = 3
    register_bold_to_atlas.inputs.input_image_type = 3
    register_bold_to_atlas.inputs.float = True # use float instead of double to avoid memory error.
    wf.connect([
        (concat_transforms, register_bold_to_atlas, [('output_image', 'transforms')]),
        (bold_reference, register_bold_to_atlas, [('outputnode.bold_motion_corrected', 'input_image')]),
        (inputnode, register_bold_to_atlas, [('atlas', 'reference_image')]),

    ])

if mc_transform_method == MotionCorrectionTransform.NO_MC:
    # mice are in ear bars, so if motion correction is deemed unnecessary, processing time can be reduced
    register_bold_to_atlas.inputs.dimension = 3
    register_bold_to_atlas.inputs.input_image_type = 3
    wf.connect([
        (concat_transforms, register_bold_to_atlas, [('output_image', 'transforms')]),
        (inputnode, register_bold_to_atlas, [('bold_file', 'input_image')]),
        (inputnode, register_bold_to_atlas, [('atlas', 'reference_image')]),

    ])

ref_file = inputnode.inputs.
metadata = layout.get_metadata(ref_file)
run_stc = (bool(metadata.get("SliceTiming"))
if run_stc is True:  # bool('TooShort') == True, so check True explicitly
    bold_stc_wf = init_bold_stc_wf(name='bold_stc_wf', metadata=metadata)
    workflow.connect([
        (bold_reference_wf, bold_stc_wf, [
            ('outputnode.skip_vols', 'inputnode.skip_vols')]),
        (bold_stc_wf, boldbuffer, [('outputnode.stc_file', 'bold_file')]),
    ])
    if not multiecho:
        workflow.connect([
            (bold_reference_wf, bold_stc_wf, [
                ('outputnode.bold_file', 'inputnode.bold_file')])])
    else:  # for meepi, iterate through stc_wf for all workflows
        meepi_echos = boldbuffer.clone(name='meepi_echos')
        meepi_echos.iterables = ('bold_file', bold_file)
        workflow.connect([
            (meepi_echos, bold_stc_wf, [('bold_file', 'inputnode.bold_file')])])
elif not multiecho:  # STC is too short or False
    # bypass STC from original BOLD to the splitter through boldbuffer
    workflow.connect([
        (bold_reference_wf, boldbuffer, [('outputnode.bold_file', 'bold_file')])])
else:
    # for meepi, iterate over all meepi echos to boldbuffer
    boldbuffer.iterables = ('bold_file', bold_file)

if __name__ == "__main__":
    wf.inputs.inputnode.bold_file = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/sub-NL311F9/ses-2020021001/func/sub-NL311F9_ses-2020021001_task-rs_run-01_bold_truncated.nii.gz'
    wf.inputs.inputnode.structural = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/sub-NL311F9/ses-2020021001/anat/sub-NL311F9_ses-2020021001_acq-TurboRARE_run-01_T2w.nii.gz'

    #too memory intensive for testing!!
    wf.inputs.inputnode.atlas = '/home/akuurstr/Desktop/Esmin_mouse_registration/test/AMBMC_model.nii.gz'
    wf.inputs.inputnode.atlas_mask = '/home/akuurstr/Desktop/Esmin_mouse_registration/test/AMBMC_model_mask.nii.gz'

    #let's do the easy template for now
    #wf.inputs.inputnode.atlas = '/softdev/akuurstr/python/modules/mouse_resting_state/mouse_model/commontemplate0_orientation_corrected.nii.gz'
    #wf.inputs.inputnode.atlas_mask = '/softdev/akuurstr/python/modules/mouse_resting_state/mouse_model/model_mask.nii.gz'


    wf.base_dir = '/storage/akuurstr/mouse_pipepline_output'
    wf.config['execution']['remove_unnecessary_outputs'] = False
    exec_graph = wf.run()




    if 0:
        # pick a node in exec_graph
        from tools.nipype_get_node_outputs import print_available_nodes, get_node
        print('available nodes:')
        print_available_nodes(exec_graph)
        print('')
        hierarchy_name = 'BoldToAtlasRegistration.BoldToStructuralRegistration.antsRegistration'
        node = get_node(exec_graph,hierarchy_name)
        print(f'{hierarchy_name} outputs:')
        print(node.outputs)
        print('')
        stop

    # investigate registrations

    from tools.nipype_get_node_outputs import get_node_output
    func_original_hierarchy_name = 'BoldToAtlasRegistration.bold_reference_wf.func_brain_extraction_wf.brainsuite_brain_extraction_wf.inu_n4'
    func_original = get_node_output(exec_graph, func_original_hierarchy_name, 'output_image')

    func_warped_to_struct_hierarchy_name = 'BoldToAtlasRegistration.BoldToStructuralRegistration.antsRegistration'
    func_warped_to_struct = get_node_output(exec_graph, func_warped_to_struct_hierarchy_name, 'warped_image')

    struct_original_hiearachy_name = 'BoldToAtlasRegistration.struct_brain_extraction_wf.brainsuite_brain_extraction_wf.inu_n4'
    struct_original = get_node_output(exec_graph,struct_original_hiearachy_name,'output_image')

    struct_warped_to_atlas_hierarchy_name = 'BoldToAtlasRegistration.StructuralToAtlasRegistration.antsRegistration'
    struct_warped_to_atlas = get_node_output(exec_graph, struct_warped_to_atlas_hierarchy_name, 'warped_image')

    if mc_transform_method == MotionCorrectionTransform.CONCAT_HIGHRES:
        func_warped_to_atlas_hierarchy_name = 'BoldToAtlasRegistration.create_4d_image'
        func_warped_to_atlas = get_node_output(exec_graph, func_warped_to_atlas_hierarchy_name, 'merged_file')
    else:
        func_warped_to_atlas_hierarchy_name = 'BoldToAtlasRegistration.register_bold_to_atlas'
        func_warped_to_atlas = get_node_output(exec_graph, func_warped_to_atlas_hierarchy_name, 'output_image')

    atlas = wf.inputs.inputnode.atlas



    import subprocess

    # bold to struct
    subprocess.call(["itksnap", '-g', struct_original, '-o', func_warped_to_struct])

    # struct to atlast
    subprocess.call(["itksnap", '-g', atlas, '-o', struct_warped_to_atlas])

    # masked atlas
    subprocess.call(["itksnap", '-g', atlas, '-o', wf.inputs.inputnode.atlas_mask])
    #masked struct
    subprocess.call(["itksnap", '-g', struct_original, '-o', get_node_output(exec_graph,'BoldToAtlasRegistration.struct_brain_extraction_wf.brainsuite_brain_extraction_wf.fixBSEValue','out_file')])

    # bold to atlas
    subprocess.call(["itksnap", '-g', atlas, '-o', func_warped_to_atlas])


