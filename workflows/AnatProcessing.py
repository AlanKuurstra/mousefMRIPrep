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


def init_anat_processing(
        # pipeline input parameters
        name='anat_processing',
        input_anat_file=None,
        atlas=None,
        atlas_mask=None,

        # anat brain extraction (only necessary if using masks in anat to atlas registration)
        anat_brain_extract_method=BrainExtractMethod.REGISTRATION_NO_INITIAL_MASK,
        # if using REGISTRATION_WITH_INITIAL_MASK or USER_PROVIDED_MASK
        anat_file_mask=None,
        # if using registration brain extraction
        anat_brain_extract_template=None,
        anat_brain_extract_template_probability_mask=None,

        # use extracted masks in anat to atlas registration?
        use_masks_anat_to_atlas_registration=True,

        # node resources
        omp_nthreads=None,
        mem_gb=50,

        # registration processing time
        interpolation="Linear",
        reduce_to_float_precision=True,

        # for brainsuite brain extraction
        n4_bspline_fitting_distance=20,
        diffusionConstant=30,
        diffusionIterations=3,
        edgeDetectionConstant=0.55,
        radius=2,
        dilateFinalMask=True,

        # for datasink
        derivatives_collection_dir=None,
        derivatives_pipeline_name='MousefMRIPrep',
):
    # set other parameters
    if use_masks_anat_to_atlas_registration and anat_brain_extract_method == BrainExtractMethod.NO_BRAIN_EXTRACTION:
        print("Warning: Not using masks in anat to atlas registration since anat_brain_extract_method = BrainExtractMethod.NO_BRAIN_EXTRACTION")
        use_masks_anat_to_atlas_registration = False

    if omp_nthreads is None or omp_nthreads < 1:
        omp_nthreads = cpu_count()


    # create workflow
    wf = pe.Workflow(name)

    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'anat_file',
        'anat_file_mask',
        'anat_template',
        'anat_template_probability_mask',
        'atlas',
        'atlas_mask',
    ]), name='inputnode')

    # NOTE: best practice is to set these inputnode values using parameters in init_func_to_atlas_registration
    # the isdefined() method is used during pipeline creation to determine which nodes to create and connect
    # if the inputnode is set after the pipeline is created, necessary nodes may not be created.

    # for example if you don't set the label_mapping in the function call:
    #   wf = init_func_to_atlas_registration(...)
    # and then try to set it later using
    #   wf.inputs.inputnode.func_file = input_func_file
    # it won't work because the correlation matrix nodes were not created during the execution of init_func_to_atlas_registration

    if input_anat_file is not None:
        inputnode.inputs.anat_file = input_anat_file
    if anat_file_mask is not None:
        inputnode.inputs.anat_file_mask = anat_file_mask
    if atlas is not None:
        inputnode.inputs.atlas = atlas
    if atlas_mask is not None:
        inputnode.inputs.atlas_mask = atlas_mask

    if anat_brain_extract_template is not None:
        inputnode.inputs.anat_template = anat_brain_extract_template
    if anat_brain_extract_template_probability_mask is not None:
        inputnode.inputs.anat_template_probability_mask = anat_brain_extract_template_probability_mask


    preprocess_anat_wf = init_n4_bias_and_brain_extraction_wf(anat_brain_extract_method,
                                                              name='preprocess_anat',
                                                              n4_bspline_fitting_distance=n4_bspline_fitting_distance,
                                                              diffusionConstant=diffusionConstant,
                                                              diffusionIterations=diffusionIterations,
                                                              edgeDetectionConstant=edgeDetectionConstant,
                                                              radius=radius,
                                                              dilateFinalMask=dilateFinalMask,
                                                              omp_nthreads=omp_nthreads,
                                                              mem_gb=mem_gb,
                                                              )

    anat_to_atlas = init_anat_to_atlas_registration(
        mask=use_masks_anat_to_atlas_registration,
        reduce_to_float_precision=reduce_to_float_precision,
        interpolation=interpolation,
        omp_nthreads=omp_nthreads,
        mem_gb=mem_gb,
    )

    wf.connect([
        (inputnode, preprocess_anat_wf, [('anat_file', 'inputnode.in_file')]),
        (inputnode, anat_to_atlas, [('atlas', 'inputnode.atlas')]),
        (preprocess_anat_wf, anat_to_atlas, [('outputnode.out_file_n4_corrected', 'inputnode.anat')]),
    ])


    if anat_brain_extract_method in (
    BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK, BrainExtractMethod.REGISTRATION_NO_INITIAL_MASK, BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK):
        wf.connect([
            (inputnode, preprocess_anat_wf, [('anat_template', 'inputnode.template')]),
            (inputnode, preprocess_anat_wf,
             [('anat_template_probability_mask', 'inputnode.template_probability_mask')]),
        ])
    if anat_brain_extract_method in (BrainExtractMethod.USER_PROVIDED_MASK,BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK):
        wf.connect([
            (inputnode, preprocess_anat_wf, [('anat_file_mask', 'inputnode.in_file_mask')]),
        ])




    if use_masks_anat_to_atlas_registration:
        wf.connect([
            (inputnode, anat_to_atlas, [('atlas_mask', 'inputnode.atlas_mask')]),
            (preprocess_anat_wf, anat_to_atlas, [('outputnode.out_file_mask', 'inputnode.anat_mask')]),
        ])


    # datasinks

    derivatives_anat_n4_corrected = init_derivatives_datasink('derivatives_anat_n4_corrected', bids_datatype='anat',
                                                              bids_description='n4Corrected',
                                                              derivatives_collection_dir=derivatives_collection_dir,
                                                              derivatives_pipeline_name=derivatives_pipeline_name)
    derivatives_anat_to_atlas = init_derivatives_datasink('derivatives_anat_to_atlas', bids_datatype='anat',
                                                          bids_description='AnatToAtlas',
                                                          derivatives_collection_dir=derivatives_collection_dir,
                                                          derivatives_pipeline_name=derivatives_pipeline_name)
    derivatives_anat_to_atlas_transform = init_derivatives_datasink('derivatives_anat_to_atlas_transform',
                                                                    bids_datatype='anat',
                                                                    bids_description='AnatToAtlasTransform',
                                                                    derivatives_collection_dir=derivatives_collection_dir,
                                                                    derivatives_pipeline_name=derivatives_pipeline_name)



    wf.connect([
        (inputnode, derivatives_anat_n4_corrected, [('anat_file', 'inputnode.original_bids_file')]),
        (preprocess_anat_wf, derivatives_anat_n4_corrected,
         [('outputnode.out_file_n4_corrected', 'inputnode.file_to_rename')]),
        (inputnode, derivatives_anat_to_atlas, [('anat_file', 'inputnode.original_bids_file')]),
        (anat_to_atlas, derivatives_anat_to_atlas, [('outputnode.anat_to_atlas', 'inputnode.file_to_rename')]),
        (inputnode, derivatives_anat_to_atlas_transform, [('anat_file', 'inputnode.original_bids_file')]),
        (anat_to_atlas, derivatives_anat_to_atlas_transform,
         [('outputnode.anat_to_atlas_composite_transform', 'inputnode.file_to_rename')]),
    ])


    if anat_brain_extract_method != BrainExtractMethod.NO_BRAIN_EXTRACTION:
        method = ''
        if anat_brain_extract_method == BrainExtractMethod.BRAINSUITE:
            method = 'Brainsuite'
        elif anat_brain_extract_method == BrainExtractMethod.REGISTRATION_NO_INITIAL_MASK:
            method = 'NoInitTemplateExtracted'
        elif anat_brain_extract_method == BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK:
            method = 'BrainsuiteInitTemplateExtracted'
        elif anat_brain_extract_method == BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK:
            method = 'UserInitTemplateExtracted'
        derivatives_anat_brain_mask = init_derivatives_datasink('derivatives_anat_brain_mask',
                                                                bids_datatype='anat',
                                                                bids_description=f'{method}BrainMask',
                                                                derivatives_collection_dir=derivatives_collection_dir,
                                                                derivatives_pipeline_name=derivatives_pipeline_name)

        wf.connect([
            (inputnode, derivatives_anat_brain_mask, [('anat_file', 'inputnode.original_bids_file')]),
            (preprocess_anat_wf, derivatives_anat_brain_mask,
             [('outputnode.out_file_mask', 'inputnode.file_to_rename')]),
        ])

    return wf


if __name__ == "__main__":

    wf = init_anat_processing(
        input_anat_file='/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/sub-NL311F9/ses-2020021001/anat/sub-NL311F9_ses-2020021001_acq-TurboRARE_run-01_T2w.nii.gz',
        atlas='/home/akuurstr/Desktop/Esmin_mouse_registration/test/AMBMC_model.nii.gz',
        atlas_mask='/home/akuurstr/Desktop/Esmin_mouse_registration/test/AMBMC_model_mask.nii.gz',

        derivatives_collection_dir='/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/derivatives',
        derivatives_pipeline_name='MousefMRIPrep',
        anat_brain_extract_template='/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/derivatives/BrainExtractionTemplatesAndProbabilityMasks/AnatTemplate_acq-TurboRARE_desc-0p15x0p15x0p55mm20200402_T2w.nii.gz',
        anat_brain_extract_template_probability_mask='/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/derivatives/BrainExtractionTemplatesAndProbabilityMasks/AnatTemplateProbabilityMask_acq-TurboRARE_desc-0p15x0p15x0p55mm20200402_T2w.nii.gz',

    )

    # #let's do the easy template for now
    # #wf.inputs.inputnode.atlas = '/softdev/akuurstr/python/modules/mouse_resting_state/mouse_model/commontemplate0_orientation_corrected.nii.gz'
    # #wf.inputs.inputnode.atlas_mask = '/softdev/akuurstr/python/modules/mouse_resting_state/mouse_model/model_mask.nii.gz'

    wf.base_dir = '/storage/akuurstr/mouse_pipepline_output'
    wf.config['execution']['remove_unnecessary_outputs'] = False
    # exec_graph = wf.run(updatehash=True)
    exec_graph = wf.run()

