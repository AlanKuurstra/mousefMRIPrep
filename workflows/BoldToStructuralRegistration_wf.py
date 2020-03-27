from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from multiprocessing import cpu_count
from nipype.interfaces.ants import Registration


def init_bold_to_structural_registration(
        omp_nthreads = None,
        name = 'BoldToStructuralRegistration',
        mask = True
):
    #not enough detail to use BBR, so we just use ants
    wf = pe.Workflow(name)

    if omp_nthreads is None or omp_nthreads < 1:
        omp_nthreads = cpu_count()

    inputnode = pe.Node(niu.IdentityInterface(fields=['bold_reference', 'bold_reference_mask', 'structural', 'structural_mask']), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(fields=['bold_to_structural_composite_transform']), name='outputnode')

    ants_method = 'Mix'

    ants_reg = pe.Node(interface=Registration(), name='antsRegistration')
    ants_reg.inputs.output_transform_prefix = "output_"
    #ants_reg.inputs.initial_moving_transform_com =  1 #the initial translation isn't needed, these scans are right on top of each other
    ants_reg.inputs.dimension = 3
    # due to distortion should we be including affine??
    ants_reg.inputs.transforms = ['Affine', 'SyN']
    ants_reg.inputs.transform_parameters = []
    if 'Affine' in ants_reg.inputs.transforms:
        ants_reg.inputs.transform_parameters.append((0.005,))
    if 'SyN' in ants_reg.inputs.transforms:
        ants_reg.inputs.transform_parameters.append((0.005, 3.0, 0.0))
        # gradient step
        # updateFieldVarianceInVoxelSpace - smooth the deformation computed on the "updated" gradient field before this is added to previous deformations to form the "total" gradient field
        # totalFieldVarianceInVoxelSpace - smooth the deformation computed on the "total" gradient field
    ants_reg.inputs.write_composite_transform = True  # transform for each stage vs composite for entire warp
    ants_reg.inputs.collapse_output_transforms = False  # combines adjacent transforms when possible
    #ants_reg.inputs.initialize_transforms_per_stage = True  # seems to be for initializing linear transforms only

    if ants_method == 'MI':
        ants_reg.inputs.number_of_iterations = []
        if 'Affine' in ants_reg.inputs.transforms:
            ants_reg.inputs.number_of_iterations.append([10000, 10000, 10000])
        if 'SyN' in ants_reg.inputs.transforms:
            ants_reg.inputs.number_of_iterations.append([100, 100, 300])
        ants_reg.inputs.metric = ['MI'] * len(ants_reg.inputs.transforms)  # MI same as Mattes
        ants_reg.inputs.metric_weight = [1] * len(ants_reg.inputs.transforms)  # weight used if you do multimodal registration
        ants_reg.inputs.radius_or_number_of_bins = [32] * len(ants_reg.inputs.transforms)  # histogram bins for MI
        ants_reg.inputs.use_histogram_matching = [False] * len(ants_reg.inputs.transforms)
    elif ants_method == 'CC':
        ants_reg.inputs.number_of_iterations = [[100, 100, 300]] * len(ants_reg.inputs.transforms)
        ants_reg.inputs.metric = ['CC'] * len(ants_reg.inputs.transforms)
        ants_reg.inputs.metric_weight = [1] * len(ants_reg.inputs.transforms)# weight used if you do multimodal registration
        ants_reg.inputs.radius_or_number_of_bins = [4] * len(ants_reg.inputs.transforms)  # radius for CC between 2-5
        ants_reg.inputs.use_histogram_matching = [True] * len(ants_reg.inputs.transforms)
    elif ants_method == 'Mix':
        ants_reg.inputs.number_of_iterations = []
        ants_reg.inputs.metric = []
        ants_reg.inputs.metric_weight = []
        ants_reg.inputs.radius_or_number_of_bins = []
        ants_reg.inputs.use_histogram_matching = []
        if 'Affine' in ants_reg.inputs.transforms:
            ants_reg.inputs.number_of_iterations.append([10000, 10000, 10000])
            ants_reg.inputs.metric.append(['MI'])
            ants_reg.inputs.metric_weight.append([1])
            ants_reg.inputs.radius_or_number_of_bins.append([32])
            ants_reg.inputs.use_histogram_matching.append(False)
        if 'SyN' in ants_reg.inputs.transforms:
            ants_reg.inputs.number_of_iterations.append([200, 200, 500])
            ants_reg.inputs.metric.append(['MI', 'CC'])
            ants_reg.inputs.metric_weight.append([0.5, 0.5])
            ants_reg.inputs.radius_or_number_of_bins.append([32, 4])
            ants_reg.inputs.use_histogram_matching.append(True)


    ants_reg.inputs.convergence_threshold = [1.e-10] * 2  # use a negative number if you want to do all iterations and never exit
    ants_reg.inputs.convergence_window_size = [15] * 2# if the cost hasn't changed by convergence threshold in the last window size iterations, exit loop
    ants_reg.inputs.smoothing_sigmas = [[0.3, 0.2, 0]] * 2
    ants_reg.inputs.sigma_units = ['mm'] * 2  # we use mm instead of vox because we don't have isotropic voxels
    ants_reg.inputs.shrink_factors = [[1, 1, 1]]*2
    ants_reg.inputs.use_estimate_learning_rate_once = [True] * 2  # estimate the learning rate step size only at the beginning of each level. Does this override the value chosen in transform_parameters?

    ants_reg.inputs.output_warped_image = 'output_warped_image.nii.gz'
    ants_reg.n_procs = omp_nthreads
    ants_reg.inputs.verbose = True

    wf.connect([
        (inputnode, ants_reg, [('structural', 'fixed_image')]),
        (inputnode, ants_reg, [('bold_reference', 'moving_image')]),
        (ants_reg, outputnode, [('composite_transform', 'bold_to_structural_composite_transform')]),
    ])
    if mask:
        wf.connect([
            (inputnode, ants_reg, [('structural_mask', 'fixed_image_mask')]),
            (inputnode, ants_reg, [('bold_reference_mask', 'moving_image_mask')]),
        ])
    return wf

if __name__=="__main__":
    wf = init_bold_to_structural_registration()
    wf.inputs.inputnode.bold = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/sub-NL311F9/ses-2020021001/func/sub-NL311F9_ses-2020021001_task-rs_run-01_bold.nii.gz'
    wf.inputs.inputnode.anat = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/sub-NL311F9/ses-2020021001/anat/sub-NL311F9_ses-2020021001_acq-TurboRARE_run-01_T2w.nii.gz'
    wf.inputs.inputnode.anat_mask = 'None'
    wf.base_dir = 'bold2anat_wf'
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run()
