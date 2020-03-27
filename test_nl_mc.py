from workflows.BrainExtractionWorkflows import init_n4_bias_and_brain_extraction_wf, BrainExtractMethod
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from multiprocessing import cpu_count
from nipype_interfaces.ants_preprocess import MotionCorr

# the interface that i downloaded can only to affine or rigid...and can't do stages

ants_mc = MotionCorr()
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
ants_mc.inputs.transformation_model = ['Affine', 'Syn']
ants_mc.inputs.gradient_step_length = 0.005
# the composite tranform is stored in csv. no examples on how to concat that transform or apply it to an input image.
# it's probably more used for antsMotionCorrStats
# however there are examples on how to use the transform as a displacement field (https://stnava.github.io/fMRIANTs/)
ants_mc.inputs.write_displacement = True


ants_mc.inputs.fixed_image = '/softdev/akuurstr/python/modules/mousersfMRIPrep/bold_truncated_avg.nii.gz'
ants_mc.inputs.moving_image = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/sub-NL311F9/ses-2020021001/func/sub-NL311F9_ses-2020021001_task-rs_run-01_bold_truncated.nii.gz'

res = ants_mc.run()
print(res)


#the real antsMotionCorr can do stages as follows:

antsMotionCorr -d 3
-t Affine[0.005]
-m CC[/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/sub-NL311F9/ses-2020021001/func/sub-NL311F9_ses-2020021001_task-rs_run-01_bold_truncated_avg.nii.gz,/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/sub-NL311F9/ses-2020021001/func/sub-NL311F9_ses-2020021001_task-rs_run-01_bold_truncated.nii.gz,1.0,2,Regular,0.1]
-i 20
-s 0.0
-f 1

-t SyN[0.005, 2.0, 0]
-m CC[/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/sub-NL311F9/ses-2020021001/func/sub-NL311F9_ses-2020021001_task-rs_run-01_bold_truncated_avg.nii.gz,/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/sub-NL311F9/ses-2020021001/func/sub-NL311F9_ses-2020021001_task-rs_run-01_bold_truncated.nii.gz,1.0,2]
-i 20
-s 0.0
-f 1

-n 15 -o [motcorr,warped.nii.gz,output_average.nii.gz] -u 1 -e 1 -w 1

#instead of making a correct nipype interface for antsMotionCorr...could probably just use ants
AverageImages (from ants) to make a reference
antsRegistration 4d data to 3d template