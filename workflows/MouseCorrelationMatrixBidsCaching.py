from workflows.MouseAnatToAtlas import MouseAnatToAtlasBIDS
from workflows.MouseCorrelationMatrix import MouseCorrelationMatrixBIDS

anat_args = [
    "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids'",
    "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives'",
    "'participant'",
    '--input_derivatives_dirs',
    "['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives']",

    '--in_file_base_bids_string', "'acq-TurboRARE_T2w.nii.gz'",
    '--in_file_subject', "['Nl311f9','Nl247m1']",

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
]


func_args = [
    "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids'",
    "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives'",
    "'participant'",
    '--input_derivatives_dirs',
    "['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives']",
    '--bids_layout_db', "'./func_corrmtx_test/bids_database'",
    '--reset_db',

    '--anat_base_bids_string', "'acq-TurboRARE_T2w.nii.gz'",
    '--func_base_bids_string', "'task-rs_bold.nii.gz'",
    '--func_subject', "['Nl311f9','Nl311f10', 'Nl247m1']",
    '--func_session', "'2020021001'",
    '--func_run', "['05','02']",

    '--label_mapping',
    "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/DownsampleAtlasBIDS/label_mapping_host_0p3x0p3x0p55mm.txt'",
    '--reg_atlas',
    "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/Atlases/sub-AMBMCc57bl6_desc-ModelDownsampled.nii.gz'",

    '--reg_downsample',

    '--reg_func_antsarg_float',
    '--reg_func_preproc_be4d_brain_extract_method', "NO_BRAIN_EXTRACTION",
    '--reg_func_preproc_skip_mc',
    '--reg_func_preproc_smooth_fwhm','0.6',
    '--reg_func_preproc_smooth_brightness_threshold','20.0',
    '--reg_func_preproc_tf_highpass_sigma', '33',

    '--nipype_processing_dir', "'./func_corrmtx_test'",
]


anat2atlas = MouseAnatToAtlasBIDS()
anat2atlas.run_bids(anat_args)
# make sure the bids database is reset during the MouseCorrelationMatrix run!
corr = MouseCorrelationMatrixBIDS()
corr.run_bids(func_args)