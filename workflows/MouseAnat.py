import argparse
from workflows.CFMMBase import CFMMWorkflow
from workflows.CFMMBIDS import BIDSAppArguments
from workflows.CFMMAnts import AntsArguments, MouseAntsRegistrationBE
from workflows.MouseBrainExtraction import MouseBrainExtractionBIDS

class MouseAnatRegistration(CFMMWorkflow):
    def __init__(self, *args, **kwargs):
        subcomponents = [BIDSAppArguments('BIDS Arguments'),
                         AntsArguments(group_name='ANTs Arguments', flag_prefix='ants_'),
                         MouseBrainExtractionBIDS(group_name='Brain Extraction',flag_prefix='be_'),
                         MouseAntsRegistrationBE(group_name='Registration', flag_prefix='reg_'),
                         ]
        super().__init__(subcomponents, *args, **kwargs)

    def add_parser_arguments(self):
        #use_masks_anat_to_atlas_registration

    def get_workflow(self, arg_dict=None):

        be_wf = self.get_subcomponent('Brain Extraction')
        inputnode = be_wf.inputs.inputnode
        # atlas
        # atlas mask


        outputnode = pe.Node(niu.IdentityInterface(fields=[
            'out_file_n4_corrected_brain_extracted',
            'out_file_n4_corrected',
            'out_file_mask',
            'out_file_registered',
            'out_file_transform',
        ]), name='outputnode')


        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        omp_nthreads = self.get_subcomponent('Nipype Arguments')._parameters['nthreads_node'].user_value
        if omp_nthreads is None or omp_nthreads < 1:
            omp_nthreads = cpu_count()

        reg_wf = self.get_subcomponent('Anat Registration')

        #datasinks

        wf.connect([
            (inputnode, bse, [('in_file', 'inputMRIFile')]),

        ])
        return wf



if __name__ == '__main__':
    # command line arguments

    cmd_args = [
        # bidsapp
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives',
        'participant',
        '--input_derivatives_dirs',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives',
        '--bids_layout_db', './brain_extract_test/bids_database',
        '--participant_label','Nl311f9',
        '--session_labels','2020021001',
        '--run_labels','01',
        '--in_file_entities_labels_string', 'acq-TurboRARE_T2w.nii.gz',
        '--ants_template_sub_label', 'AnatTemplate',
        '--ants_template_probability_mask_sub_label', 'AnatTemplateProbabilityMask',
        '--ants_template_desc_label','0p15x0p15x0p55mm20200804',
        '--brain_extract_method', 'REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK',
        #'--brain_extract_method', 'BRAINSUITE',
        #'--in_file', '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/atlases/AMBMC_model_downsampled.nii.gz',
        '--nipype_processing_dir', './brain_extract_test',
        '--keep_unnecessary_outputs',
    ]

    parser = argparse.ArgumentParser(description=__doc__)

    be_obj = MouseBrainExtractionBIDS(parser=parser)
    nipype_run_arguments = NipypeRunArguments(parser=parser)

    parser.print_help()

    args = parser.parse_args(cmd_args)
    args_dict = vars(args)

    nipype_run_arguments.populate_parameters(arg_dict=args_dict)
    be_wf = be_obj.get_workflow(arg_dict=args_dict)

    # be_wf.inputs.inputnode.in_file = '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/sub-Nl311f9/ses-2020021001/anat/sub-Nl311f9_ses-2020021001_acq-TurboRARE_run-01_T2w.nii.gz'

    # from workflows.CFMMCommon import BIDSLayoutDB
    # tmp = BIDSLayoutDB('/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids',
    #                     ['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives'],
    #                    './test_run/bids_database')
    # be_wf.inputs.inputnode.bids_layout_db = tmp

    nipype_run_arguments.run_workflow(be_wf)
