from workflows.CFMMBase import CFMMWorkflow
from workflows.CFMMCommon import NipypeWorkflowArguments, NipypeRunArguments, get_node_inputs_to_list, get_node_existing_inputs_to_list
from workflows.CFMMBIDS import BIDSAppArguments, get_node_get_input_file_entities_labels_dict, \
    get_node_bids_file_multiplexer, get_node_batch_update_entities_labels_dict, get_node_update_entities_labels_dict
from workflows.CFMMBrainSuite import MouseBse
from workflows.CFMMAnts import AntsArguments, CFMMApplyTransforms, CFMMThresholdImage, MouseAntsRegistrationBE, \
    MouseN4BiasFieldCorrection
from nipype_interfaces.DerivativesDatasink import get_node_derivatives_datasink
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from multiprocessing import cpu_count
from nipype.interfaces.fsl import ImageMaths, CopyGeom, ApplyMask
from workflows.CFMMEnums import BrainExtractMethod
import configargparse as argparse
import os


# explain
# _parameters
# _subcomponents
# exclude_list
# parent changing default value for child parameters (eg. nipype arguments)
# parent hiding child parameters and setting values with it's own (eg. in_file, brain_extract_method)

# add arguments (order of super() )
# validate arguments (order of super() )
# get workflow
#   inputnode
#   outputnode
#   conditional connections using command line parameters (if options is guaranteed to come from the commandline and not set upstream)
#   conditional connections using multiplexing node (multiplex inputnode)

# eg in_file should be  multiplexed, while brain_extract_method should be coming from command line and hidden from input_node
# do we need an in_node?
# not more than one parent can override child.
# if not used in connections, should not be in inputnode ... if used for flow control


class MouseBrainSuiteBrainExtraction(CFMMWorkflow):
    def __init__(self, *args, **kwargs):
        subcomponents = [NipypeWorkflowArguments(group_name='Nipype Arguments',
                                                 flag_prefix='nipype_',
                                                 exclude_list=['nthreads_mapnode','mem_gb_mapnode']),
                         MouseBse(group_name='BrainSuite Bse', flag_prefix='bse_'),
                         ]
        self.outputs = ['out_file_brain_extracted', 'out_file_mask']
        super().__init__(subcomponents, *args, **kwargs)

    def add_parser_arguments(self):
        self.add_parser_argument('in_file',
                                 help='Specify location of the input file for brain extraction.')
        super().add_parser_arguments()

    def get_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        omp_nthreads = self.get_subcomponent('Nipype Arguments')._parameters['nthreads_node'].user_value
        if omp_nthreads is None or omp_nthreads < 1:
            omp_nthreads = cpu_count()

        bse = self.get_subcomponent('BrainSuite Bse').get_node(name='BSE', n_procs=omp_nthreads)

        # default behaviour of brainsuite is to rotate to LPI orientation
        # this can be overridden by using the noRotate option, however this option will create a nifti with inconsistent
        # qform and sform values.  To fix this, copy the header information from the original image to the mask using fsl.
        fix_bse_orientation = pe.Node(interface=CopyGeom(), name='fixBSEOrientation', n_procs=omp_nthreads)

        # brainsuite outputs mask value as 255, change it to 1
        fix_bse_value = pe.Node(interface=ImageMaths(), name='fixBSEValue', n_procs=omp_nthreads)
        fix_bse_value.inputs.op_string = '-div 255'

        apply_mask = pe.Node(ApplyMask(), name='apply_mask', n_procs=omp_nthreads)

        inputnode, outputnode, wf = self.get_io_and_workflow()

        wf.connect([
            (inputnode, bse, [('in_file', 'inputMRIFile')]),

            (inputnode, fix_bse_orientation, [('in_file', 'in_file')]),
            (bse, fix_bse_orientation, [('outputMaskFile', 'dest_file')]),

            (fix_bse_orientation, fix_bse_value, [('out_file', 'in_file')]),

            (fix_bse_value, apply_mask, [('out_file', 'mask_file')]),
            (inputnode, apply_mask, [('in_file', 'in_file')]),

            (apply_mask, outputnode, [('out_file', 'out_file_brain_extracted')]),
            (fix_bse_value, outputnode, [('out_file', 'out_file_mask')]),
        ])
        return wf


class MouseAntsBrainExtraction(CFMMWorkflow):
    def __init__(self, *args, **kwargs):
        subcomponents = [NipypeWorkflowArguments(group_name='Nipype Arguments',
                                                 flag_prefix='nipype_',
                                                 exclude_list=['nthreads_mapnode','mem_gb_mapnode']),
                         AntsArguments(group_name='ANTs Arguments', flag_prefix='ants_'),
                         MouseAntsRegistrationBE(group_name='ANTs Registration', flag_prefix='reg_'),
                         CFMMApplyTransforms(group_name='Apply Transforms', flag_prefix='apply_'),
                         CFMMThresholdImage(group_name='Threshold Image', flag_prefix='thresh_'),
                         ]
        self.outputs = ['out_file_brain_extracted', 'out_file_mask']
        super().__init__(subcomponents, *args, **kwargs)

    def add_parser_arguments(self):
        self.add_parser_argument('in_file',
                                 help='Explicitly specify location of the input file for brain extraction.')

        self.add_parser_argument('in_file_mask',
                                 help='Explicitly specify location of an input file mask used in registration based brain extraction.')

        self.add_parser_argument('template',
                                 help='Explicitly specify location of the template used in registration based brain extraction.')

        self.add_parser_argument('template_probability_mask',
                                 help='Explicitly specify location of the probability mask used in registration based brain extraction.')

        self.add_parser_argument('brain_extract_method',
                                 choices=list(BrainExtractMethod),
                                 default=BrainExtractMethod.NO_BRAIN_EXTRACTION.name,
                                 type=BrainExtractMethod.argparse_convert,
                                 help="Brain extraction method for image.",
                                 add_to_inputnode=False, )
        super().add_parser_arguments()

    def validate_parameters(self):
        template = self._parameters['template']
        template_probability_mask = self._parameters['template_probability_mask']
        if ((template.user_value is not None) or (template_probability_mask.user_value is not None)) \
                and \
                ((template.user_value is None) or (template_probability_mask.user_value is None)):
            self.parser.error(
                f"{template.parser_flag} and {template_probability_mask.parser_flag} must be defined together")

    def get_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        omp_nthreads = self.get_subcomponent('Nipype Arguments')._parameters['nthreads_node'].user_value
        if omp_nthreads is None or omp_nthreads < 1:
            omp_nthreads = cpu_count()

        ants_reg = self.get_subcomponent('ANTs Registration').get_node(n_procs=omp_nthreads, name='ants_reg')
        apply_transform = self.get_subcomponent('Apply Transforms').get_node(name='antsApplyTransforms')
        thr_brainmask = self.get_subcomponent('Threshold Image').get_node(name='thr_brainmask', n_procs=omp_nthreads)
        apply_mask = pe.Node(ApplyMask(), name='apply_mask', n_procs=omp_nthreads)

        # USE ATROPOS TO CLEAN UP??
        # atropos doesn't seem to do so well on T2w mouse data
        # atropos = pe.Node(Atropos(
        #     dimension=3,
        #     initialization='KMeans',
        #     number_of_tissue_classes=3,
        #     n_iterations=3,
        #     convergence_threshold=0.0,
        #     mrf_radius=[1, 1, 1],
        #     mrf_smoothing_factor=0.1,
        #     likelihood_model='Gaussian',
        #     use_random_seed=True),
        #     name='01_atropos', n_procs=nthreads_node)

        inputnode, outputnode, wf = self.get_io_and_workflow()

        wf.connect([
            (inputnode, ants_reg, [('in_file', 'fixed_image')]),
            (inputnode, ants_reg, [('template', 'moving_image')]),

            (inputnode, apply_transform, [('template_probability_mask', 'input_image')]),
            (ants_reg, apply_transform, [('composite_transform', 'transforms')]),
            (inputnode, apply_transform, [('in_file', 'reference_image')]),

            (apply_transform, thr_brainmask, [('output_image', 'input_image')]),

            (thr_brainmask, apply_mask, [('output_image', 'mask_file')]),
            (inputnode, apply_mask, [('in_file', 'in_file')]),

            (apply_mask, outputnode, [('out_file', 'out_file_brain_extracted')]),
            (thr_brainmask, outputnode, [('output_image', 'out_file_mask')]),

            # (thr_brainmask, atropos, [('output_image', 'mask_image')]),
            # (apply_mask, atropos, [('out_file', 'intensity_images')]),
        ])
        if self._parameters['brain_extract_method'].user_value in (
                BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK,
                BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK):
            # per stage masks are possible with fixed_image_masks and moving_image_masks (note the s at the end of masks)
            # we go with the simpler fixed_image_mask which is applied to all stages
            # note: that moving_image_mask depends on the existence of fixed_image_mask, although I don't believe
            # this is true for the per stage version moving_image_masks
            # for this reason, we only connect the moving_image_mask if the fixed_image_mask is present
            wf.connect([
                (inputnode, ants_reg, [('in_file_mask', 'fixed_image_mask')]),
                (inputnode, ants_reg, [('template_probability_mask', 'moving_image_mask')]),
            ])

        return wf


class MouseBrainSuiteBrainExtractionBIDS(MouseBrainSuiteBrainExtraction):
    def __init__(self, *args, **kwargs):
        self.add_subcomponent(BIDSAppArguments('BIDS Arguments'))
        super().__init__(*args, **kwargs)
        self.outputs = {
            'out_file_brain_extracted': 'BrainSuiteBrainExtracted',
            'out_file_mask': 'BrainSuiteBrainMask'
        }

    def add_parser_arguments(self):
        super().add_parser_arguments()
        self.add_parser_argument('in_file_entities_labels_string',
                                 help=f'BIDS entity-label search string for in_file. Some entities are reused if doing a bids search for the in_file mask, template, or template probability mask. The in_file search can be overridden by --{self._parameters["in_file"].parser_flag}.')

        self.get_subcomponent('BIDS Arguments').modify_parser_argument('analysis_level', 'choices', ['participant'])

    def validate_parameters(self):
        super().validate_parameters()
        # warning if in_file overrides in_file_entities_labels_string

    def get_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        # depending on the values of inputnode, we will either use the explicitly defined
        # input files or we will do a bids search for the input image using input_bids_entities_string
        # we can't make conditional workflow connections to the ANTs workflow based on if statements involving the
        # inputnode.inputs attributes because those attributes can be overridden at runtime by upstream connections
        # to inputnode. Instead we must make a multiplexer node which takes the inputnode as input and provides an
        # output based on conditional statements inside the multiplexer node. Note: choose_input() is the multiplexer

        input_file_entities_labels_dict = get_node_get_input_file_entities_labels_dict(
            'input_file_entities_labels_dict')
        choose_in_file = get_node_bids_file_multiplexer('choose_in_file')

        (inputnode,derivatives_original_file), outputnode, wf, brainsuite_be_wf = self.get_io_and_workflow(get_superclass_workflow=True,
                                                                               superclass_inputnode_exclude_list=[
                                                                                   'in_file'],
                                                                               output_bids_derivatives=True, )

        wf.connect([
            # deciding between in_file and bids search
            (inputnode, input_file_entities_labels_dict, [('participant_label', 'participant_label')]),
            (inputnode, input_file_entities_labels_dict, [('session_labels', 'session_labels')]),
            (inputnode, input_file_entities_labels_dict, [('run_labels', 'run_labels')]),
            (inputnode, input_file_entities_labels_dict, [('in_file_entities_labels_string', 'entities_string')]),

            (input_file_entities_labels_dict, choose_in_file, [('entities_labels_dict', 'entities_labels_dict')]),
            (inputnode, choose_in_file, [('in_file', 'input_file')]),
            (inputnode, choose_in_file, [('bids_layout_db', 'bids_layout_db')]),
            (choose_in_file, derivatives_original_file, [('chosen_file', 'bids_file')]),

            # connect chosen file brainsuite_be_wf
            (choose_in_file, brainsuite_be_wf, [('chosen_file', 'inputnode.in_file')]),
            (brainsuite_be_wf, outputnode, [('outputnode.out_file_brain_extracted', 'out_file_brain_extracted')]),
            (brainsuite_be_wf, outputnode, [('outputnode.out_file_mask', 'out_file_mask')]),
        ])
        return wf


class MouseAntsBrainExtractionBIDS(MouseAntsBrainExtraction):
    def __init__(self, *args, **kwargs):
        self.add_subcomponent(BIDSAppArguments('BIDS Arguments'))
        super().__init__(*args, **kwargs)
        self.outputs = {
            'out_file_brain_extracted': 'ANTsBrainExtracted',
            'out_file_mask': 'ANTsBrainMask'
        }

    def add_parser_arguments(self):
        super().add_parser_arguments()
        self.add_parser_argument('in_file_entities_labels_string',
                                 help=f'BIDS entity-label search string for in_file. Some entities are reused if doing a bids search for the in_file mask, template, or template probability mask. The in_file search can be overridden by --{self._parameters["in_file"].parser_flag}.')
        self.add_parser_argument('in_file_mask_desc_label',
                                 help=f'BIDS description label used to search for in_file_mask. Overridden by --{self._parameters["in_file_mask"].parser_flag}.')
        self.add_parser_argument('template_sub_label',
                                 help=f'BIDS subject label used to search for the template. Overridden by --{self._parameters["template"].parser_flag}.')
        self.add_parser_argument('template_probability_mask_sub_label',
                                 help=f'BIDS subject label used to search for the template probability mask. Overridden by --{self._parameters["template_probability_mask"].parser_flag}.')
        self.add_parser_argument('template_desc_label',
                                 help=f'BIDS description label used to search for the template and probability mask. Overridden by --{self._parameters["template"].parser_flag} and --{self._parameters["template_probability_mask"].parser_flag}.')
        self.get_subcomponent('BIDS Arguments').modify_parser_argument('analysis_level', 'choices', ['participant'])

    def validate_parameters(self):
        super().validate_parameters()
        # warning if :
        # in_file overrides in_file_entities_labels_string
        # template overrides template_sub_label and template_desc_label
        # template_probability_mask overrides template_probability_mask_sub_label and template_desc_label
        #
        # depending on brain extraction method, warning if
        # in_file_mask overrides in_file_mask_desc_label

        template = self._parameters['template']
        template_probability_mask = self._parameters['template_probability_mask']
        template_bids_desc = self._parameters['template_desc_label']
        if ((template.user_value is not None) and (template_probability_mask.user_value is not None)):
            if template_bids_desc.user_value is not None:
                print(
                    f"Waring: overriding {template_bids_desc.parser_flag}={template_bids_desc.user_value} search with {template.parser_flag} {template.user_value} and {template_probability_mask.parser_flag} {template_probability_mask.user_value}")

    def get_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        input_file_entities_labels_dict = get_node_get_input_file_entities_labels_dict(
            'input_file_entities_labels_dict')

        choose_in_file = get_node_bids_file_multiplexer('choose_in_file')

        remove_in_file_extension = get_node_batch_update_entities_labels_dict('remove_in_file_extension')
        remove_in_file_extension.inputs.remove_entities_list = ['extension']
        in_file_mask_entities_dict = get_node_update_entities_labels_dict('in_file_mask_entities_dict')
        in_file_mask_entities_dict.inputs.entity = 'desc'
        choose_in_file_mask = get_node_bids_file_multiplexer('choose_in_file_mask')

        template_base_entities_dict = get_node_batch_update_entities_labels_dict('template_base_entities_dict')
        template_base_entities_dict.inputs.remove_entities_list = ['subject', 'session', 'run', 'datatype', 'extension']

        template_labels_list = get_node_existing_inputs_to_list('template_labels_list')
        template_entities_dict = get_node_update_entities_labels_dict('template_entities_dict')
        template_entities_dict.inputs.entity = ['subject', 'desc']
        choose_template = get_node_bids_file_multiplexer('choose_template')

        template_probability_mask_labels_list = get_node_existing_inputs_to_list('template_probability_mask_labels_list')
        template_probability_mask_entities_dict = get_node_update_entities_labels_dict(
            'template_probability_mask_entities_dict')
        template_probability_mask_entities_dict.inputs.entity = ['subject', 'desc']
        choose_template_probability_mask = get_node_bids_file_multiplexer('choose_template_probability_mask')

        (inputnode,derivatives_original_file), outputnode, wf, ants_be_wf = self.get_io_and_workflow(get_superclass_workflow=True,
                                                                         superclass_inputnode_exclude_list=['in_file',
                                                                                                            'in_file_mask',
                                                                                                            'template',
                                                                                                            'template_probability_mask'
                                                                                                            ],
                                                                         output_bids_derivatives=True)




        # acrobatics for deciding between in_file and bids search
        # not a fan of this
        wf.connect([
            (inputnode, input_file_entities_labels_dict, [('participant_label', 'participant_label')]),
            (inputnode, input_file_entities_labels_dict, [('session_labels', 'session_labels')]),
            (inputnode, input_file_entities_labels_dict, [('run_labels', 'run_labels')]),
            (inputnode, input_file_entities_labels_dict, [('in_file_entities_labels_string', 'entities_string')]),

            (input_file_entities_labels_dict, choose_in_file, [('entities_labels_dict', 'entities_labels_dict')]),
            (inputnode, choose_in_file, [('in_file', 'input_file')]),
            (inputnode, choose_in_file, [('bids_layout_db', 'bids_layout_db')]),
            (choose_in_file, derivatives_original_file, [('chosen_file', 'bids_file')]),

            (input_file_entities_labels_dict, remove_in_file_extension,
             [('entities_labels_dict', 'entities_labels_dict')]),

            (remove_in_file_extension, in_file_mask_entities_dict, [('entities_labels_dict', 'entities_labels_dict')]),
            (inputnode, in_file_mask_entities_dict, [('in_file_mask_desc_label', 'label')]),
            (in_file_mask_entities_dict, choose_in_file_mask, [('entities_labels_dict', 'entities_labels_dict')]),
            (inputnode, choose_in_file_mask, [('in_file_mask', 'input_file')]),
            (inputnode, choose_in_file_mask, [('bids_layout_db', 'bids_layout_db')]),

            (input_file_entities_labels_dict, template_base_entities_dict,
             [('entities_labels_dict', 'entities_labels_dict')]),

            (inputnode, template_labels_list, [('template_sub_label', 'input1')]),
            (inputnode, template_labels_list, [('template_desc_label', 'input2')]),
            (template_base_entities_dict, template_entities_dict, [('entities_labels_dict', 'entities_labels_dict')]),
            (template_labels_list, template_entities_dict, [('return_list', 'label')]),
            (template_entities_dict, choose_template, [('entities_labels_dict', 'entities_labels_dict')]),
            (inputnode, choose_template, [('template', 'input_file')]),
            (inputnode, choose_template, [('bids_layout_db', 'bids_layout_db')]),

            (inputnode, template_probability_mask_labels_list, [('template_probability_mask_sub_label', 'input1')]),
            (inputnode, template_probability_mask_labels_list, [('template_desc_label', 'input2')]),
            (template_base_entities_dict, template_probability_mask_entities_dict,
             [('entities_labels_dict', 'entities_labels_dict')]),
            (
                template_probability_mask_labels_list, template_probability_mask_entities_dict,
                [('return_list', 'label')]),
            (template_probability_mask_entities_dict, choose_template_probability_mask,
             [('entities_labels_dict', 'entities_labels_dict')]),
            (inputnode, choose_template_probability_mask, [('template_probability_mask', 'input_file')]),
            (inputnode, choose_template_probability_mask, [('bids_layout_db', 'bids_layout_db')]),
        ])




        wf.connect([
            # connect chosen file ants_be_wf
            (choose_in_file, ants_be_wf, [('chosen_file', 'inputnode.in_file')]),
            (choose_in_file_mask, ants_be_wf, [('chosen_file', 'inputnode.in_file_mask')]),
            (choose_template, ants_be_wf, [('chosen_file', 'inputnode.template')]),
            (choose_template_probability_mask, ants_be_wf, [('chosen_file', 'inputnode.template_probability_mask')]),

            (ants_be_wf, outputnode, [('outputnode.out_file_brain_extracted', 'out_file_brain_extracted')]),
            (ants_be_wf, outputnode, [('outputnode.out_file_mask', 'out_file_mask')]),
        ])

        return wf


class MouseBrainExtractionBIDS(CFMMWorkflow):
    def __init__(self, *args, **kwargs):
        subcomponents = [BIDSAppArguments('BIDS Arguments'),
                         NipypeWorkflowArguments(group_name='Nipype Arguments',
                                                 exclude_list=['nthreads_mapnode','mem_gb_mapnode']),
                         MouseN4BiasFieldCorrection(group_name='N4 Bias Field Correction', flag_prefix='n4_'),
                         MouseBrainSuiteBrainExtractionBIDS(group_name='BrainSuite Brain Extraction BIDS',
                                                            flag_prefix='bs_'),
                         MouseAntsBrainExtractionBIDS(group_name='ANTs Brain Extraction BIDS',
                                                      flag_prefix='ants_'),
                         ]
        self.outputs = {
            'out_file_n4_corrected': 'N4Corrected',
            'out_file_n4_corrected_brain_extracted': 'N4CorrectedBrainExtracted',
            'out_file_mask': 'BrainMask',
            'ants_out_file_n4_corrected_brain_extracted': 'N4CorrectedANTsBrainExtracted',
            'ants_out_file_mask': 'ANTsBrainMask',
            'brainsuite_out_file_n4_corrected_brain_extracted': 'N4CorrectedBrainSuiteBrainExtracted',
            'brainsuite_out_file_mask': 'BrainSuiteBrainMask',
        }
        super().__init__(subcomponents, *args, **kwargs)

    def add_parser_arguments(self):
        super().add_parser_arguments()
        self.get_subcomponent('BIDS Arguments').modify_parser_argument('analysis_level', 'choices', ['participant'])
        self.add_parser_argument('in_file',
                                 help='Explicitly specify location of the input file for brain extraction.',
                                 override_parameters=[
                                     ('in_file', 'BrainSuite Brain Extraction BIDS'),
                                     ('in_file', 'ANTs Brain Extraction BIDS')
                                 ],
                                 )
        self.add_parser_argument('in_file_entities_labels_string',
                                 help=f'BIDS entity-label search string for in_file. Some entities are reused if doing a bids search for the in_file mask, template, or template probability mask. The in_file search can be overridden by --{self._parameters["in_file"].parser_flag}.',
                                 override_parameters=[
                                     ('in_file_entities_labels_string', 'BrainSuite Brain Extraction BIDS'),
                                     ('in_file_entities_labels_string', 'ANTs Brain Extraction BIDS')
                                 ],
                                 )
        self.add_parser_argument('brain_extract_method',
                                 choices=list(BrainExtractMethod),
                                 default=BrainExtractMethod.NO_BRAIN_EXTRACTION.name,
                                 type=BrainExtractMethod.argparse_convert,
                                 help="Brain extraction method for image.",
                                 add_to_inputnode=False,
                                 override_parameters=[
                                     ('brain_extract_method', 'ANTs Brain Extraction BIDS')
                                 ],
                                 )
        parent_nipype = self.get_subcomponent('Nipype Arguments')
        child_nipype = self.get_subcomponent(['ANTs Brain Extraction BIDS', 'Nipype Arguments'])
        self.replace_defaults(parent_nipype, child_nipype)

    def validate_parameters(self):
        super().validate_parameters()
        # warning if in_file overrides in_file_entities_labels_string
        brain_extraction_parameter = self._parameters['brain_extract_method']

        brain_extraction_method = brain_extraction_parameter.user_value
        if brain_extraction_method in (
                BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK, BrainExtractMethod.REGISTRATION_NO_INITIAL_MASK,
                BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK):

            ants_be_base_obj = self.get_subcomponent('ANTs Brain Extraction BIDS')
            template = ants_be_base_obj._parameters['template']
            template_probability_mask = ants_be_base_obj._parameters['template_probability_mask']
            template_bids_entities = ants_be_base_obj._parameters['template_sub_label']

            if template.user_value is None and template_probability_mask.user_value is None and template_bids_entities.user_value is None:
                self.parser.error(
                    f'When using {brain_extraction_parameter.parser_flag}={brain_extraction_parameter.user_value}, \n'
                    f'either {template_bids_entities.parser_flag} or {template.parser_flag} and {template_probability_mask.parser_flag} must be defined.')

    def get_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        omp_nthreads = self.get_subcomponent('Nipype Arguments')._parameters['nthreads_node'].user_value
        if omp_nthreads is None or omp_nthreads < 1:
            omp_nthreads = cpu_count()

        input_file_entities_labels_dict = get_node_get_input_file_entities_labels_dict(
            'input_file_entities_labels_dict')
        # what if choose_in_file uses bids and returns multiple files???
        choose_in_file = get_node_bids_file_multiplexer('choose_in_file')

        n4 = self.get_subcomponent('N4 Bias Field Correction').get_node(n_procs=omp_nthreads, name='n4')
        ants_wf = self.get_subcomponent('ANTs Brain Extraction BIDS').get_workflow()
        brainsuite_wf = self.get_subcomponent('BrainSuite Brain Extraction BIDS').get_workflow()

        (inputnode,derivatives_original_file), outputnode, wf = self.get_io_and_workflow(overridden_inputnode_exclude_list=['in_file', ],
                                                             output_bids_derivatives=True)

        wf.connect([
            (inputnode, input_file_entities_labels_dict, [('participant_label', 'participant_label')]),
            (inputnode, input_file_entities_labels_dict, [('session_labels', 'session_labels')]),
            (inputnode, input_file_entities_labels_dict, [('run_labels', 'run_labels')]),
            (inputnode, input_file_entities_labels_dict, [('in_file_entities_labels_string', 'entities_string')]),
            (input_file_entities_labels_dict, choose_in_file, [('entities_labels_dict', 'entities_labels_dict')]),
            (inputnode, choose_in_file, [('in_file', 'input_file')]),
            (inputnode, choose_in_file, [('bids_layout_db', 'bids_layout_db')]),
            (choose_in_file,derivatives_original_file, [('chosen_file','bids_file')]),
            (choose_in_file, n4, [('chosen_file', 'input_image')]),
            (n4, outputnode, [('output_image', 'out_file_n4_corrected')]),
        ])

        brain_extraction_method = self._parameters['brain_extract_method'].user_value

        # these conditionals do not need to be inside a node because they depend on command line values and not on
        # node values - ie. they can't be overridden the same way a node's inputs attributes can be
        # ref. choose_template() in MouseAntsBrainExtractionBIDSAbstractWorkflow.get_workflow()
        if brain_extraction_method == BrainExtractMethod.BRAINSUITE:
            wf.connect([
                (n4, brainsuite_wf, [('output_image', 'inputnode.in_file')]),
                (brainsuite_wf, outputnode,[('outputnode.out_file_brain_extracted', 'brainsuite_out_file_n4_corrected_brain_extracted')]),
                (brainsuite_wf, outputnode, [('outputnode.out_file_mask', 'brainsuite_out_file_mask')]),
            ])

        elif brain_extraction_method in (
                BrainExtractMethod.REGISTRATION_NO_INITIAL_MASK,
                BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK,
                BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK,
        ):

            if brain_extraction_method == BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK:
                wf.connect([
                    (n4, brainsuite_wf, [('output_image', 'inputnode.in_file')]),
                    (brainsuite_wf, ants_wf, [('outputnode.out_file_mask', 'inputnode.in_file_mask')]),
                ])
            elif brain_extraction_method == BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK:
                wf.connect[(
                    (inputnode, ants_wf, [('in_file_mask', 'inputnode.in_file_mask')])
                )]

            wf.connect([
                (n4, ants_wf, [('output_image', 'inputnode.in_file')]),
                (ants_wf, outputnode,[('outputnode.out_file_brain_extracted', 'ants_out_file_n4_corrected_brain_extracted')]),
                (ants_wf, outputnode, [('outputnode.out_file_mask', 'ants_out_file_mask')]),
            ])

        elif brain_extraction_method == BrainExtractMethod.USER_PROVIDED_MASK:
            apply_mask = pe.Node(ApplyMask(), name='apply_mask', n_procs=omp_nthreads)
            wf.connect([
                (inputnode, apply_mask, [('in_file_mask', 'mask_file')]),
                (n4, apply_mask, [('output_image', 'in_file')]),
                (apply_mask, outputnode, [('out_file', 'out_file_n4_corrected_brain_extracted')]),
                (inputnode, outputnode, [('in_file_mask', 'out_file_mask')]),
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
        '--participant_label', 'Nl311f9',
        '--session_labels', '2020021001',
        '--run_labels', '01',
        '--in_file_entities_labels_string', 'acq-TurboRARE_T2w.nii.gz',
        '--ants_template_sub_label', 'AnatTemplate',
        '--ants_template_probability_mask_sub_label', 'AnatTemplateProbabilityMask',
        '--ants_template_desc_label', '0p15x0p15x0p55mm20200804',
        '--brain_extract_method', 'REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK',
        # '--brain_extract_method', 'BRAINSUITE',
        # '--in_file', '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/atlases/AMBMC_model_downsampled.nii.gz',
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
    nipype_run_arguments.run_workflow(be_wf)
