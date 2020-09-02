from workflows.CFMMBase import CFMMWorkflow
from workflows.CFMMCommon import NipypeWorkflowArguments, NipypeRunArguments, get_node_existing_inputs_to_list
from workflows.CFMMBIDS import BIDSAppArguments, get_node_get_input_file_entities_labels_dict, \
    get_node_bids_file_multiplexer, get_node_batch_update_entities_labels_dict, get_node_update_entities_labels_dict
from workflows.CFMMBrainSuite import CFMMBse
from workflows.CFMMAnts import AntsArguments, CFMMApplyTransforms, CFMMThresholdImage, CFMMAntsRegistration, \
    CFMMN4BiasFieldCorrection
from nipype.pipeline import engine as pe
from multiprocessing import cpu_count
from nipype.interfaces.fsl import ImageMaths, CopyGeom, ApplyMask
from workflows.CFMMEnums import BrainExtractMethod
from workflows.CFMMArgumentParser import CFMMArgumentParser as ArgumentParser
from workflows.CFMMLogging import NipypeLogger as logger
import os
from workflows.CFMMConfigFile import CFMMConfig

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

def BrainSuiteBrainExtraction_add_parser_arguments(self):
    self.add_parser_argument('in_file',
                             help='Specify location of the input file for brain extraction.')
    super(type(self),self).add_parser_arguments()

class BrainSuiteBrainExtraction(CFMMWorkflow):
    group_name = 'BrainSuite Brain Extraction'
    flag_prefix = 'bs_be_'
    def __init__(self, *args, **kwargs):
        self.nipype = NipypeWorkflowArguments(exclude_list=['nthreads_mapnode', 'mem_gb_mapnode'])
        self.bse = CFMMBse()
        self.outputs = ['out_file_brain_extracted', 'out_file_mask']

        subcomponents = [self.nipype, self.bse]
        super().__init__(subcomponents, *args, **kwargs)

    add_parser_arguments = BrainSuiteBrainExtraction_add_parser_arguments

    def get_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        omp_nthreads = self.get_subcomponent(NipypeWorkflowArguments.group_name).get_parameter('nthreads_node').user_value
        if omp_nthreads is None or omp_nthreads < 1:
            omp_nthreads = cpu_count()

        bse = self.get_subcomponent(CFMMBse.group_name).get_node(name='BSE', n_procs=omp_nthreads)

        # default behaviour of brainsuite is to rotate to LPI orientation
        # this can be overridden by using the noRotate option, however this option will create a nifti with inconsistent
        # qform and sform values.  To fix this, copy the header information from the original image to the mask using fsl.
        fix_bse_orientation = pe.Node(interface=CopyGeom(), name='fixBSEOrientation', n_procs=omp_nthreads)

        # brainsuite outputs mask value as 255, change it to 1
        fix_bse_value = pe.Node(interface=ImageMaths(), name='fixBSEValue', n_procs=omp_nthreads)
        fix_bse_value.inputs.op_string = '-div 255'

        apply_mask = pe.Node(ApplyMask(), name='apply_mask', n_procs=omp_nthreads)

        #inputnode, outputnode, wf = self.get_io_and_workflow(calling_class=BrainSuiteBrainExtraction)
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




def AntsBrainExtraction_add_parser_arguments(self):
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
    super(type(self),self).add_parser_arguments()

class AntsBrainExtraction(CFMMWorkflow):
    group_name = 'ANTs Brain Extraction'
    flag_prefix = 'ants_be_'

    def __init__(self, *args, **kwargs):
        subcomponents = [NipypeWorkflowArguments(exclude_list=['nthreads_mapnode', 'mem_gb_mapnode']),
                         AntsArguments(),
                         CFMMAntsRegistration(),
                         CFMMApplyTransforms(),
                         CFMMThresholdImage(),
                         ]
        self.outputs = ['out_file_brain_extracted', 'out_file_mask']
        super().__init__(subcomponents, *args, **kwargs)

    add_parser_arguments = AntsBrainExtraction_add_parser_arguments

    def validate_parameters(self):
        template = self.get_parameter('template')
        template_probability_mask = self.get_parameter('template_probability_mask')
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

        omp_nthreads = self.get_subcomponent(NipypeWorkflowArguments.group_name).get_parameter('nthreads_node').user_value
        if omp_nthreads is None or omp_nthreads < 1:
            omp_nthreads = cpu_count()

        ants_reg = self.get_subcomponent(CFMMAntsRegistration.group_name).get_node(n_procs=omp_nthreads, name='ants_reg')
        apply_transform = self.get_subcomponent(CFMMApplyTransforms.group_name).get_node(name='antsApplyTransforms')
        thr_brainmask = self.get_subcomponent(CFMMThresholdImage.group_name).get_node(name='thr_brainmask', n_procs=omp_nthreads)
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
        if self.get_parameter('brain_extract_method').user_value in (
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

# problem:
# both subclass and superclass have the same parameter name, and connections are made to connect subclass to superclass
# in bids this is necessary, both subclass and superclass need in_file (bids needs it to determine between bids search and input)

# option 1
# have nested workflows instead of sublcassing
# give subclass flags a different name like bids_override instead of in_file
# this is kind of confusing

# option 2
# have nested workflows instead of sublcassing
# subclass has in_file and superclass has in_file. flag prefixes are used to make sure superclass is different from subclass
# parameters from subclass and superclass are in same help section, but have different flag prefixes

# option 3
# have nested workflows instead of sublcassing
# first create the superclass parameters using a helper function
# next modify the flag names of any of the parameters that correspond to superclass inputnode
# finally, use the same helper function to add the inputnode parameters to the subclass (no name conflicts)
# confusing for the user

# option 4
# the user passes the subclass name when calling get_workflow
# the baseclass memeber is a dict which stores a workflow under the sublcass key

# option 5
# should reset instance attributes when returning youur build wf.  instance attributes are just to help get your workflow,
# after it's delivered you should reset them

# option 6
# have a get_super_workflow method which stores and then resets all class members (workflow,innode,etc),
# calls the super get_workflow, then restores the class members to the previously stored values. basically save the state
# of the self instance, reset it, let the super do its thing, then restore the state


#use super with resetting inputnode and workflow
# class BrainSuiteBrainExtractionBIDS(BrainSuiteBrainExtraction):
#     def __init__(self, *args, **kwargs):
#         self.add_subcomponent(BIDSAppArguments())
#
#         super().__init__(*args, **kwargs)
#     def add_parser_arguments(self):
#         super().add_parser_arguments()
#         self.add_parser_argument('in_file_entities_labels_string',
#                                  help=f'BIDS entity-label search string for in_file. Some entities are reused if doing a bids search for the in_file mask, template, or template probability mask. The in_file search can be overridden by ')
#
#     def get_workflow(self, arg_dict=None):
#         if arg_dict is not None:
#             self.populate_parameters(arg_dict)
#             self.validate_parameters()
#         wf1 = super().get_workflow()
#         inp,outp,wf = self.get_io_and_workflow()


class BrainSuiteBrainExtractionBIDS(CFMMWorkflow):
    group_name=BrainSuiteBrainExtraction.group_name
    flag_prefix = 'bids_bs_be_'
    def __init__(self, *args, **kwargs):
        subcomponents = [
            BIDSAppArguments(),
            BrainSuiteBrainExtraction(group_name='')
        ]

        self.outputs = {
            'out_file_brain_extracted': 'BrainSuiteBrainExtracted',
            'out_file_mask': 'BrainSuiteBrainMask'
            }

        super().__init__(subcomponents, *args, **kwargs)


    def add_parser_arguments(self):
        # BrainSuiteBrainExtraction_add_parser_arguments instead of super().add_parser_arguments()
        BrainSuiteBrainExtraction_add_parser_arguments(self)
        subordinate_subcomponent = self.get_subcomponent('')
        for parameter_name,superior_parameter in self._parameters.items():
            subordinate_subcomponent.get_parameter(parameter_name).replaced_by(superior_parameter)
            self._param_subordinates[superior_parameter] = (parameter_name, subordinate_subcomponent)
        BrainSuiteBrainExtraction = self.get_subcomponent('')
        self.add_parser_argument('in_file_entities_labels_string',
                                 help=f'BIDS entity-label search string for in_file. Some entities are reused if doing a bids search for the in_file mask, template, or template probability mask. The in_file search can be overridden by --{BrainSuiteBrainExtraction.get_parameter("in_file").parser_flag}.')

        self.get_subcomponent(BIDSAppArguments.group_name).modify_parser_argument('analysis_level', 'choices', ['participant'])

    def validate_parameters(self):
        explicit_method = self.get_subcomponent('').get_parameter('in_file')
        bids_method = self.get_parameter('in_file_entities_labels_string')

        full_group_name = os.sep.join([self.get_group_name_chain(),self.group_name])
        if (not explicit_method.obtaining_value_from_superior()) \
            and \
                (explicit_method.user_value is not None and bids_method.user_value is not None):
            logger.warning(f"{full_group_name}: BIDS search for input is being overridden by --{explicit_method.parser_flag}")
        super().validate_parameters()

    def get_workflow(self, arg_dict=None):
        if self.workflow is not None:
            return self.workflow
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

        brainsuite_be_wf = self.get_subcomponent('').get_workflow()

        inputnode, outputnode, wf = self.get_io_and_workflow(connection_exclude_list=['in_file'])
        derivatives_node = self.get_node_derivatives_datasink()

        #self.connect_to_superclass_inputnode(brainsuite_be_wf, exclude_list=['in_file'])

        wf.connect([
            # deciding between in_file and bids search
            (inputnode, input_file_entities_labels_dict, [('participant_label', 'participant_label')]),
            (inputnode, input_file_entities_labels_dict, [('session_labels', 'session_labels')]),
            (inputnode, input_file_entities_labels_dict, [('run_labels', 'run_labels')]),
            (inputnode, input_file_entities_labels_dict, [('in_file_entities_labels_string', 'entities_string')]),

            (input_file_entities_labels_dict, choose_in_file, [('entities_labels_dict', 'entities_labels_dict')]),
            (inputnode, choose_in_file, [('in_file', 'input_file')]),
            (inputnode, choose_in_file, [('bids_layout_db', 'bids_layout_db')]),
            (choose_in_file, derivatives_node, [('chosen_file', 'original_bids_file')]),

            # connect chosen file brainsuite_be_wf
            (choose_in_file, brainsuite_be_wf, [('chosen_file', 'inputnode.in_file')]),
            (brainsuite_be_wf, outputnode, [('outputnode.out_file_brain_extracted', 'out_file_brain_extracted')]),
            (brainsuite_be_wf, outputnode, [('outputnode.out_file_mask', 'out_file_mask')]),
        ])
        return wf


class AntsBrainExtractionBIDS(CFMMWorkflow):
    group_name = AntsBrainExtraction.group_name
    flag_prefix = 'bids_ants_be_'

    def __init__(self, *args, **kwargs):
        subcomponents = [
            BIDSAppArguments(),
            AntsBrainExtraction(group_name='')
        ]

        self.outputs = {
            'out_file_brain_extracted': 'ANTsBrainExtracted',
            'out_file_mask': 'ANTsBrainMask'
            }

        # CFMMWorkflow.__init__(self,subcomponents, *args, **kwargs)
        super().__init__(subcomponents, *args, **kwargs)


    def add_parser_arguments(self):
        # AntsBrainExtraction_add_parser_arguments instead of super().add_parser_arguments()
        AntsBrainExtraction_add_parser_arguments(self)
        subordinate_subcomponent = self.get_subcomponent('')
        for parameter_name,superior_parameter in self._parameters.items():
            subordinate_subcomponent.get_parameter(parameter_name).replaced_by(superior_parameter)
            self._param_subordinates[superior_parameter] = (parameter_name, subordinate_subcomponent)


        AntsBrainExtraction = self.get_subcomponent('')
        self.add_parser_argument('in_file_entities_labels_string',
                                 help=f'BIDS entity-label search string for in_file. Some entities are reused if doing a bids search for the in_file mask, template, or template probability mask. The in_file search can be overridden by --{AntsBrainExtraction.get_parameter("in_file").parser_flag}.')
        self.add_parser_argument('in_file_mask_desc_label',
                                 help=f'BIDS description label used to search for in_file_mask. Overridden by --{AntsBrainExtraction.get_parameter("in_file_mask").parser_flag}.')
        self.add_parser_argument('template_sub_label',
                                 help=f'BIDS subject label used to search for the template. Overridden by --{AntsBrainExtraction.get_parameter("template").parser_flag}.')
        self.add_parser_argument('template_probability_mask_sub_label',
                                 help=f'BIDS subject label used to search for the template probability mask. Overridden by --{AntsBrainExtraction.get_parameter("template_probability_mask").parser_flag}.')
        self.add_parser_argument('template_desc_label',
                                 help=f'BIDS description label used to search for the template and probability mask. Overridden by --{AntsBrainExtraction.get_parameter("template").parser_flag} and --{AntsBrainExtraction.get_parameter("template_probability_mask").parser_flag}.')
        self.get_subcomponent(BIDSAppArguments.group_name).modify_parser_argument('analysis_level', 'choices', ['participant'])

    def validate_parameters(self):

        super().validate_parameters()

        explicit_method = self.get_subcomponent('').get_parameter('in_file')
        bids_method = self.get_parameter('in_file_entities_labels_string')
        full_group_name = os.sep.join([self.get_group_name_chain(),self.group_name])
        if (not explicit_method.obtaining_value_from_superior()) \
            and \
                (explicit_method.user_value is not None and bids_method.user_value is not None):
            logger.warning(f"{full_group_name}: BIDS search for input is being overridden by --{explicit_method.parser_flag}")
        if (not (explicit_method.obtaining_value_from_superior() or bids_method.obtaining_value_from_superior())) \
        and explicit_method.user_value is None and bids_method.user_value is None:
            logger.error(f"{full_group_name}: Either --{explicit_method.parser_flag} or --{bids_method.parser_flag} must be supplied")

        explicit_method = self.get_subcomponent('').get_parameter('template')
        bids_method = self.get_parameter('template_sub_label')
        full_group_name = os.sep.join([self.get_group_name_chain(),self.group_name])
        if (not explicit_method.obtaining_value_from_superior()) \
            and \
                (explicit_method.user_value is not None and bids_method.user_value is not None):
            logger.warning(f"{full_group_name}: BIDS search for template is being overridden by --{explicit_method.parser_flag}")
        if (not (explicit_method.obtaining_value_from_superior() or bids_method.obtaining_value_from_superior())) \
        and explicit_method.user_value is None and bids_method.user_value is None:
            logger.error(f"{full_group_name}: Either --{explicit_method.parser_flag} or --{bids_method.parser_flag} must be supplied")

        explicit_method = self.get_subcomponent('').get_parameter('template_probability_mask')
        bids_method = self.get_parameter('template_probability_mask_sub_label')
        full_group_name = os.sep.join([self.get_group_name_chain(),self.group_name])
        if (not explicit_method.obtaining_value_from_superior()) \
            and \
                (explicit_method.user_value is not None and bids_method.user_value is not None):
            logger.warning(f"{full_group_name}: BIDS search for template probability mask is being overridden by --{explicit_method.parser_flag}")
        if (not (explicit_method.obtaining_value_from_superior() or bids_method.obtaining_value_from_superior())) \
        and explicit_method.user_value is None and bids_method.user_value is None:
            logger.error(f"{full_group_name}: Either --{explicit_method.parser_flag} or --{bids_method.parser_flag} must be supplied")

        brain_extract_method = self.get_subcomponent('').get_parameter('brain_extract_method')
        if brain_extract_method.user_value == BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK:
            explicit_method = self.get_subcomponent('').get_parameter('in_file_mask')
            bids_method = self.get_parameter('in_file_mask_desc_label')
            full_group_name = os.sep.join([self.get_group_name_chain(), self.group_name])
            if (not explicit_method.obtaining_value_from_superior()) \
                    and \
                    (explicit_method.user_value is not None and bids_method.user_value is not None):
                logger.warning(
                    f"{full_group_name}: BIDS search for input mask is being overridden by --{explicit_method.parser_flag}")
                if (
                not (explicit_method.obtaining_value_from_superior() or bids_method.obtaining_value_from_superior())) \
                        and explicit_method.user_value is None and bids_method.user_value is None:
                    logger.error(
                        f"{full_group_name}: Either --{explicit_method.parser_flag} or --{bids_method.parser_flag} must be supplied when using ")


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

        template_probability_mask_labels_list = get_node_existing_inputs_to_list(
            'template_probability_mask_labels_list')
        template_probability_mask_entities_dict = get_node_update_entities_labels_dict(
            'template_probability_mask_entities_dict')
        template_probability_mask_entities_dict.inputs.entity = ['subject', 'desc']
        choose_template_probability_mask = get_node_bids_file_multiplexer('choose_template_probability_mask')

        ants_be_wf = self.get_subcomponent('').get_workflow()

        inputnode, outputnode, wf = self.get_io_and_workflow(connection_exclude_list=['in_file',
                                               'in_file_mask',
                                               'template',
                                               'template_probability_mask'
                                               ])
        derivatives_node = self.get_node_derivatives_datasink()

        # self.connect_to_superclass_inputnode(ants_be_wf, exclude_list=['in_file',
        #                                        'in_file_mask',
        #                                        'template',
        #                                        'template_probability_mask'
        #                                        ])

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
            (choose_in_file, derivatives_node, [('chosen_file', 'original_bids_file')]),

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


class BrainExtractionBIDS(CFMMWorkflow):
    group_name = 'Brain Extraction'
    #flag_prefix = 'be_'

    def __init__(self, *args, **kwargs):
        subcomponents = [BIDSAppArguments(),
                         NipypeWorkflowArguments(exclude_list=['nthreads_mapnode', 'mem_gb_mapnode']),
                         CFMMN4BiasFieldCorrection(),
                         BrainSuiteBrainExtractionBIDS(),
                         AntsBrainExtractionBIDS(),
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
        self.get_subcomponent(BIDSAppArguments.group_name).modify_parser_argument('analysis_level', 'choices', ['participant'])
        self.add_parser_argument('in_file',
                                 help='Explicitly specify location of the input file for brain extraction.',
                                 override_parameters=[
                                     ('in_file', BrainSuiteBrainExtractionBIDS.group_name),
                                     ('in_file', AntsBrainExtractionBIDS.group_name)
                                 ],
                                 )
        self.add_parser_argument('in_file_entities_labels_string',
                                 help=f'BIDS entity-label search string for in_file. Some entities are reused if doing a bids search for the in_file mask, template, or template probability mask. The in_file search can be overridden by --{self.get_parameter("in_file").parser_flag}.',
                                 override_parameters=[
                                     ('in_file_entities_labels_string', BrainSuiteBrainExtractionBIDS.group_name),
                                     ('in_file_entities_labels_string', AntsBrainExtractionBIDS.group_name)
                                 ],
                                 )
        self.add_parser_argument('brain_extract_method',
                                 choices=list(BrainExtractMethod),
                                 default=BrainExtractMethod.NO_BRAIN_EXTRACTION.name,
                                 type=BrainExtractMethod.argparse_convert,
                                 help="Brain extraction method for image.",
                                 add_to_inputnode=False,
                                 override_parameters=[
                                     ('brain_extract_method', [AntsBrainExtractionBIDS.group_name,''])
                                 ],
                                 )
        parent_nipype = self.get_subcomponent(NipypeWorkflowArguments.group_name)
        child_nipype = self.get_subcomponent([AntsBrainExtractionBIDS.group_name, '', NipypeWorkflowArguments.group_name])
        self.replace_defaults(parent_nipype, child_nipype)

    def validate_parameters(self):
        super().validate_parameters()

        explicit_method = self.get_parameter('in_file')
        bids_method = self.get_parameter('in_file_entities_labels_string')
        full_group_name = os.sep.join([self.get_group_name_chain(),self.group_name])
        if (not explicit_method.obtaining_value_from_superior()) \
            and \
                (explicit_method.user_value is not None and bids_method.user_value is not None):
            logger.warning(f"{full_group_name}: BIDS search for input is being overridden by --{explicit_method.parser_flag}")

        brain_extraction_parameter = self.get_parameter('brain_extract_method')

        brain_extraction_method = brain_extraction_parameter.user_value
        if brain_extraction_method in (
                BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK, BrainExtractMethod.REGISTRATION_NO_INITIAL_MASK,
                BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK):

            ants_be_base_obj = self.get_subcomponent([AntsBrainExtractionBIDS.group_name,''])
            ants_be_bids_obj = self.get_subcomponent([AntsBrainExtractionBIDS.group_name])
            template = ants_be_base_obj.get_parameter('template')
            template_probability_mask = ants_be_base_obj.get_parameter('template_probability_mask')
            template_bids_entities = ants_be_bids_obj.get_parameter('template_sub_label')

            if template.user_value is None and template_probability_mask.user_value is None and template_bids_entities.user_value is None:
                self.parser.error(
                    f'When using {brain_extraction_parameter.parser_flag}={brain_extraction_parameter.user_value}, \n'
                    f'either {template_bids_entities.parser_flag} or {template.parser_flag} and {template_probability_mask.parser_flag} must be defined.')

    def get_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        omp_nthreads = self.get_subcomponent(NipypeWorkflowArguments.group_name).get_parameter('nthreads_node').user_value
        if omp_nthreads is None or omp_nthreads < 1:
            omp_nthreads = cpu_count()

        input_file_entities_labels_dict = get_node_get_input_file_entities_labels_dict(
            'input_file_entities_labels_dict')
        # what if choose_in_file uses bids and returns multiple files???
        choose_in_file = get_node_bids_file_multiplexer('choose_in_file')

        n4 = self.get_subcomponent(CFMMN4BiasFieldCorrection.group_name).get_node(n_procs=omp_nthreads, name='n4')

        brainsuite_obj = self.get_subcomponent(BrainSuiteBrainExtractionBIDS.group_name)
        brainsuite_wf = brainsuite_obj.get_workflow()

        ants_obj = self.get_subcomponent(AntsBrainExtractionBIDS.group_name)
        ants_wf = ants_obj.get_workflow()

        inputnode, outputnode, wf = self.get_io_and_workflow(connection_exclude_list=['in_file'])
        derivatives_node = self.get_node_derivatives_datasink()


        wf.connect([
            (inputnode, input_file_entities_labels_dict, [('participant_label', 'participant_label')]),
            (inputnode, input_file_entities_labels_dict, [('session_labels', 'session_labels')]),
            (inputnode, input_file_entities_labels_dict, [('run_labels', 'run_labels')]),
            (inputnode, input_file_entities_labels_dict, [('in_file_entities_labels_string', 'entities_string')]),
            (input_file_entities_labels_dict, choose_in_file, [('entities_labels_dict', 'entities_labels_dict')]),
            (inputnode, choose_in_file, [('in_file', 'input_file')]),
            (inputnode, choose_in_file, [('bids_layout_db', 'bids_layout_db')]),
            (choose_in_file, derivatives_node, [('chosen_file', 'original_bids_file')]),
            (choose_in_file, n4, [('chosen_file', 'input_image')]),
            (n4, outputnode, [('output_image', 'out_file_n4_corrected')]),
        ])

        brain_extraction_method = self.get_parameter('brain_extract_method').user_value

        # these conditionals do not need to be inside a node because they depend on command line values and not on
        # node values - ie. they can't be overridden the same way a node's inputs attributes can be
        # ref. choose_template() in MouseAntsBrainExtractionBIDSAbstractWorkflow.get_workflow()
        if brain_extraction_method == BrainExtractMethod.BRAINSUITE:
            wf.connect([
                (n4, brainsuite_wf, [('output_image', 'inputnode.in_file')]),
                (brainsuite_wf, outputnode,
                 [('outputnode.out_file_brain_extracted', 'brainsuite_out_file_n4_corrected_brain_extracted')]),
                (brainsuite_wf, outputnode, [('outputnode.out_file_mask', 'brainsuite_out_file_mask')]),
            ])

        elif brain_extraction_method in (
                BrainExtractMethod.REGISTRATION_NO_INITIAL_MASK,
                BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK,
                BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK,
        ):

            if brain_extraction_method == BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK:
                wf.connect(n4, 'output_image', brainsuite_wf, 'inputnode.in_file')
                wf.connect(brainsuite_wf,'outputnode.out_file_mask',ants_wf,'inputnode.in_file_mask')
            elif brain_extraction_method == BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK:
                wf.connect(inputnode, 'in_file_mask', ants_wf, 'inputnode.in_file_mask')

            wf.connect(n4, 'output_image', ants_wf, 'inputnode.in_file')

            wf.connect([
                (ants_wf, outputnode,
                 [('outputnode.out_file_brain_extracted', 'ants_out_file_n4_corrected_brain_extracted')]),
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








def modify_bse_for_mouse(bse_component):
    bse_component.modify_parser_argument('diffusionConstant', 'default', 30.0)
    bse_component.modify_parser_argument('diffusionIterations', 'default', 3)
    bse_component.modify_parser_argument('edgeDetectionConstant', 'default', 0.55)
    bse_component.modify_parser_argument('radius', 'default', 2)
    bse_component.modify_parser_argument('dilateFinalMask', 'default', True)
    bse_component.modify_parser_argument('trim', 'default', False)
    bse_component.modify_parser_argument('noRotate', 'default', True)

def modify_antsreg_for_mouse(antsreg_component):
    # note: the type conversion function you provided to argparse is only called on string defaults
    # therefore a default of 3 will set the argument to 3 (both integers)
    # a default of '3' will go through the convert function and in our case convert_argparse_using_eval.convert()'s
    # eval() function will convert the string to integer 3
    # it is important to to include two sets of quotes if the default value is supposed to be a string
    # so that after the eval function, it will still be a string
    antsreg_component.modify_parser_argument('output_transform_prefix', 'default', "'output_'")
    antsreg_component.modify_parser_argument('dimension', 'default', 3)
    antsreg_component.modify_parser_argument('initial_moving_transform_com', 'default', "1")
    antsreg_component.modify_parser_argument('transforms', 'default', "['Affine', 'SyN']")
    # transform_parameters:
    # gradient step
    # updateFieldVarianceInVoxelSpace - smooth the deformation computed on the "updated" gradient field before this is added to previous deformations to form the "total" gradient field
    # totalFieldVarianceInVoxelSpace - smooth the deformation computed on the "total" gradient field
    antsreg_component.modify_parser_argument('transform_parameters', 'default', "[(0.1,), (0.1, 3.0, 0.0)]")
    antsreg_component.modify_parser_argument('number_of_iterations', 'default', "[[10, 5, 3], [10, 5, 3]]")
    # transform for each stage vs composite for entire warp
    antsreg_component.modify_parser_argument('write_composite_transform', 'default', "True")
    # combines adjacent transforms when possible
    antsreg_component.modify_parser_argument('collapse_output_transforms', 'default', "False")
    # ants_reg.inputs.initialize_transforms_per_stage = False #seems to be for initializing linear transforms only
    # using CC when atlas was made using same protocol
    antsreg_component.modify_parser_argument('metric', 'default', "['CC'] * 2")
    # weight used if you do multimodal registration. Default is 1 (value ignored currently by ANTs)
    antsreg_component.modify_parser_argument('metric_weight', 'default', "[1] * 2")
    # radius for CC between 2-5
    antsreg_component.modify_parser_argument('radius_or_number_of_bins', 'default', "[5] * 2")
    # not entirely sure why we don't need to specify sampling strategy and percentage for non-linear syn registration
    # but I'm just following ANTs examples
    antsreg_component.modify_parser_argument('sampling_strategy', 'default', "['Regular', None]")
    antsreg_component.modify_parser_argument('sampling_percentage', 'default', "[0.5, None]")
    # use a negative number if you want to do all iterations and never exit
    antsreg_component.modify_parser_argument('convergence_threshold', 'default', "[1.e-8,1.e-9]")
    # if the cost hasn't changed by convergence threshold in the last window size iterations, exit loop
    antsreg_component.modify_parser_argument('convergence_window_size', 'default', "[10] * 2")
    antsreg_component.modify_parser_argument('smoothing_sigmas', 'default', "[[0.3, 0.15, 0], [0.3, 0.15, 0]]")
    # we use mm instead of vox because we don't have isotropic voxels
    antsreg_component.modify_parser_argument('sigma_units', 'default', "['mm'] * 2  ")
    antsreg_component.modify_parser_argument('shrink_factors', 'default', "[[3, 2, 1], [3, 2, 1]]")
    # estimate the learning rate step size only at the beginning of each level. Does this override the value chosen in transform_parameters?
    antsreg_component.modify_parser_argument('use_estimate_learning_rate_once', 'default', "[True,True]")
    antsreg_component.modify_parser_argument('use_histogram_matching', 'default', "[True, True]")
    antsreg_component.modify_parser_argument('output_warped_image', 'default', "'output_warped_image.nii.gz'")

  
def modify_n4_for_mouse(n4_component):
    n4_component.modify_parser_argument('bspline_fitting_distance', 'default', 20.0)
    n4_component.modify_parser_argument('dimension', 'default', 3)
    n4_component.modify_parser_argument('save_bias', 'default', False)
    n4_component.modify_parser_argument('copy_header', 'default', True)
    #n4_component.modify_parser_argument('n_iterations', 'default', [50] * 4)
    n4_component.modify_parser_argument('n_iterations', 'default', str([50] * 4))
    n4_component.modify_parser_argument('convergence_threshold', 'default', 1e-7)
    n4_component.modify_parser_argument('shrink_factor', 'default', 4)
    
class MouseBrainSuiteBrainExtraction(BrainSuiteBrainExtraction):
    def add_parser_arguments(self):
        super().add_parser_arguments()
        bse_component = self.get_subcomponent(CFMMBse.group_name)
        modify_bse_for_mouse(bse_component)

class MouseAntsBrainExtraction(AntsBrainExtraction):
    def add_parser_arguments(self):
        super().add_parser_arguments()
        antsreg_component = self.get_subcomponent(CFMMAntsRegistration.group_name)
        modify_antsreg_for_mouse(antsreg_component)

class MouseBrainExtractionBIDS(BrainExtractionBIDS):
    def add_parser_arguments(self):
        super().add_parser_arguments()
        bse_component = self.get_subcomponent([BrainSuiteBrainExtractionBIDS.group_name,'',CFMMBse.group_name])
        antsreg_component = self.get_subcomponent([AntsBrainExtractionBIDS.group_name,'',CFMMAntsRegistration.group_name])        
        n4_component = self.get_subcomponent(CFMMN4BiasFieldCorrection.group_name)
        modify_bse_for_mouse(bse_component)
        modify_antsreg_for_mouse(antsreg_component)
        modify_n4_for_mouse(n4_component)


if __name__ == '__main__':
    # command line arguments

    cmd_args = [
        # bidsapp
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives',
        'participant',
        '--input_derivatives_dirs',
        "['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives']",
        '--bids_layout_db', './brain_extract_test/bids_database',
        '--participant_label', 'Nl311f9',
        '--session_labels', '2020021001',
        '--run_labels', '01',
        '--in_file_entities_labels_string', 'acq-TurboRARE_T2w.nii.gz',
        #'--in_file','fake',
        '--bids_ants_be_template_sub_label', 'AnatTemplate',
        '--bids_ants_be_template_probability_mask_sub_label', 'AnatTemplateProbabilityMask',
        '--bids_ants_be_template_desc_label', '0p15x0p15x0p55mm20200804',
        '--brain_extract_method', 'REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK',
        # '--brain_extract_method', 'BRAINSUITE',
        # '--in_file', '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/atlases/AMBMC_model_downsampled.nii.gz',
        '--nipype_processing_dir', './brain_extract_test',
        '--keep_unnecessary_outputs',
    ]

    parser = ArgumentParser(description=__doc__)

    config_file_obj = CFMMConfig()
    nipype_run_arguments = NipypeRunArguments()
    be_obj = MouseBrainExtractionBIDS()

    config_file_obj.set_parser(parser)
    nipype_run_arguments.set_parser(parser)
    be_obj.set_parser(parser)

    config_file_obj.add_parser_arguments()
    nipype_run_arguments.add_parser_arguments()
    be_obj.add_parser_arguments()

    parser.print_help()
    args = config_file_obj.parse_args(cmd_args)

    args_dict = vars(args)


    nipype_run_arguments.populate_parameters(arg_dict=args_dict)
    be_wf = be_obj.get_workflow(arg_dict=args_dict)



    logger.info('Starting Program!')

    nipype_run_arguments.run_workflow(be_wf)

    logger.info('Finished Program!')
