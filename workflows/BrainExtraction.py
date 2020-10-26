from workflows.CFMMAnts import AntsDefaultArguments, CFMMApplyTransforms, CFMMThresholdImage, CFMMAntsRegistration, \
    CFMMN4BiasFieldCorrection
from workflows.CFMMWorkflow import CFMMWorkflow
from workflows.CFMMBIDS import CFMMBIDSWorkflowMixer
from workflows.CFMMBrainSuite import CFMMBse
from workflows.CFMMBIDS import BIDSAppArguments
from workflows.CFMMEnums import BrainExtractMethod
from workflows.CFMMCommon import get_node_existing_inputs_to_list
from nipype.pipeline import engine as pe
from nipype.interfaces.fsl import ImageMaths, CopyGeom, ApplyMask, Reorient2Std
from workflows.CFMMCommon import NipypeWorkflowArguments, get_node_delistify
from workflows.CFMMLogging import NipypeLogger as logger
from workflows.CFMMInterface import CFMMInterface
from nipype.interfaces.fsl import ExtractROI
from nipype.interfaces.fsl.maths import MeanImage
from workflows.CFMMMapNode import CFMMMapNode
from nipype.interfaces.utility import Function


# Iteration of brain extraction classes was accomplished using mapnodes.
# However if the brain extraction workflows are used as subworkflows in a parent which does not use mapnodes,
# the brain extraction workflow will output singleton lists which cause problems for the parent worfklow when connected
# to normal nodes (since normal nodes want a single item, not a list of a single item). Thus we delist singleton lists.


class BrainSuiteBrainExtraction(CFMMWorkflow):
    group_name = 'BrainSuite Brain Extraction'
    flag_prefix = 'bs_be_'

    def _add_parameters(self):
        self._add_parameter('in_file',
                            help='Explicitly specify location of the input file for brain extraction.')

    def __init__(self, *args, **kwargs):
        # super() deals with the ordering of setting class groupname, dealing with upstream parameter replacements
        # and executing local _add_parser_arguments()
        super().__init__(*args, **kwargs)

        # when instatiating subcomponents, indicate which parameters are going to be replaced
        self.nipype = NipypeWorkflowArguments(owner=self, exclude_list=['nthreads_mapnode', 'mem_gb_mapnode'])
        self.bse = CFMMBse(owner=self)
        self.outputs = ['out_file_brain_extracted', 'out_file_mask']

    def validate_parameters(self):
        # validations if standalone only
        if self.get_toplevel_owner() == self and self.__class__.__name__ == 'BrainSuiteBrainExtraction':
            in_file_argument = self.get_parameter('in_file')
            if in_file_argument.user_value is None:
                logger.error(f"--{in_file_argument.flagname} must be defined. ")
        super().validate_parameters()

    def create_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        omp_nthreads = self.nipype.get_parameter('nthreads_node').user_value

        # bse segfaults on some orientations - however it doesn't happen if the image is in std orientation
        reorient2std = CFMMMapNode(interface=Reorient2Std(), name='reorient2std', iterfield=['in_file'])

        bse = self.get_subcomponent(CFMMBse.group_name).get_node(name='BSE', mapnode=True, iterfield=['inputMRIFile'])

        # default behaviour of brainsuite is to rotate to LPI orientation
        # this can be overridden by using the noRotate option, however this option will create a nifti with inconsistent
        # qform and sform values.  To fix this, copy the header information from the original image to the mask using fsl.
        fix_bse_orientation = CFMMMapNode(interface=CopyGeom(), name='fixBSEOrientation',
                                          iterfield=['in_file', 'dest_file'])

        # brainsuite outputs mask value as 255, change it to 1
        fix_bse_value = CFMMMapNode(interface=ImageMaths(), name='fixBSEValue', iterfield=['in_file'])
        fix_bse_value.inputs.op_string = '-div 255'

        apply_mask = CFMMMapNode(ApplyMask(), name='apply_mask', iterfield=['in_file', 'mask_file'])
        inputnode, outputnode, wf = self.get_io_and_workflow()
        delist_out_file_brain_extracted = get_node_delistify(name='delist_out_file_brain_extracted')
        delist_out_file_mask = get_node_delistify(name='delist_out_file_mask')

        wf.connect([
            # (inputnode, bse, [('in_file', 'inputMRIFile')]),
            # (inputnode, fix_bse_orientation, [('in_file', 'in_file')]),
            # (inputnode, apply_mask, [('in_file', 'in_file')]),
            (inputnode, reorient2std, [('in_file', 'in_file')]),
            (reorient2std, bse, [('out_file', 'inputMRIFile')]),
            (reorient2std, fix_bse_orientation, [('out_file', 'in_file')]),
            (reorient2std, apply_mask, [('out_file', 'in_file')]),

            (bse, fix_bse_orientation, [('outputMaskFile', 'dest_file')]),
            (fix_bse_orientation, fix_bse_value, [('out_file', 'in_file')]),

            (fix_bse_value, apply_mask, [('out_file', 'mask_file')]),

            (apply_mask, delist_out_file_brain_extracted, [('out_file', 'input_list')]),
            (delist_out_file_brain_extracted, outputnode, [('output', 'out_file_brain_extracted')]),
            (fix_bse_value, delist_out_file_mask, [('out_file', 'input_list')]),
            (delist_out_file_mask, outputnode, [('output', 'out_file_mask')]),
        ])
        return wf


class AntsBrainExtraction(CFMMWorkflow):
    group_name = 'ANTs Brain Extraction'
    flag_prefix = 'ants_be_'

    def _add_parameters(self):
        self._add_parameter('in_file',
                            help='Explicitly specify location of the input file for brain extraction.',
                            )

        self._add_parameter('in_file_mask',
                            help='Explicitly specify location of an initial input file mask used in registration-based brain extraction.',
                            )

        self._add_parameter('template',
                            help='Explicitly specify location of the template used in registration based brain extraction.')

        self._add_parameter('template_probability_mask',
                            help='Explicitly specify location of the probability mask used in registration based brain extraction.')

        self._add_parameter('brain_extract_method',
                            choices=list(BrainExtractMethod),
                            default=BrainExtractMethod.NO_BRAIN_EXTRACTION.name,
                            type=BrainExtractMethod.argparse_convert,
                            help="Brain extraction method for image.",
                            add_to_inputnode=False, )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nipype = NipypeWorkflowArguments(owner=self, exclude_list=['nthreads_mapnode', 'mem_gb_mapnode'])
        self.ants_args = AntsDefaultArguments(owner=self)
        self.ants_reg = CFMMAntsRegistration(owner=self)
        self.apply_transform = CFMMApplyTransforms(owner=self)
        self.thresh = CFMMThresholdImage(owner=self)

        self.ants_reg.get_parameter('float').default_provider = self.ants_args.get_parameter('float')
        self.ants_reg.get_parameter('interpolation').default_provider = self.ants_args.get_parameter('interpolation')
        self.ants_reg.get_parameter('num_threads').default_provider = self.nipype.get_parameter('nthreads_node')

        self.outputs = ['out_file_brain_extracted', 'out_file_mask']

    def validate_parameters(self):
        template = self.get_parameter('template')
        template_probability_mask = self.get_parameter('template_probability_mask')
        # validations if standalone only #these should be baseclass functions
        if self.get_toplevel_owner() == self and self.__class__.__name__ == 'AntsBrainExtraction':
            in_file_argument = self.get_parameter('in_file')
            if in_file_argument.user_value is None:
                logger.error(f"--{in_file_argument.flagname} must be defined. ")
        # validations irrespective if standalone or subworkflow
        if ((template.user_value is not None) or (template_probability_mask.user_value is not None)) \
                and \
                ((template.user_value is None) or (template_probability_mask.user_value is None)):
            self.parser.error(
                f"{template.flagname} and {template_probability_mask.flagname} must be defined together")

    def create_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        omp_nthreads = self.nipype.get_parameter('nthreads_node').user_value

        ants_reg = self.ants_reg.get_node('ants_reg', mapnode=True, iterfield=['fixed_image'])
        apply_transform = self.apply_transform.get_node('apply_transform', mapnode=True,
                                                        iterfield=['transforms', 'reference_image'])
        thr_brainmask = self.thresh.get_node('thr_brainmask', mapnode=True, iterfield=['input_image'])
        apply_mask = CFMMMapNode(ApplyMask(), name='apply_mask', iterfield=['in_file', 'mask_file'],
                                 n_procs=omp_nthreads)

        inputnode, outputnode, wf = self.get_io_and_workflow()
        delist_out_file_brain_extracted = get_node_delistify(name='delist_out_file_brain_extracted')
        delist_out_file_mask = get_node_delistify(name='delist_out_file_mask')

        wf.connect([
            (inputnode, ants_reg, [('in_file', 'fixed_image')]),
            (inputnode, ants_reg, [('template', 'moving_image')]),

            (inputnode, apply_transform, [('template_probability_mask', 'input_image')]),
            (ants_reg, apply_transform, [('composite_transform', 'transforms')]),
            (inputnode, apply_transform, [('in_file', 'reference_image')]),

            (apply_transform, thr_brainmask, [('output_image', 'input_image')]),

            (thr_brainmask, apply_mask, [('output_image', 'mask_file')]),
            (inputnode, apply_mask, [('in_file', 'in_file')]),

            (apply_mask, delist_out_file_brain_extracted, [('out_file', 'input_list')]),
            (delist_out_file_brain_extracted, outputnode, [('output', 'out_file_brain_extracted')]),
            (thr_brainmask, delist_out_file_mask, [('output_image', 'input_list')]),
            (delist_out_file_mask, outputnode, [('output', 'out_file_mask')]),

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
            ants_reg.iterfield.append('fixed_image_mask')
            wf.connect([
                (inputnode, ants_reg, [('in_file_mask', 'fixed_image_mask')]),
                (inputnode, ants_reg, [('template_probability_mask', 'moving_image_mask')]),
            ])

        return wf


class BrainExtraction(CFMMWorkflow):
    group_name = 'Brain Extraction'
    flag_prefix = 'be_'

    def _add_parameters(self):
        self._add_parameter('in_file',
                            help='Explicitly specify location of the input file for brain extraction.',
                            )

        self._add_parameter('in_file_mask',
                            help='Explicitly specify location of an initial input file mask used in registration-based brain extraction.',
                            )
        self._add_parameter('brain_extract_method',
                            choices=list(BrainExtractMethod),
                            default=BrainExtractMethod.NO_BRAIN_EXTRACTION.name,
                            type=BrainExtractMethod.argparse_convert,
                            help="Brain extraction method for image.",
                            add_to_inputnode=False,
                            )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nipype = NipypeWorkflowArguments(owner=self, exclude_list=['nthreads_mapnode', 'mem_gb_mapnode'])
        self.n4 = CFMMN4BiasFieldCorrection(owner=self)

        #what if in_file is disabled?
        self.bse = BrainSuiteBrainExtraction(
            owner=self,
            exclude_list=['in_file']
        )

        self.ants = AntsBrainExtraction(
            owner=self,
            exclude_list=['in_file','in_file_mask'],
            replaced_parameters={
                'brain_extract_method': self.get_parameter('brain_extract_method'),
            })

        self.outputs = ['out_file_n4_corrected',
                        'out_file_n4_corrected_brain_extracted',
                        'out_file_mask',
                        ]

    def create_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        omp_nthreads = self.nipype.get_parameter('nthreads_node').user_value

        n4 = self.n4.get_node(n_procs=omp_nthreads, name='n4', mapnode=True, iterfield=['input_image'])
        brainsuite_wf = self.bse.create_workflow()
        ants_wf = self.ants.create_workflow()

        inputnode, outputnode, wf = self.get_io_and_workflow()
        delist_out_file_n4_corrected = get_node_delistify(name='delist_out_file_n4_corrected')

        wf.connect([
            (inputnode, n4, [('in_file', 'input_image')]),
            (n4, delist_out_file_n4_corrected, [('output_image', 'input_list')]),
            (delist_out_file_n4_corrected, outputnode, [('output', 'out_file_n4_corrected')]),
        ])

        brain_extraction_method = self.get_parameter('brain_extract_method').user_value

        if brain_extraction_method == BrainExtractMethod.BRAINSUITE:
            wf.connect([
                (n4, brainsuite_wf, [('output_image', 'inputnode.in_file')]),
                (brainsuite_wf, outputnode, [
                    ('outputnode.out_file_brain_extracted', 'out_file_n4_corrected_brain_extracted'),
                    ('outputnode.out_file_mask', 'out_file_mask')]),
            ])

        elif brain_extraction_method in (
                BrainExtractMethod.REGISTRATION_NO_INITIAL_MASK,
                BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK,
                BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK,
        ):

            if brain_extraction_method == BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK:
                wf.connect(n4, 'output_image', brainsuite_wf, 'inputnode.in_file')
                wf.connect(brainsuite_wf, 'outputnode.out_file_mask', ants_wf, 'inputnode.in_file_mask')
            elif brain_extraction_method == BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK:
                wf.connect(inputnode, 'in_file_mask', ants_wf, 'inputnode.in_file_mask')

            wf.connect([
                (n4, ants_wf, [('output_image', 'inputnode.in_file')]),
            ])

            wf.connect([
                (ants_wf, outputnode, [
                    ('outputnode.out_file_brain_extracted', 'out_file_n4_corrected_brain_extracted'),
                    ('outputnode.out_file_mask', 'out_file_mask'),
                ]),
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


class CFMMExtractROI(CFMMInterface):
    group_name = 'ExtractROI'
    flag_prefix = 'roi_'

    def __init__(self, *args, **kwargs):
        super().__init__(ExtractROI, *args, **kwargs)


class CFMMVolumesToAvg(CFMMExtractROI):
    group_name = 'Volumes to Average'

    def _add_parameters(self):
        super()._add_parameters()
        self._modify_parameter('t_min', 'default', '0')
        self._modify_parameter('t_size', 'default', '10')

    def __init__(self, *args, **kwargs):
        exclude_list = ['crop_list', 'in_file', 'roi_file', 'x_min', 'x_size', 'y_min', 'y_size', 'z_min', 'z_size']
        if hasattr(kwargs, 'exclude_list'):
            kwargs[exclude_list] += exclude_list
        else:
            kwargs['exclude_list'] = exclude_list

        super().__init__(*args, **kwargs)


class BrainExtraction4D(BrainExtraction):
    group_name = 'Brain Extraction 4D'
    flag_prefix = 'be4d_'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.roi = CFMMVolumesToAvg(owner=self)

    def create_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()
        wf = super().create_workflow()

        # inject the time average node right after inputnode.in_file
        srcnode = wf.get_node('inputnode')
        srcnode_output_name = 'in_file'
        new_srcnode = self.roi.get_node(name='roi', mapnode=True, iterfield=['in_file'])
        new_srcnode_output_name = 'roi_file'
        wf.connect(srcnode, srcnode_output_name, new_srcnode, 'in_file')
        self.replace_srcnode_connections(srcnode, srcnode_output_name, new_srcnode, new_srcnode_output_name)

        return wf


class BrainSuiteBrainExtractionBIDS(BrainSuiteBrainExtraction, CFMMBIDSWorkflowMixer):
    def __init__(self, *args, save_derivatives=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.bids = BIDSAppArguments(owner=self)
        self.bids._modify_parameter('analysis_level', 'choices', ['participant'])

        if save_derivatives:
            in_file_derivatives = {
                'out_file_brain_extracted': 'BrainSuiteBrainExtracted',
                'out_file_mask': 'BrainSuiteBrainMask'
            }
        else:
            in_file_derivatives = None

        self.create_bids_input('in_file', output_derivatives=in_file_derivatives)

    def validate_parameters(self):
        super().validate_parameters()

        explicit_method = self.get_parameter('in_file')
        bids_method = self.get_parameter('in_file_entities_labels_string')
        full_group_name = self.join_nested_groupnames(self.get_nested_groupnames())
        # validations if standalone only
        if self.get_toplevel_owner() == self:
            if explicit_method.user_value is None and bids_method.user_value is None:
                logger.error(
                    f"{full_group_name}: Either --{explicit_method.flagname} or --{bids_method.flagname} must be supplied")
        # validations irrespective if standalone or subworkflow
        if explicit_method.user_value is not None and bids_method.user_value is not None:
            logger.warning(
                f"{full_group_name}: BIDS search for input is being overridden by --{explicit_method.flagname} argument.")

    def create_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        wf = super().create_workflow()
        mapnode = True
        inputnode = self.inputnode
        outputnode = self.outputnode
        bids_parameter_group = self.bids

        self.add_bids_to_workflow(wf, inputnode, outputnode, bids_parameter_group, mapnode)
        return wf


class AntsBrainExtractionBIDS(AntsBrainExtraction, CFMMBIDSWorkflowMixer):
    def __init__(self, *args, save_derivatives=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.bids = BIDSAppArguments(owner=self)
        self.bids._modify_parameter('analysis_level', 'choices', ['participant'])

        if save_derivatives:
            in_file_derivatives = {
                'out_file_brain_extracted': 'ANTsBrainExtracted',
                'out_file_mask': 'ANTsBrainMask'
            }
        else:
            in_file_derivatives = None

        self.create_bids_input('in_file',
                               output_derivatives=in_file_derivatives)
        self.create_bids_input('in_file_mask',
                               existing_base_entities_string='in_file_entities_labels_string',
                               entities_to_remove=['extension'],
                               entities_to_add=['desc'])
        self.create_bids_input('template',
                               existing_base_entities_string='in_file_entities_labels_string',
                               entities_to_remove=['extension', 'session', 'run'],
                               entities_to_add=['subject', 'desc'])
        self.create_bids_input('template_probability_mask',
                               existing_base_entities_string='in_file_entities_labels_string',
                               entities_to_remove=['extension', 'session', 'run'],
                               entities_to_add=['subject', 'desc'])

    def validate_parameters(self):
        super().validate_parameters()
        full_group_name = self.join_nested_groupnames(self.get_nested_groupnames())

        # validations if standalone only
        if self.get_toplevel_owner() == self:
            explicit_method = self.get_parameter('in_file')
            bids_method = self.get_parameter('in_file_entities_labels_string')
            if explicit_method.user_value is None and bids_method.user_value is None:
                logger.error(
                    f"{full_group_name}: Either --{explicit_method.flagname} or --{bids_method.flagname} must be supplied")

            explicit_method = self.get_parameter('template')
            bids_method = self.get_parameter('template_subject')
            if explicit_method.user_value is None and bids_method.user_value is None:
                logger.error(
                    f"{full_group_name}: Either --{explicit_method.flagname} or --{bids_method.flagname} must be supplied")

            explicit_method = self.get_parameter('template_probability_mask')
            bids_method = self.get_parameter('template_probability_mask_subject')
            if explicit_method.user_value is None and bids_method.user_value is None:
                logger.error(
                    f"{full_group_name}: Either --{explicit_method.flagname} or --{bids_method.flagname} must be supplied")

            brain_extract_method = self.get_parameter('brain_extract_method')
            if brain_extract_method.user_value == BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK:
                explicit_method = self.get_parameter('in_file_mask')
                bids_method = self.get_parameter('in_file_mask_desc')
                if explicit_method.user_value is None and bids_method.user_value is None:
                    logger.error(
                        f"{full_group_name}: Either --{explicit_method.flagname} or --{bids_method.flagname} must be supplied")

        # validations irrespective if standalone or subworkflow
        explicit_method = self.get_parameter('in_file')
        bids_method = self.get_parameter('in_file_entities_labels_string')
        if explicit_method.user_value is not None and bids_method.user_value is not None:
            logger.warning(
                f"{full_group_name}: BIDS search for input is being overridden by --{explicit_method.flagname} argument.")

        explicit_method = self.get_parameter('template')
        bids_method = self.get_parameter('template_subject')
        if (explicit_method.user_value is not None and bids_method.user_value is not None):
            logger.warning(
                f"{full_group_name}: BIDS search for template is being overridden by --{explicit_method.flagname}")

        explicit_method = self.get_parameter('template_probability_mask')
        bids_method = self.get_parameter('template_probability_mask_subject')
        if (explicit_method.user_value is not None and bids_method.user_value is not None):
            logger.warning(
                f"{full_group_name}: BIDS search for template probability mask is being overridden by --{explicit_method.flagname}")

        brain_extract_method = self.get_parameter('brain_extract_method')
        if brain_extract_method.user_value == BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK:
            explicit_method = self.get_parameter('in_file_mask')
            bids_method = self.get_parameter('in_file_mask_desc')
            if (explicit_method.user_value is not None and bids_method.user_value is not None):
                logger.warning(
                    f"{full_group_name}: BIDS search for input mask is being overridden by --{explicit_method.flagname}")

    def create_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        wf = super().create_workflow()
        mapnode = True
        inputnode = self.inputnode
        outputnode = self.outputnode
        bids_parameter_group = self.bids

        self.add_bids_to_workflow(wf, inputnode, outputnode, bids_parameter_group, mapnode)
        return wf


class BrainExtractionBIDS(BrainExtraction, CFMMBIDSWorkflowMixer):

    def __init__(self, *args, save_derivatives=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.bids = BIDSAppArguments(owner=self)
        self.bids._modify_parameter('analysis_level', 'choices', ['participant'])

        if save_derivatives:
            in_file_derivatives = {
                'out_file_n4_corrected': 'N4Corrected',
                'out_file_n4_corrected_brain_extracted': 'N4CorrectedBrainExtracted',
                'out_file_mask': 'BrainMask',
            }
        else:
            in_file_derivatives = None

        self.create_bids_input('in_file', output_derivatives=in_file_derivatives)
        self._remove_subcomponent('bse')
        self._remove_subcomponent('ants')
        self.bse = BrainSuiteBrainExtractionBIDS(
            owner=self,
            replaced_parameters={
                'in_file': self.get_parameter('in_file'),
                'in_file_entities_labels_string': self.get_parameter('in_file_entities_labels_string'),
            },
            save_derivatives=False
        )

        self.ants = AntsBrainExtractionBIDS(
            owner=self,
            replaced_parameters={
                'in_file': self.get_parameter('in_file'),
                'in_file_entities_labels_string': self.get_parameter('in_file_entities_labels_string'),
                'in_file_mask': self.get_parameter('in_file_mask'),
                'in_file_mask_entities_labels_string': self.get_parameter('in_file_mask_entities_labels_string'),
                'brain_extract_method': self.get_parameter('brain_extract_method'),
            },
            save_derivatives=False
        )

    def validate_parameters(self):
        super().validate_parameters()
        full_group_name = self.join_nested_groupnames(self.get_nested_groupnames())

        # validations if standalone only
        if self.get_toplevel_owner() == self:
            brain_extract_method = self.get_parameter('brain_extract_method')

            explicit_method = self.get_parameter('in_file')
            bids_method = self.get_parameter('in_file_entities_labels_string')
            if explicit_method.user_value is None and bids_method.user_value is None:
                logger.error(
                    f"{full_group_name}: Either --{explicit_method.flagname} or --{bids_method.flagname} must be supplied")

            if brain_extract_method.user_value in (
                    BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK, BrainExtractMethod.REGISTRATION_NO_INITIAL_MASK,
                    BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK):

                explicit_method = self.ants.get_parameter('template')
                bids_method = self.ants.get_parameter('template_subject')
                if explicit_method.user_value is None and bids_method.user_value is None:
                    logger.error(
                        f"{full_group_name}: When using {brain_extract_method.flagname}={brain_extract_method.user_value}, \n"
                        f"either --{explicit_method.flagname} or --{bids_method.flagname} must be supplied")

                explicit_method = self.ants.get_parameter('template_probability_mask')
                bids_method = self.ants.get_parameter('template_probability_mask_subject')
                if explicit_method.user_value is None and bids_method.user_value is None:
                    logger.error(
                        f"{full_group_name}: When using {brain_extract_method.flagname}={brain_extract_method.user_value}, \n"
                        f"either --{explicit_method.flagname} or --{bids_method.flagname} must be supplied")

            if brain_extract_method.user_value == BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK:
                explicit_method = self.ants.get_parameter('in_file_mask')
                bids_method = self.ants.get_parameter('in_file_mask_desc')
                if explicit_method.user_value is None and bids_method.user_value is None:
                    logger.error(
                        f"{full_group_name}: When using {brain_extract_method.flagname}={brain_extract_method.user_value}, \n"
                        f"either --{explicit_method.flagname} or --{bids_method.flagname} must be supplied")

    def create_workflow(self, arg_dict=None):

        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        wf = super().create_workflow()
        mapnode = True
        inputnode = self.inputnode
        outputnode = self.outputnode
        bids_parameter_group = self.bids
        self.add_bids_to_workflow(wf, inputnode, outputnode, bids_parameter_group, mapnode)

        brain_extraction_method = self.get_parameter('brain_extract_method').user_value
        brainsuite_wf = self.bse.workflow
        ants_wf = self.ants.workflow

        if brain_extraction_method == BrainExtractMethod.BRAINSUITE:
            wf.connect([
                (inputnode, brainsuite_wf, [('in_file', 'inputnode.in_file_original_file')]),
            ])

        elif brain_extraction_method in (
                BrainExtractMethod.REGISTRATION_NO_INITIAL_MASK,
                BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK,
                BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK,
        ):
            wf.connect([
                (inputnode, ants_wf, [('in_file', 'inputnode.in_file_original_file')]),
            ])

        return wf


class BrainExtraction4DBIDS(BrainExtraction4D, CFMMBIDSWorkflowMixer):

    def __init__(self, *args, save_derivatives=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.bids = BIDSAppArguments(owner=self)
        self.bids._modify_parameter('analysis_level', 'choices', ['participant'])

        if save_derivatives:
            in_file_derivatives = {
                'out_file_n4_corrected': 'AvgN4Corrected',
                'out_file_n4_corrected_brain_extracted': 'AvgN4CorrectedBrainExtracted',
                'out_file_mask': 'BrainMask',
            }
        else:
            in_file_derivatives = None

        self.create_bids_input('in_file', output_derivatives=in_file_derivatives)
        self._remove_subcomponent('bse')
        self._remove_subcomponent('ants')
        self.bse = BrainSuiteBrainExtractionBIDS(
            owner=self,
            replaced_parameters={
                'in_file': self.get_parameter('in_file'),
                'in_file_entities_labels_string': self.get_parameter('in_file_entities_labels_string'),
            },
            save_derivatives=False,
        )

        self.ants = AntsBrainExtractionBIDS(
            owner=self,
            replaced_parameters={
                'in_file': self.get_parameter('in_file'),
                'in_file_entities_labels_string': self.get_parameter('in_file_entities_labels_string'),
                'brain_extract_method': self.get_parameter('brain_extract_method'),
            },
            save_derivatives=False,
        )

    def validate_parameters(self):
        super().validate_parameters()
        full_group_name = self.join_nested_groupnames(self.get_nested_groupnames())

        # validations if standalone only
        if self.get_toplevel_owner() == self:
            brain_extract_method = self.get_parameter('brain_extract_method')

            explicit_method = self.get_parameter('in_file')
            bids_method = self.get_parameter('in_file_entities_labels_string')
            if explicit_method.user_value is None and bids_method.user_value is None:
                logger.error(
                    f"{full_group_name}: Either --{explicit_method.flagname} or --{bids_method.flagname} must be supplied")

            if brain_extract_method.user_value in (
                    BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK, BrainExtractMethod.REGISTRATION_NO_INITIAL_MASK,
                    BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK):

                explicit_method = self.ants.get_parameter('template')
                bids_method = self.ants.get_parameter('template_subject')
                if explicit_method.user_value is None and bids_method.user_value is None:
                    logger.error(
                        f"{full_group_name}: When using {brain_extract_method.flagname}={brain_extract_method.user_value}, \n"
                        f"either --{explicit_method.flagname} or --{bids_method.flagname} must be supplied")

                explicit_method = self.ants.get_parameter('template_probability_mask')
                bids_method = self.ants.get_parameter('template_probability_mask_subject')
                if explicit_method.user_value is None and bids_method.user_value is None:
                    logger.error(
                        f"{full_group_name}: When using {brain_extract_method.flagname}={brain_extract_method.user_value}, \n"
                        f"either --{explicit_method.flagname} or --{bids_method.flagname} must be supplied")

            if brain_extract_method.user_value == BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK:
                explicit_method = self.ants.get_parameter('in_file_mask')
                bids_method = self.ants.get_parameter('in_file_mask_desc')
                if explicit_method.user_value is None and bids_method.user_value is None:
                    logger.error(
                        f"{full_group_name}: When using {brain_extract_method.flagname}={brain_extract_method.user_value}, \n"
                        f"either --{explicit_method.flagname} or --{bids_method.flagname} must be supplied")

    def create_workflow(self, arg_dict=None):

        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
            self.validate_parameters()

        wf = super().create_workflow()
        mapnode = True
        inputnode = self.inputnode
        outputnode = self.outputnode
        bids_parameter_group = self.bids

        self.add_bids_to_workflow(wf, inputnode, outputnode, bids_parameter_group, mapnode)

        brain_extraction_method = self.get_parameter('brain_extract_method').user_value
        brainsuite_wf = self.bse.workflow
        ants_wf = self.ants.workflow

        if brain_extraction_method == BrainExtractMethod.BRAINSUITE:
            wf.connect([
                (inputnode, brainsuite_wf, [('in_file', 'inputnode.in_file_original_file')]),
            ])

        elif brain_extraction_method in (
                BrainExtractMethod.REGISTRATION_NO_INITIAL_MASK,
                BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK,
                BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK,
        ):
            wf.connect([
                (inputnode, ants_wf, [('in_file', 'inputnode.in_file_original_file')]),
            ])
        return wf
