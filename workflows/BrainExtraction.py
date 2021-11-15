from workflows.CFMMAnts import AntsDefaultArguments, CFMMApplyTransforms, CFMMThresholdImage, CFMMAntsRegistration, \
    CFMMN4BiasFieldCorrection
from cfmm.workflow import Workflow
from cfmm.bids_parameters import BIDSWorkflowMixin, BIDSInputExternalSearch, CMDLINE_VALUE
from workflows.CFMMBrainSuite import CFMMBse
from workflows.CFMMEnums import BrainExtractMethod
from nipype.interfaces.fsl import ImageMaths, CopyGeom, ApplyMask, Reorient2Std
from cfmm.CFMMCommon import NipypeWorkflowArguments, get_node_delistify
from cfmm.logging import NipypeLogger as logger
from cfmm.interface import Interface
from nipype.interfaces.fsl import ExtractROI
from cfmm.mapnode import MapNode


# Iteration of brain extraction classes was accomplished using mapnodes.
# However if the brain extraction workflows are used as subworkflows in a parent which does not use mapnodes,
# the brain extraction workflow will output singleton lists which cause problems for the parent worfklow when connected
# to normal nodes (since normal nodes want a single item, not a list of a single item). Thus we delist singleton lists.


class BrainSuiteBrainExtraction(Workflow):
    group_name = 'BrainSuite Brain Extraction'
    flag_prefix = 'bs_be_'

    def _add_parameters(self):
        self._add_parameter('in_file',
                            help='Explicitly specify location of the input file for brain extraction.')

    def __init__(self, *args, **kwargs):
        # super() deals with the ordering of setting class groupname, dealing with upstream parameter replacements
        # and executing local _add_parser_arguments()
        super().__init__(*args, **kwargs)

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

        # bse segfaults on some orientations - however it doesn't happen if the image is in std orientation
        reorient2std = MapNode(interface=Reorient2Std(), name='reorient2std', iterfield=['in_file'])

        bse = self.get_subcomponent(CFMMBse.group_name).get_node(name='BSE', mapnode=True, iterfield=['inputMRIFile'])

        # default behaviour of brainsuite is to rotate to LPI orientation
        # this can be overridden by using the noRotate option, however this option will create a nifti with inconsistent
        # qform and sform values.  To fix this, copy the header information from the original image to the mask using fsl.

        # should actually flip back to original orientation.

        fix_bse_orientation = MapNode(interface=CopyGeom(), name='fixBSEOrientation',
                                      iterfield=['in_file', 'dest_file'])


        # brainsuite outputs mask value as 255, change it to 1
        fix_bse_value = MapNode(interface=ImageMaths(), name='fixBSEValue', iterfield=['in_file'])
        fix_bse_value.inputs.op_string = '-div 255'

        apply_mask = MapNode(ApplyMask(), name='apply_mask', iterfield=['in_file', 'mask_file'])
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


class AntsBrainExtraction(Workflow):
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
                            help='Explicitly specify location of the probability mask used in registration-based brain extraction.')

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
        apply_mask = MapNode(ApplyMask(), name='apply_mask', iterfield=['in_file', 'mask_file'],
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


class BrainExtraction(Workflow):
    group_name = 'Brain Extraction'
    flag_prefix = 'be_'

    def _add_parameters(self):
        self._add_parameter('in_file',
                            help='Explicitly specify location of the input file for brain extraction.',
                            iterable=True
                            )

        self._add_parameter('in_file_mask',
                            help='Explicitly specify location of an initial input file mask used in registration-based brain extraction.',
                            iterable=True
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
        self.n4 = CFMMN4BiasFieldCorrection(owner=self)

        # what if in_file is disabled?
        self.bse = BrainSuiteBrainExtraction(
            owner=self,
            exclude_list=['in_file']
        )

        self.ants = AntsBrainExtraction(
            owner=self,
            exclude_list=['in_file', 'in_file_mask'],
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


        n4 = self.n4.get_node(name='n4', mapnode=True, iterfield=['input_image'])
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
            apply_mask = MapNode(ApplyMask(), name='apply_mask', iterfield=['in_file', 'mask_file'])
            wf.connect([
                (inputnode, apply_mask, [('in_file_mask', 'mask_file')]),
                (n4, apply_mask, [('output_image', 'in_file')]),
                (apply_mask, outputnode, [('out_file', 'out_file_n4_corrected_brain_extracted')]),
                (inputnode, outputnode, [('in_file_mask', 'out_file_mask')]),
            ])
        return wf


class CFMMExtractROI(Interface):
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


class BrainExtractionBIDS(BrainExtraction, BIDSWorkflowMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_bids_parameter_group()
        self.bids._modify_parameter('analysis_level', 'choices', ['participant'])

        self.in_file_bids = BIDSInputExternalSearch(self,
                                                    'in_file',
                                                    entities_to_overwrite={'subject': CMDLINE_VALUE,
                                                                           'session': CMDLINE_VALUE,
                                                                           'run': CMDLINE_VALUE,
                                                                           'extension': ['.nii', '.nii.gz'],
                                                                           },
                                                    output_derivatives={
                                                        'out_file_n4_corrected': 'N4Corrected',
                                                        'out_file_n4_corrected_brain_extracted': 'N4CorrectedBrainExtracted',
                                                        'out_file_mask': 'BrainMask',
                                                    },
                                                    derivatives_mapnode=True)

        self.in_file_mask_bids = BIDSInputExternalSearch(self,
                                                         'in_file_mask',
                                                         dependent_search=self.in_file_bids,
                                                         dependent_entities=['subject', 'session', 'run'],
                                                         create_base_bids_string=False,
                                                         entities_to_overwrite={
                                                             'desc': CMDLINE_VALUE,
                                                             'extension': ['.nii', '.nii.gz'],
                                                         },
                                                         )

    def validate_parameters(self):
        super().validate_parameters()
        # if using registration method, template and template probability mask should exist
        # subworkflow validations depend on this wf flow control ie. don't need to use ANTs subcomponent validations
        # if method is BrainSuite
        # make sure a mask is provided if method is REGISTRATION_WITH_INITIAL_MASK or USER_PROVIDED_MASK



    def create_workflow(self, arg_dict=None):
        wf = super().create_workflow()
        self.add_bids_to_workflow(wf)

        return wf


class BrainExtraction4DBIDS(BrainExtraction4D, BIDSWorkflowMixin):

    def __init__(self, *args, save_derivatives=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_bids_parameter_group()
        self.bids._modify_parameter('analysis_level', 'choices', ['participant'])

        self.in_file_bids = BIDSInputExternalSearch(self,
                                                    'in_file',
                                                    entities_to_overwrite={'subject': CMDLINE_VALUE,
                                                                           'session': CMDLINE_VALUE,
                                                                           'run': CMDLINE_VALUE,
                                                                           'extension': ['.nii', '.nii.gz'],
                                                                           },
                                                    output_derivatives={
                                                        'out_file_n4_corrected': 'AvgN4Corrected',
                                                        'out_file_n4_corrected_brain_extracted': 'AvgN4CorrectedBrainExtracted',
                                                        'out_file_mask': 'BrainMask',
                                                    },
                                                    derivatives_mapnode=True)

        self.in_file_mask_bids = BIDSInputExternalSearch(self,
                                                         'in_file_mask',
                                                         dependent_search=self.in_file_bids,
                                                         dependent_entities=['subject', 'session', 'run'],
                                                         create_base_bids_string=False,
                                                         entities_to_overwrite={
                                                             'desc': CMDLINE_VALUE,
                                                             'extension': ['.nii', '.nii.gz'],
                                                         },
                                                         )


    def create_workflow(self, arg_dict=None):
        wf = super().create_workflow()
        self.add_bids_to_workflow(wf)

        return wf

