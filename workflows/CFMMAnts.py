from cfmm.commandline.parameter_group import HierarchicalParameterGroup
from cfmm.interface import Interface
from nipype.interfaces.ants import Registration, N4BiasFieldCorrection, ApplyTransforms, ThresholdImage
from nipype.pipeline import engine as pe
from nipype.interfaces.utility import Function


class AntsDefaultArguments(HierarchicalParameterGroup):
    group_name = "ANTs Default Arguments"
    flag_prefix = 'antsarg_'

    def _add_parameters(self):
        # add parser arguments
        self._add_parameter('interpolation',
                            default="'Linear'",
                            help="Default interpolation method to use in ANTs programs.")
        self._add_parameter('float',
                            action='store_true',
                            help="ANTs programs will default to using float instead of double for computations.")

        # get values from upstream if not specified on commandline
        # this works if the parent sets their AntsDefaultArguments before other pieces
        # should maybe make an ANTSWorkflow class which has an __init__ ensuring this
        curr_component = self.owner
        upstream_ants_default_arguments = None
        while curr_component.owner is not None and upstream_ants_default_arguments is None:
            curr_component = curr_component.owner
            for subcomponent in curr_component.subcomponents:
                if type(subcomponent) == type(self):
                    upstream_ants_default_arguments = subcomponent
        if upstream_ants_default_arguments:
            self.copy_node_defaults(upstream_ants_default_arguments, self)


class CFMMN4BiasFieldCorrection(Interface):
    group_name = 'N4 Correction'
    flag_prefix = 'n4_'

    def __init__(self, *args, **kwargs):
        super().__init__(N4BiasFieldCorrection, *args, **kwargs)

    def _add_parameters(self):
        super()._add_parameters()
        self._modify_parameter('bspline_fitting_distance',
                               'help',
                               'Distance between spline knots. Should be appropriate for object size.')


class CFMMAntsRegistration(Interface):
    group_name = 'ANTs Registration'
    flag_prefix = 'antsreg_'

    def __init__(self, *args, **kwargs):
        super().__init__(Registration, *args, **kwargs)

    def _add_parameters(self):
        super()._add_parameters()


class CFMMApplyTransforms(Interface):
    group_name = 'Apply Transforms'
    flag_prefix = 'applytf_'

    def __init__(self, *args, **kwargs):
        super().__init__(ApplyTransforms, *args, **kwargs)

    def _add_parameters(self):
        super()._add_parameters()
        self._modify_parameter('dimension', 'default', "3")
        self._modify_parameter('output_image', 'default', "'deformed_mask.nii'")
        self._modify_parameter('interpolation', 'default', "'Linear'")
        self._modify_parameter('default_value', 'default', "0")
        # apply_transform.inputs.transforms = ['ants_Warp.nii.gz', 'trans.mat']
        # apply_transform.inputs.invert_transform_flags = [False, False]
        # apply_transform.inputs.transforms = composite_transform


class CFMMThresholdImage(Interface):
    group_name = "Threshold Image"
    flag_prefix = 'thresh_'

    def __init__(self, *args, **kwargs):
        super().__init__(ThresholdImage, *args, **kwargs)

    def _add_parameters(self):
        super()._add_parameters()
        self._modify_parameter('dimension', 'default', "3")
        self._modify_parameter('th_low', 'default', "0.5")
        self._modify_parameter('th_high', 'default', "1.0")
        self._modify_parameter('inside_value', 'default', "1")
        self._modify_parameter('outside_value', 'default', "0")


def ants_transform_concat_list(apply_first=None,
                               apply_second=None,
                               apply_third=None,
                               apply_fourth=None,
                               apply_fifth=None,
                               apply_sixth=None,
                               apply_seventh=None,
                               apply_eighth=None,
                               apply_nineth=None,
                               apply_tenth=None,
                               ):
    # ApplyTransforms wants transforms listed in reverse order of application
    from cfmm.CFMMCommon import existing_inputs_to_list
    list = existing_inputs_to_list(apply_first,
                                   apply_second,
                                   apply_third,
                                   apply_fourth,
                                   apply_fifth,
                                   apply_sixth,
                                   apply_seventh,
                                   apply_eighth,
                                   apply_nineth,
                                   apply_tenth, )
    list.reverse()
    return list


def get_node_ants_transform_concat_list(name='ants_transform_concat_list'):
    node = pe.Node(
        Function(input_names=[
            'apply_first',
            'apply_second',
            'apply_third',
            'apply_fourth',
            'apply_fifth',
            'apply_sixth',
            'apply_seventh',
            'apply_eighth',
            'apply_nineth',
            'apply_tenth',
        ],
            output_names=["transforms"],
            function=ants_transform_concat_list), name=name)
    return node
