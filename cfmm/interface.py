from nipype.interfaces.base.traits_extension import Undefined
from nipype.pipeline import engine as pe

from cfmm.commandline.argparse_type_functions import eval_with_trait_validation
from cfmm.mapnode import MapNode
from cfmm.commandline.parameter_group import ParameterGroup


class Interface(ParameterGroup):
    """
    Class for exposing a nipype interface's input traits to the commandline.
    """

    def __init__(self, nipype_interface, *args, **kwargs):
        """
        :param nipype_interface: nipype interface
        :param args:
        :param kwargs:
        """

        self.interface = nipype_interface
        super().__init__(*args, **kwargs)

    def _add_trait_as_parameter(self, parameter_name, trait):
        """
        Helper function that uses a trait to create an argparse argument.
        :param parameter_name: name of parameter
        :param trait: nipype interface trait
        """
        # setting default to Undefined allows the nipype interface to use its own logic to
        # figure out the default value during initialization. Due to the way nipype does subclassing,
        # when an interface receives Undefined for a parameter sometimes performs complicated logic
        # to eventually figure out the correct default value.
        default = trait.default if trait.usedefault else Undefined

        if type(default) is str:
            # prepare string defaults for eval() inside convert_argparse_using_eval.convert()
            if default == '' or not ((default[0] == default[-1] == "'") or (default[0] == default[-1] == '"')):
                default = '"' + default + '"'

        convert_obj = eval_with_trait_validation(trait.trait_type)
        self._add_parameter(parameter_name,
                            default=default,
                            type=convert_obj.convert,
                            help=trait.desc)

    def _add_parameters(self):
        """
        Adds all of a nipype interface's input traits as commandline arguments.
        """
        # add parser arguments
        parameter_names = list(self.interface().inputs.trait_get().keys())
        parameter_names.sort()
        trait_dict = self.interface.input_spec().traits()
        for parameter in parameter_names:
            self._add_trait_as_parameter(parameter, trait_dict[parameter])

    def get_interface(self, parsed_args_dict=None):
        """
        Create nipype interface with input traits set by user's commandline input. This function should be called after
        self.populate_parameters or parsed_args_dict should be provided so self.populate_parameters can be called.
        :param parsed_args_dict: Dictionary returned by :func:`ArgumentParser.parse_args`
        :return: nipype interface
        """
        if parsed_args_dict is not None:
            self.populate_parameters(parsed_args_dict)

        keyword_arguments = {name: parameter.user_value for name, parameter in self._parameters.items()}
        nipype_interface = self.interface(**keyword_arguments)
        return nipype_interface

    def get_node(self, name=None, parsed_args_dict=None, mapnode=False, **kwargs):
        """
        Helper function returning self.get_interface as a nipype node.
        :param name: Node name
        :param parsed_args_dict: Dictionary returned by :func:`ArgumentParser.parse_args`
        :return: nipype node
        """
        if name is None:
            name = self.interface.__name__
        nipype_interface = self.get_interface(parsed_args_dict)
        return MapNode(interface=nipype_interface, name=name, **kwargs) if mapnode else pe.Node(
            interface=nipype_interface, name=name, **kwargs)
