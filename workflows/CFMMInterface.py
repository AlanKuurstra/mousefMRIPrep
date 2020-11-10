from nipype.interfaces.base.traits_extension import isdefined, Undefined
from nipype.pipeline import engine as pe
from workflows.CFMMMapNode import CFMMMapNode
from workflows.CFMMParameterGroup import CFMMParameterGroup
from workflows.argparse_conversion_functions import convert_argparse_using_eval

class CFMMInterface(CFMMParameterGroup):
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

    def convert_trait_to_argument(self, parameter_name, trait):
        """
        Helper function that uses a trait to create an argparse argument.
        :param parameter_name: name of parameter
        :param trait: nipype interface trait
        """
        convert_obj = convert_argparse_using_eval(trait.trait_type)
        default = Undefined
        if trait.usedefault:
            default = trait.default

        if type(default) is str:
            # prepare string defaults for eval() inside convert_argparse_using_eval.convert()
            if default == '' or not ((default[0] == default[-1] == "'") or (default[0] == default[-1] == '"')):
                default = '"' + default + '"'

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
            self.convert_trait_to_argument(parameter, trait_dict[parameter])

    def get_interface(self, parsed_args_dict=None):
        """
        Create nipype interface with input traits set by user's commandline input. This function should be called after
        self.populate_parameters or parsed_args_dict should be provided so self.populate_parameters can be called.
        :param parsed_args_dict: Dictionary returned by :func:`ArgumentParser.parse_args`
        :return: nipype interface
        """
        if parsed_args_dict is not None:
            self.populate_parameters(parsed_args_dict)

        keyword_arguments = {parameter_name:parameter.user_value for parameter_name,parameter in self._parameters.items()}

        nipype_interface = self.interface(**keyword_arguments)

        # # sometimes an interface has an input with default value <undefined>.  The init function of the interface
        # # knows what to do when the input <undefined>, however if you attempt to set the input to <undefined> after
        # # the interface is already created, then an error is thrown. better to set defaults during init
        # for parameter in self._parameters.keys():
        #     user_value = self.get_parameter(parameter).user_value
        #     setattr(nipype_interface.inputs, parameter, user_value)
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
        if mapnode:
            node = CFMMMapNode(interface=nipype_interface, name=name, **kwargs)
        else:
            node = pe.Node(interface=nipype_interface, name=name, **kwargs)
        return node