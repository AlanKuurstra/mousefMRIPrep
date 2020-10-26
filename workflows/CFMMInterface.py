from nipype.interfaces.base.traits_extension import isdefined, Undefined
import configargparse as argparse
from traits.has_traits import HasTraits
from nipype.pipeline import engine as pe
from workflows.CFMMMapNode import CFMMMapNode
from workflows.CFMMParameterGroup import CFMMParameterGroup

class convert_argparse_using_eval():
    """
    Class used by :func:`CFMMInterface.convert_trait_to_argument` and :func:`ArgumentParser.add_argument` to convert
    commandline text during argument parsing.  Python's eval is used to cast a commandline string to a python object
    which is then validated as a useable input for the desired trait.
    """

    def __init__(self, trait_type):
        """
        Save the trait so self.convert knows which trait to validate the argparse input for.
        :param trait_type:
        """
        self.trait_type = trait_type

    def convert(self, argparse_value):
        """
        The function provided to type argument of :func:`ArgumentParser.add_argument`. Python's eval is used to cast a
        commandline string to a python object which is then validated as a useable input for self.trait_type.
        :param argparse_value:
        :return:
        """
        # everything that comes from the commandline will be put through python's eval()
        # this causes trouble for strings - every argument from the commandline is read in as a string
        # so it's hard to determine which arguments were meant as strings for eval() and which arguments were meant
        # as something else for eval

        # when a user is inputing a string for eval(), we require them to indicate it using double quotes
        # it's not good enough to use:
        # --string_input mystringinput
        # --string_input 'mystringinput'
        # --string_input "mystringinput"
        # because they will all evaluate to the string mystringinput which can't be processed by eval()

        # We require the user to put
        # --string_input '"mystringinput"' which evaluates to "mystringinput" and can be processed by eval()
        # or
        # --string_input "'mystringinput'"which evaluates to 'mystringinput' and can be processed by eval()

        # similarly, when using a dictionary or list the user should use
        # --my_dict_input "{'string_key':3}"
        # --my_list_input "['string1','string2']"

        # this is annoying for the user, but low maitenance
        # since users will mostly use config files, the annoying strings on the commandline are acceptable


        # we could put in logic for strings outside of lists to behave differently (ie. enum and string traits)

        # if trait is string, don't do eval() - but then a string in list and on it's own is input differently

        # with enum traits we have the additional problem of inputting a python None object.
        # some numerated types have both 'None' the string and None the python object
        # using eval we can differentiate the two with '"None"' for the string and 'None" for the python object
        # But without eval, this becomes difficult.  We could decide that '' will convert to the python None object.
        # And every other string input can be cast using the enum.values.
        # But then None in enum and None in a list or tuple is input differently on the commandline

        try:
            if argparse_value == "<undefined>":
                trait_value = Undefined
            else:
                trait_value = eval(argparse_value)
        except Exception as e:
            raise argparse.ArgumentTypeError(
                f'input "{argparse_value}" must be a valid input for python\'s eval(). Did you forget quotes around a string? eg. for a string input use "\'string_1\'" and for a list input use "[\'string_1\',\'string_2\']"')
        class dummy(HasTraits):
            trait_argument = self.trait_type

        try:
            # Undefined throws an error with this trait type check, but is actually a special type that is safe to
            # use when setting a trait value.
            if trait_value != Undefined:
                self.trait_type.validate(dummy(), 'trait_argument', trait_value)
        except Exception as e:
            raise argparse.ArgumentTypeError(
                str(e).replace("'trait_argument' trait of a dummy instance", f'input "{str(trait_value)}"'))
        return trait_value



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