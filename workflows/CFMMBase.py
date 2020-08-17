from nipype.interfaces.base.traits_extension import isdefined, Undefined
import configargparse as argparse
from traits.has_traits import HasTraits
import os
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from workflows.CFMMEnums import ParameterListeningMode
from workflows.CFMMLogging import NipypeLogger as logger

class CFMMFlagValuePair():
    """
    Class storing an argparse action, its command line flag, and the value returned from :func:`ArgumentParser.parse_args`.
    There is optional functionality for indicating if the parameter value should instead be obtained from an argument
    hierarchy.
    """

    def __init__(self, parser_flag, user_value, parser_action,
                 superior_parameter=None, listening_mode=ParameterListeningMode.IGNORE):
        """
        :param parser_flag: argparse flag for optional arguments and (parser return dictionary) dest for optional and positional arguments
        :type parser_flag: str
        :param user_value: value from parser return dictionary
        :param parser_action: argparse action used to change argument (eg. used to change default value, or command line help, or even to hide the argument)
        :param add_to_inputnode: Boolean determining if this argument should be added to a corresponding workflow's inputnode. See :func:`CFMMParserArguments.get_inputnode`
        :param superior_parameter: Parameter that either overrides or provides a default value for this parameter depending on listening_mode. Stored as tuple (parameter name string, CFMMParserArguments instance). See :func:`CFMMParserArguments.populate_parameters`
        :param subordinate_parameters: Parameters that this parameter overrides or provides default values for. List of tuples (parameter name string, CFMMParserArguments instance). See :func:`CFMMWorkflow.pass_along_input`
        :param listening_mode: Determines whether a superior parameter should override this parameter or just provide a default value.
        """
        self.parser_flag = parser_flag
        self.user_value = user_value
        self.parser_action = parser_action
        self.superior_parameter = superior_parameter
        self.listening_mode = listening_mode

    def replace_default_by(self, superior):
        """
        Specify another CFMMFlagValuePair to use as the default value instead of using an argparse default value.
        :param superior: CFMMFlagValuePair instance to use as default value
        """
        self.parser_action.default = argparse.SUPPRESS
        self.superior_parameter = superior
        self.listening_mode = ParameterListeningMode.REPLACE_DEFAULT

    def replaced_by(self, superior):
        """
        Specify a CFMMFlagValuePair instance that will provide this parameter's value. This parameter will be removed
        from the command line help.
        :param superior: CFMMFlagValuePair instance to use as default value
        """
        # unable to remove existing argparse parameters. https://bugs.python.org/issue19462#msg251739
        self.parser_action.default = argparse.SUPPRESS
        self.parser_action.help = argparse.SUPPRESS
        self.superior_parameter = superior
        self.listening_mode = ParameterListeningMode.REPLACE_VALUE

    # def get_highest_replacing_superior(self):
    #     current_parameter = self
    #     while current_parameter.obtaining_value_from_superior():
    #         current_parameter = current_parameter.superior_parameter
    #     return current_parameter

    def obtaining_value_from_superior(self, parsed_args_dict=None):
        """
        Indicates whether or not this instance is expected to obtain its user_value from self.superior.
        :param parsed_args_dict: Dictionary returned by :func:`ArgumentParser.parse_args`
        :return: Boolean
        """
        if self.listening_mode == ParameterListeningMode.IGNORE:
            return False
        elif self.listening_mode == ParameterListeningMode.REPLACE_VALUE:
            return True
        elif self.listening_mode == ParameterListeningMode.REPLACE_DEFAULT:
            if parsed_args_dict is None:
                return None
            return not (self.parser_flag in parsed_args_dict.keys())

    def populate_user_value(self, parsed_args_dict):
        """
        Traverse superiors to return the value that should be stored in self.user_value.
        :param parsed_args_dict: Dictionary returned by :func:`ArgumentParser.parse_args`
        :type parsed_args_dict: dict
        :param ignore_superiors_existing_values: If True, ignores any superior's manually set user_value and returns the value that would have been automatically populated
        :param store_superiors: If True, stores the superiors' user_value while traversing
        :return: value that should be stored in self.user_value
        """
        if not self.obtaining_value_from_superior(parsed_args_dict):
            self.user_value = parsed_args_dict[self.parser_flag]
        else:
            self.user_value = self.superior_parameter.populate_user_value(parsed_args_dict)
        return self.user_value

    def __str__(self):
        return f'({self.parser_flag},{self.user_value},{"Action present" if self.parser_action else None})'

    def __repr__(self):
        return f'({self.parser_flag},{self.user_value},{"Action present" if self.parser_action else None})'


class CFMMParserArguments():
    """
    Base class for grouping a number of related argparse arguments.
    """
    def __init__(self, group_name=None, parent=None, parser=None, exclude_list=None, flag_prefix=None,
                 flag_suffix=None):
        """
        :param group_name: Name used in parser's argument group. See :func:`CFMMParserArguments.set_parser`.
        :param parent: If the current instance is a subcomponent, parent stores the owner.
        :param parser: The parser instance that arguments will be added to.
        :param exclude_list: List of names of arguments to exclude from the group.
        :param flag_prefix: Prefix to add to flag names to make unique.
        :param flag_suffix: Suffix to add to flag names to make unique.
        """
        if group_name is not None:
            self.group_name = group_name
        elif hasattr(self, 'group_name'):
            pass
        else:
            self.group_name = self.__class__.__name__

        # prefix and suffix facilitate individualising commandline flags
        if flag_prefix is not None:
            self.flag_prefix = flag_prefix
        elif hasattr(self, 'flag_prefix'):
            pass
        else:
            self.flag_prefix=''

        if flag_suffix is not None:
            self.flag_suffix = flag_suffix
        elif hasattr(self, 'flag_suffix'):
            pass
        else:
            self.flag_suffix=''

        self.parent = parent

        if type(exclude_list) in (str, type(None)):
            self.exclude_list = [exclude_list]
        elif type(exclude_list) != list:
            self.exclude_list = list(exclude_list)
        else:
            self.exclude_list = exclude_list

        self._parameters = {}

        if parser is not None:
            self.set_parser(parser)
            self.add_parser_arguments()

    def get_concatenated_flag_affixes(self):
        """
        Traverse the composition chain and concatenate all flag affixes to obtain a unique prefix and suffix for the flag.
        :return: concatenated_flag_prefix, concatenated_flag_suffix
        """
        current_parent = self.parent
        flag_prefix_list = [self.flag_prefix]
        flag_suffix_list = [self.flag_suffix]
        while current_parent is not None:
            flag_prefix_list.insert(0, current_parent.flag_prefix)
            flag_suffix_list.insert(0, current_parent.flag_suffix)
            current_parent = current_parent.parent

        concatenated_flag_prefix = ''.join(flag_prefix_list)
        concatenated_flag_suffix = ''.join(flag_suffix_list)
        return concatenated_flag_prefix, concatenated_flag_suffix

    def get_nested_group_names(self):
        """
        Traverse the composition chain and append all group_names to a list.
        :return: group_names
        """
        group_names=[]
        current_component = self
        while current_component.parent is not None:
            group_names.append(current_component.group_name)
            current_component = current_component.parent
        group_names.append(current_component.group_name)
        group_names.reverse()
        return group_names

    def get_nested_group_name_str(self):
        """
        Traverse the composition chain and append all group_names to a list.
        :return: concatenated_group_name
        """

        return os.sep.join(self.get_nested_group_names())

    def get_group_name_chain(self):
        """
        Traverse the composition chain and concatenate all group names to obtain a nested, unique argument group name.
        :return: concatenated_group_name
        """
        current_child = self
        current_parent = self.parent
        group_names = []
        while current_parent is not None:
            for subcomponent in current_parent.subcomponents:
                if subcomponent == current_child:
                    group_names.insert(0, subcomponent.group_name)
                    break
            current_child = current_parent
            current_parent = current_parent.parent
        if current_child.flag_prefix != '':
            group_names.insert(0, current_child.flag_prefix.rstrip('_'))
        return os.sep.join(group_names)

    def get_toplevel_parent(self):
        """
        Traverse the composition chain and return first instance without a parent.
        :return: toplevel_parent
        """

        current_child = self
        current_parent = current_child.parent
        while current_parent is not None:
            current_child = current_parent
            current_parent = current_parent.parent
        return current_child

    def set_parser(self, parser):
        """
        Set the parser instance (common to all nested CFMMParserArguments instances) and the
        parser argument group (unique to this CFMMParserArguments instance)
        :param parser: parser instance
        """
        self.parser = parser
        #nested_group_name = self.get_group_name_chain()
        #print(self.get_group_name_chain(), self.get_nested_group_names(), self.get_nested_group_names()[:-1])
        tmp=self.get_nested_group_names()[1:]
        nested_group_name = os.sep.join(self.get_nested_group_names()[1:])
        if nested_group_name == '':
            self.parser_group = parser
        else:
            self.parser_group = parser.add_argument_group(nested_group_name)

    def add_parser_argument(self, parameter_name, *args, parser_flag=None, optional=True,
                            **kwargs):
        """
        Helper class for :func:`CFMMParserArguments.add_parser_arguments`.
        :param parameter_name: Name of parameter to be added to argument_group
        :param args: arguments for :func:`ArgumentParser.add_argument`
        :param parser_flag: Optional flag that will override automated flag name
        :param optional: If true add as keyword commandline argument, if false add as positional commandline argument
        :param add_to_inputnode: If true, will be added as field of the identity interface returned by :func:`CFMMParserArguments.get_inputnode`
        :param kwargs: keyword arguments for :func:`ArgumentParser.add_argument`
        """

        if parameter_name not in self.exclude_list:
            parser_group = self.parser_group
            if parameter_name not in self._parameters.keys():
                self._parameters[parameter_name] = CFMMFlagValuePair(None, None, None)
            # in order to customize the flag name, you can use the parser_flag keyword
            # if parser_flag keyword is not set and an existing flag name is in self.get_parameter(parameter_name).parser_flag
            # then the existing flag value will be respected
            # otherwise, a default flag will be created
            if parser_flag is None:
                parser_flag = self.get_parameter(parameter_name).parser_flag
            if parser_flag is None:
                concatenated_flag_prefix, concatenated_flag_suffix = self.get_concatenated_flag_affixes()
                parser_flag = f'{concatenated_flag_prefix}{parameter_name}{concatenated_flag_suffix}'
                self.get_parameter(parameter_name).parser_flag = parser_flag

            flag_dash = ''
            if optional:
                flag_dash = '--'

            self.get_parameter(parameter_name).parser_action = parser_group.add_argument(
                f'{flag_dash}{parser_flag}', *args, **kwargs)

    def modify_parser_argument(self, parameter_name, attribute, value):
        """
        :param parameter_name: Name of parameter to modify (first argument given to :func:`CFMMParserArguments.add_parser_argument` and the keyword in self._parameters).
        :param attribute: argparse action attribute to modify (eg. default, help, etc)
        :param value: value to give argparse action attribute
        """
        if attribute == 'flag':
            # not sure how well this will work, might need to argparse.SUPPRESS existing parameter and create a new one
            setattr(self.get_parameter(parameter_name).parser_action, 'option_strings', [f'--{value}'])
            setattr(self.get_parameter(parameter_name).parser_action, 'dest', value)
        else:
            setattr(self.get_parameter(parameter_name).parser_action, attribute, value)

    def add_parser_arguments(self):
        """
        To be implemented by subclass. Customize commandline arguments to add to the parser and store in
        self._parameters. Use helper function `CFMMParserArguments.add_parser_argument`.
        """
        raise NotImplementedError('Subclass must define add_parser_arguments function.')

    def hide_parser_argument(self, parameter_name):
        """
        Set an argument's help to argparse.SUPPRESS to hide it from the command line help output.
        :param parameter_name: Name of parameter to modify (first argument given to :func:`CFMMParserArguments.add_parser_argument` and the keyword in self._parameters).
        """
        # unable to remove existing argparse parameters. https://bugs.python.org/issue19462#msg251739
        self.get_parameter(parameter_name).parser_action.help = argparse.SUPPRESS

    @staticmethod
    def replace_defaults(superior, subordinate):
        """
        Use parameter values from the superior CFMMParserArguments instance as defaults for the parameter values
        of the subordinate CFMMParserArguments instance.
        :param superior: Supplies values to be used as default
        :param subordinate: Receives values to be set as default
        """
        for parameter_name in subordinate._parameters.keys():
            superior_param = superior.get_parameter(parameter_name)
            subordinate_param = subordinate.get_parameter(parameter_name)
            subordinate_param.replace_default_by(superior_param)

    def replace_defaults_by(self, superior):
        """
        Use parameter values from the superior CFMMParserArguments instance as defaults for the parameter values
        of this instance.
        :param superior: Supplies values to be used as default
        """
        for parameter_name in superior._parameters.keys():
            superior_param = superior.get_parameter(parameter_name)
            subordinate_param = self.get_parameter(parameter_name)
            subordinate_param.replace_default_by(superior_param)

    def populate_parameters(self, parsed_args_dict):
        """
        Automatically populate each parameter's user_value.
        :param parsed_args_dict: Dictionary returned by :func:`ArgumentParser.parse_args`
        """
        for parameter_name, parameter in self._parameters.items():
            if parameter_name not in self.exclude_list:
                parameter.populate_user_value(parsed_args_dict=parsed_args_dict)

    def get_parameter(self, parameter_name):
        """
        :param parameter_name:
        :return: CFMMFlagValuePair object stored in self._parameters
        """
        return self._parameters[parameter_name]

    def validate_parameters(self):
        """
        To be implemented by subclass. Give warning and errors to the user if commandline options are
        mutually exclusive.
        """
        pass




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
        # this is annoying for the user, but low maitenance
        # since users will mostly use config files, the annoying strings are acceptable
        # we can put in logic for strings outside of lists to behave differently (ie. enum and string traits)
        # if trait is string, don't do eval()
        # if trait is enum, '' converts to None, and the casting should be done by the enum.values ignoring a
        # None enum.value if it exists
        # but then None in enum and None in a list or tuple is input differently.
        try:
            trait_value = eval(argparse_value)
        except Exception as e:
            raise argparse.ArgumentTypeError(
                f'input "{argparse_value}" must be a valid input for python\'s eval(). Did you forget quotes around a string? eg. for a string input use "\'string_1\'" and for a list input use "[\'string_1\',\'string_2\']"')

        class dummy(HasTraits):
            trait_argument = self.trait_type

        try:
            self.trait_type.validate(dummy(), 'trait_argument', trait_value)
        except Exception as e:
            raise argparse.ArgumentTypeError(
                str(e).replace("'trait_argument' trait of a dummy instance", f'input "{str(trait_value)}"'))
        return trait_value


class CFMMInterface(CFMMParserArguments):
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
        self.add_parser_argument(parameter_name,
                                 default=default,
                                 type=convert_obj.convert,
                                 help=trait.desc)

    def add_parser_arguments(self):
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
        nipype_interface = self.interface()
        for parameter in self._parameters.keys():
            user_value = self.get_parameter(parameter).user_value
            setattr(nipype_interface.inputs, parameter, user_value)
        return nipype_interface

    def get_node(self, name='CFMMNode', parsed_args_dict=None, **kwargs):
        """
        Helper function returning self.get_interface as a nipype node.
        :param name: Node name
        :param parsed_args_dict: Dictionary returned by :func:`ArgumentParser.parse_args`
        :return: nipype node
        """
        nipype_interface = self.get_interface(parsed_args_dict)
        return pe.Node(interface=nipype_interface, name=name, **kwargs)


class CFMMWorkflow(CFMMParserArguments):
    """
    Class for managing a nipype workflow with commandline arguments. Manages a list self.subcomponents of
    CFMMParserArguments subclasses (eg. a CFMMInterface or a CFMMWorkflow used as a subworkflow) which will have their
    commandline arguments displayed in this workflow's help.
    """

    def __init__(self, subcomponents, *args, pipeline_name=None, pipeline_version=None, **kwargs):
        """
        :param subcomponents: List of CFMMParserArguments subclasses
        :param args:
        :param pipeline_name:
        :param pipeline_version:
        :param kwargs:
        """
        for proposed_subcomponent in subcomponents:
            self.add_subcomponent(proposed_subcomponent)
        if pipeline_name is None:
            pipeline_name = self.__class__.__name__
        self.pipeline_name = pipeline_name
        self.pipeline_short_desc = ''

        self._inputnode_params = []
        self._param_subordinates = {}
        self.calling_subclass = None
        self.inputnode = None
        self.outputnode = None
        self.workflow = None


        if pipeline_version is None:
            import inspect
            from git import Repo
            subclass_dir = inspect.getfile(self.__class__)
            subclass_git_repo = Repo(subclass_dir, search_parent_directories=True)
            self.pipeline_version = subclass_git_repo.head.commit.hexsha
        super().__init__(*args, **kwargs)

    def set_parser(self, parser):
        """
        Set the parser for all subcomponents.
        :param parser:
        """
        # location of super().set_parser() call determines order of argparser help groups
        # putting super() at the beginning means the nested group arguments are shown below their owners

        super().set_parser(parser)
        for subcomponent in self.subcomponents:
            subcomponent.set_parser(parser)

    def add_parser_argument(self, parameter_name, *args, add_to_inputnode=True, override_parameters=None, **kwargs):
        """
        Helper function for :func:`CFMMParserArguments.add_parser_argument`. Allows a workflow parameter to hide
        and override parameters from its subcomponents.
        :param parameter_name: Name of parameter to be added to argument_group
        :param args:
        :param override_parameters: Subcomponent parameters to be overridden by current parameter.
        :param kwargs:
        """
        super().add_parser_argument(parameter_name, *args, **kwargs)
        if add_to_inputnode:
            self._inputnode_params.append(parameter_name)
        if override_parameters is not None:
            superior_parameter = self.get_parameter(parameter_name)
            for subordinate_parameter_name, subordinate_subcomponent_name in override_parameters:
                subordinate_subcomponent = self.get_subcomponent(subordinate_subcomponent_name)
                # hide overridden parameter
                subordinate_subcomponent.get_parameter(subordinate_parameter_name).replaced_by(superior_parameter)
                self._param_subordinates[superior_parameter] = (subordinate_parameter_name, subordinate_subcomponent)

    def add_parser_arguments(self):
        """
        Adds parser arguments from all subcomponents. This function should be called from a user redefined
        add_parser_arguments using super().add_parser_arguments().
        """
        for subcomponent in self.subcomponents:
            if not subcomponent.parent == self:
                continue
            subcomponent.add_parser_arguments()

    def populate_parameters(self, parsed_args_dict):
        """
        Automatically populate the parameters from all subcomponents.
        :param parsed_args_dict: Dictionary returned by :func:`ArgumentParser.parse_args`
        """
        for subcomponent in self.subcomponents:
            if not subcomponent.parent == self:
                continue
            subcomponent.populate_parameters(parsed_args_dict)
        super().populate_parameters(parsed_args_dict)

    def validate_parameters(self):
        """
        Validate user inputs for arguments in all subcomponents.
        """
        for subcomponent in self.subcomponents:
            if not subcomponent.parent == self:
                continue
            subcomponent.validate_parameters()

    def add_subcomponent(self, subcomponent):
        """
        Add a subcomponent to this workflow and assign self as parent.
        :param subcomponent:
        """
        if not hasattr(self, 'subcomponents'):
            self.subcomponents = []
        # maybe should still use a dictionary _subcomponents for unique keys
        proposed_subcomponents_name = subcomponent.group_name
        for existing_subcomponent in self.subcomponents:
            if proposed_subcomponents_name == existing_subcomponent.group_name:
                raise ValueError(
                    f'Cannot add subcomponent with group_name {proposed_subcomponents_name}, a subcomponent with that name already exists. Subcomponent group_name must be unique.')
        subcomponent.parent = self
        # disable all bids derivatives for subcomponents
        # toplevel workflow can connect the two workflows outputnodes together to put a result in their own folder
        # or give a desc name to subworkflow.output[key] which will make the subworkflow save it is a subdirectory
        # of the parent's derivatives folder
        if hasattr(subcomponent, 'outputs') and type(subcomponent.outputs) == dict:
            for k in subcomponent.outputs: subcomponent.outputs[k] = None
        self.subcomponents.append(subcomponent)

    def get_subcomponent(self, group_names):
        """
        Get a nested subcomponent using the nested group names.
        :param group_names: Either list of nested group names or the string returned by
        :func:`CFMMParserArguments.get_group_name_chain`
        :return: The desired subcomponent instance
        """
        if type(group_names) == str:
            group_names = group_names.split(os.sep)
        current_subcomponent = self
        # go down the chain to get desired subcomponent
        for group_name in group_names:
            search_hit = False
            # maybe should still use a dictionary _subcomponents for its hash table
            for subcomponent in current_subcomponent.subcomponents:
                if subcomponent.group_name == group_name:
                    current_subcomponent = subcomponent
                    search_hit = True
                    break
            if not search_hit:
                return None

        return current_subcomponent

    def get_parameter(self, parameter_name, subcomponent_chain=None):
        """
        Return a parameter from this workflow or a subcomponent's parameter.
        :param parameter_name:
        :param subcomponent_chain: Either list of nested group names or the string returned by
        :func:`CFMMParserArguments.get_group_name_chain`
        :return: CFMMFlagValuePair object stored in _parameters
        """
        if subcomponent_chain is None:
            return super().get_parameter(parameter_name)
        else:
            subcomponent = self.get_subcomponent(subcomponent_chain)
        return subcomponent.get_parameter(parameter_name)

    def set_inputnode(self, inputnode):
        self.inputnode = inputnode


    def get_inputnode(self, extra_fields=None):
        """
        Create a nipype IdentityInterface named inputnode for the argument group. The inputnode inputs are the
        argument names (first argument given to :func:`CFMMParserArguments.add_parser_argument`). The inputs
        attributes are set to the CFMMFlagValuePair.user_value. This can be considered the default value for the
        inputnode that can be overridden by an upstream nipype workflow connection.
        :param extra_fields: additional list of (field_name, default_value) tuples to be included in the inputnode.
        :return:inputnode
        """
        if self.inputnode is not None:
            return self.inputnode

        inputnode_fields = self._inputnode_params

        if extra_fields is not None:
            for extra_field in extra_fields:
                parameter_name = extra_field[0]
                inputnode_fields.append(parameter_name)

        inputnode = pe.Node(niu.IdentityInterface(fields=inputnode_fields), name='inputnode')

        # if inputnode isn't connected to an upstream parent workflow, the node should be set by command line parameters
        # if any of the inputnode's inputs are connected upstream, it's the parent's job to either use this object's
        # command line parameters to set its own inputnode or to hide this object's command line parameters from
        # the parser. If the parent hides any of the the object's parameters from the parser, it becomes responsible
        # for performing relevant validate_parameters() checks.
        for parameter_name in self._inputnode_params:
            parameter = self.get_parameter(parameter_name)
            setattr(inputnode.inputs, parameter_name, parameter.user_value)

        if extra_fields is not None:
            for extra_field in extra_fields:
                parameter_name, default_value = extra_field
                setattr(inputnode.inputs, parameter_name, default_value)
        self.inputnode = inputnode
        return inputnode


    def set_workflow(self,workflow):
        self.workflow = workflow

    def get_base_workflow(self):
        """
        Create a nipype workflow with same name as self.pipeline_name and store in self.workflow. If a
        NipypeWorkflowArguments subcomponent exists, set the workflow base_dir using the parameter value.
        If being called inside a subclass' :func:`CFMMWorkflow.get_workflow` after super().get_workflow(), overwrite
        should be True. This ensures that the subclass call to get_base_workflow will overwrite the self.workflow
        stored by super().get_workflow()'s call to get_base_workflow.
        :param overwrite: Overwrite existing self.workflow
        :return: nipype workflow
        """
        if self.workflow is not None:
            return self.workflow

        from workflows.CFMMCommon import NipypeWorkflowArguments
        workflow = pe.Workflow(self.pipeline_name)
        if self.parent is None:
            for subcomponent in self.subcomponents:
                if type(subcomponent) == NipypeWorkflowArguments:
                    workflow.base_dir = subcomponent.get_parameter('base_dir').user_value

        # when subclassing, in subclass.get_workflow() we usually define super.get_workflow() before
        # calling subclass.get_base_workflow(). Since the super sets self.workflow, we want to overwrite it
        # with the subclass workflow
        # if (self.workflow is None) or overwrite:
        #     self.set_workflow(workflow)
        self.workflow = workflow

        return workflow

    def connect_to_child_inputnode(self,node,node_output,child_CFMMWorkflow, input_name):
        if child_CFMMWorkflow.workflow is None:
            child_CFMMWorkflow.get_workflow()
        if child_CFMMWorkflow.inputnode is None:
            logger.error("Child must set CFMMWorkflow.inputnode in order to use this function. The child's"
                         "CFMMWorkflow.get_workflow() should use CFMMWorkflow.create_io_and_workflow() or "
                         "CFMMWorkflow.create_inputnode().")
        if self.workflow is None:
            self.get_base_workflow()

        if input_name in child_CFMMWorkflow._inputnode_params:
            child_CFMMWorkflow.hide_parser_argument(input_name)
            self.workflow.connect(node, node_output, child_CFMMWorkflow.workflow, f'inputnode.{input_name}')


    def connect_inputs(self, inputnode_parameter, child_CFMMWorkflow, child_input_name):
        if self.inputnode is None:
            logger.error("Must set self.inputnode before using this helper function.")
        self.connect_to_child_inputnode(self.inputnode,inputnode_parameter,child_CFMMWorkflow,child_input_name)


    def connect_to_overridden_inputnodes(self, exclude_list=[]):
        """
        Connect this workflow's inputnode to the subcomponent inputnodes that it overrides. See override_parameters
        in :func:`CFMMWorkflow.add_parser_argument`
        :param exclude_list: List of parameter names that should not be connected to their overridden subcomponents
        """
        for parameter_name in self._inputnode_params:
            if (parameter_name not in exclude_list) and (parameter_name in self._param_subordinates.keys()):
                parameter = self.get_parameter(parameter_name)
                for subordinate_parameter_name, subordinate_subcomponent in self._param_subordinates[parameter]:
                    if subordinate_parameter_name in subordinate_subcomponent._inputnode_params:
                        self.connect_inputs(parameter_name, subordinate_subcomponent, subordinate_parameter_name)

    def connect_to_superclass_inputnode(self, superclass_workflow, exclude_list=[]):
        """
        If this workflow is a subclass, connect its input node to the superclass inputnode
        :param superclass_workflow: Nipype workflow obtained from super().get_workflow()
        :param exclude_list: List of parameter names that should not be connected to the superclass inputnode
        """
        workflow = self.get_base_workflow()
        inputnode = self.get_inputnode()

        superclass_inputnode = superclass_workflow.get_node('inputnode')
        # superclass_inputnode.interface._fields
        for field in superclass_inputnode.inputs.trait_get().keys():
            if field not in exclude_list:
                # dont' use CFMMWorkflow.connect_to_child_inputnode() because subclass and superclass share
                # the same _parameters and connect_to_child_inputnode hides the superclass (and therefore subclass)
                # argparse argument.
                workflow.connect(inputnode, field, superclass_workflow, f'inputnode.{field}')

    def get_outputnode(self, extra_fields=None):
        if self.outputnode[self.calling_subclass] is not None:
            return self.outputnode[self.calling_subclass]


        if type(self.outputs) == dict:
            fields = list(self.outputs.keys())
        else:
            fields = self.outputs

        if extra_fields is not None:
            fields = fields+extra_fields
        outputnode = pe.Node(niu.IdentityInterface(fields=fields), name='outputnode')
        self.outputnode = outputnode
        return outputnode

    def set_calling_subclass(self,calling_subclass=None):
        self.calling_subclass = calling_subclass

    def get_io_and_workflow(self, calling_subclass=None,
                            extra_inputnode_fields=[],
                            extra_outputnode_fields=[],
                            connection_exclude_list=[],
                            ):
        """
        Helper function for self.get_inputnode, self.get_base_workflow, self.connect_overridden_inputnode, and
        self.connect_superclass_inputnode.
        :param extra_inputnode_fields: additional list of (field_name, default_value) tuples to be included in the inputnode.
        :param overridden_inputnode_exclude_list: List of parameter names that should not be connected to their overridden subcomponents
        :param get_superclass_workflow: Also return the superclass workflow
        :param superclass_inputnode_exclude_list: List of parameter names that should not be connected to the superclass inputnode
        :return:
        """
        # # is there a way to inspect which subclass called me so the user doesn't need to pass the subclass to the superclass?
        # import inspect
        # for x in inspect.stack():
        #     print(x)
        # stop
        self.set_calling_subclass(calling_subclass)
        inputnode = self.get_inputnode(extra_fields=extra_inputnode_fields)
        workflow = self.get_base_workflow()
        outputnode = self.get_outputnode(extra_fields=extra_outputnode_fields)
        self.connect_to_overridden_inputnodes(exclude_list=connection_exclude_list)
        # if get_superclass_workflow:
        #     pipeline_name = self.pipeline_name
        #     self.pipeline_name = self.__class__.__base__.__name__
        #     superclass_workflow = super(self.__class__, self).get_workflow()
        #     self.pipeline_name = pipeline_name
        #     # this rename needs to be done before the superclass workflow has made any connections in order for it
        #     # to be effective. So this is useless here. Rename the pipeline instead.
        #     # superclass_workflow.name = self.__class__.__base__.__name__
        #     workflow.base_dir = superclass_workflow.base_dir
        #     self.connect_to_superclass_inputnode(superclass_workflow, exclude_list=superclass_inputnode_exclude_list)
        #     return inputnode, outputnode, workflow, superclass_workflow
        return inputnode, outputnode, workflow

    def create_workflow(self, arg_dict=None):
        """
        To be implemented by subclass. Use subcomponents and other nipype components to connect and return
        a nipype workflow.
        """
        raise NotImplementedError('Subclass must define get_workflow function.')

    def get_bids_derivatives_description(self):
        """
        Bare bones derivatives description for derivatives datasinks. Should be redefined by subclass to provide a
        more detailed description.
        """
        # how to automate finding bids version?
        bids_version = '1.1.1'

        dataset_desc = {
            'Name': f'{self.pipeline_name} - {self.pipeline_short_desc}',
            'BIDSVersion': bids_version,
            'PipelineDescription': {
                'Name': self.pipeline_name,
                'Version': self.pipeline_version,
                'CodeURL': 'unknown',
            },
        }
        return dataset_desc

    # do we need to add bids stuff to inputnode???

    # def create_inputnode(self, overwrite=False, extra_fields=[]):
    #     """
    #     Create a nipype IdentityInterface named inputnode for the argument group and store in self.inputnode.
    #     If being called inside a subclass' :func:`CFMMWorkflow.get_workflow` before super().get_workflow(), overwrite
    #     should be False. This ensures that the super().get_workflow()'s call to get_inputnode will not overwrite the
    #     existing self.inputnode.
    #     :param args:
    #     :param overwrite: Overwrite existing self.inputnode
    #     :param extra_fields: additional list of (field_name, default_value) tuples to be included in the inputnode.
    #     :param kwargs:
    #     :return: inputnode
    #     """
    #     bids_fields = []
    #     bids_subcomponent = self.get_subcomponent('BIDS Arguments')
    #     if bids_subcomponent is not None:
    #         for parameter_name, parameter in bids_subcomponent._parameters.items():
    #             if parameter.add_to_inputnode:
    #                 bids_fields.append((parameter_name, parameter.user_value))
    #     extra_fields = extra_fields + bids_fields
    #
    #     inputnode = self.get_inputnode(extra_fields=extra_fields)
    #
    #     # when subclassing, in subclass.get_workflow() we usually define subclass.get_inputnode() before
    #     # calling super().get_workflow() which calls super().get_inputnode(). Since the super sets self.inputnode, we
    #     # want to avoid overwriting the existing subclass self.inputnode
    #     if (self.inputnode is None) or overwrite:
    #         self.set_inputnode(inputnode)
    #     return inputnode

    def get_node_derivatives_datasink(self):
        from nipype_interfaces.DerivativesDatasink import get_node_derivatives_datasink
        from workflows.CFMMCommon import get_node_inputs_to_list

        fields = list(self.outputs.keys())
        # self.outputs (key,value) is (outputnode field name, bids desc )
        # disable any output with a None value for the bids desc
        bids_descs = [x for x in self.outputs.values() if x is not None]

        if len(bids_descs) == 0:
            # if all the derivatives outputs are disabled, then we won't use the derivatives_datasink node
            # but we still return an IdentityInterface to avoid erros if the user connected things to the
            # datasink node (eg. original_bids_file)
            fields = list(get_node_derivatives_datasink().inputs.trait_get().keys())
            fields.remove('function_str')
            return pe.Node(niu.IdentityInterface(fields=fields), name='derivatives_datasink')


        pipeline_name = self.get_toplevel_parent().pipeline_name
        pipeline_dataset_desc = self.get_toplevel_parent().get_bids_derivatives_description()
        pipeline_nested_path = os.path.join(pipeline_name, os.sep.join(self.get_nested_group_names()[:-1]))
        pipeline_output_list = get_node_inputs_to_list()
        pipeline_output_list.inputs.list_length = len(bids_descs)

        derivatives_datasink = get_node_derivatives_datasink('derivatives_datasink')
        derivatives_datasink.inputs.dataset_description_dict = pipeline_dataset_desc
        derivatives_datasink.inputs.pipeline_name = pipeline_nested_path
        derivatives_datasink.inputs.derivatives_description_list = bids_descs

        # derivatives connection
        wf = self.get_base_workflow()
        inputnode = self.get_inputnode()
        outputnode = self.get_outputnode()

        for index in range(len(fields)):
            if bids_descs[index] is not None:
                self.workflow.connect(outputnode, fields[index], pipeline_output_list, f'input{index + 1}')

        wf.connect([
            (pipeline_output_list, derivatives_datasink, [('return_list', 'derivatives_files_list')]),
            (inputnode, derivatives_datasink, [('output_derivatives_dir', 'derivatives_dir')]),
        ])
        return derivatives_datasink