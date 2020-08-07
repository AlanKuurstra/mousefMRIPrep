from nipype.interfaces.base.traits_extension import isdefined, Undefined
import configargparse as argparse
from traits.has_traits import HasTraits
import os
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from workflows.CFMMEnums import ParameterListeningMode

class CFMMFlagValuePair():
    """
    Class storing an argparse action, its command line flag, and the value returned from :func:`ArgumentParser.parse_args`.
    There is optional functionality for indicating if the parameter value should instead be obtained from an argument
    hierarchy.
    """
    def __init__(self, parser_flag, user_value, parser_action, add_to_inputnode=False,
                 superior_parameter=None, subordinate_parameters=None, listening_mode = ParameterListeningMode.IGNORE):
        """
        :param parser_flag: argparse flag for optional arguments and (parser return dictionary) dest for optional and positional arguments
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
        self.add_to_inputnode = add_to_inputnode
        self.superior_parameter = superior_parameter
        self.subordinate_parameters = subordinate_parameters
        self.listening_mode = listening_mode
        self.user_value_already_populated = False

    def set_user_value(self, value, force=False):
        if (not self.user_value_already_populated) or force:
            self.user_value = value
            self.user_value_already_populated = True

    def populate_user_value(self, parsed_args_dict, force_repopulate_one=False, force_repopulate_chain=False):
        if self.listening_mode == ParameterListeningMode.REPLACE_VALUE:
            superior_parameter_name, superior_parameter_owner = self.superior_parameter
            superior_parameter = superior_parameter_owner.get_parameter(superior_parameter_name)
            superior_parameter.populate_user_value(parsed_args_dict,force_repopulate_chain=force_repopulate_chain)
            self.set_user_value(superior_parameter.user_value, force = (force_repopulate_one or force_repopulate_chain))
        elif self.listening_mode == ParameterListeningMode.REPLACE_DEFAULT:
            if self.parser_flag not in parsed_args_dict.keys():
                superior_parameter_name, superior_parameter_owner = self.superior_parameter
                superior_parameter = superior_parameter_owner.get_parameter(superior_parameter_name)
                superior_parameter.populate_user_value(parsed_args_dict, force_repopulate_chain=force_repopulate_chain)
                self.set_user_value(superior_parameter.user_value, force = (force_repopulate_one or force_repopulate_chain))
            else:
                self.set_user_value(parsed_args_dict[self.parser_flag], force = (force_repopulate_one or force_repopulate_chain))
        elif self.listening_mode == ParameterListeningMode.IGNORE:
            self.set_user_value(parsed_args_dict[self.parser_flag], force = (force_repopulate_one or force_repopulate_chain))
        else:
            raise

    def __str__(self):
        return f'({self.parser_flag},{self.user_value},{"Action present" if self.parser_action else None})'

    def __repr__(self):
        return f'({self.parser_flag},{self.user_value},{"Action present" if self.parser_action else None})'


class CFMMParserArguments():
    """
    Base class grouping a number of common argparse arguments.
    """
    def __init__(self, group_name = None, parent=None, parser=None, exclude_list=None, flag_prefix='',
                 flag_suffix=''):
        """
        :param group_name: Name used in parser's argument group. See :func:`CFMMParserArguments.set_parser`.
        :param parent: If the current instance is a subcomponent, parent stores the owner.
        :param parser: The parser instance that arguments will be added to.
        :param exclude_list: List of names of arguments to exclude from the group.
        :param flag_prefix: Prefix to add to flag names to make unique.
        :param flag_suffix: Suffix to add to flag names to make unique.
        """
        if group_name is None:
            group_name = self.__class__.__name__
        self.group_name = group_name
        self.parent = parent

        if type(exclude_list) in (str, type(None)):
            self.exclude_list = [exclude_list]
        elif type(exclude_list) != list:
            self.exclude_list = list(exclude_list)

        self._parameters = {}

        # prefix and suffix facilitate individualising commandline flags
        self.flag_prefix = flag_prefix
        self.flag_suffix = flag_suffix
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
        group_name = self.get_group_name_chain()
        if group_name == '':
            self.parser_group = parser
        else:
            self.parser_group = parser.add_argument_group(group_name)

    def add_parser_argument(self, parameter_name, *args, parser_flag=None, optional=True, add_to_inputnode=True,
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
                self._parameters[parameter_name] = CFMMFlagValuePair(None, None, None,add_to_inputnode=add_to_inputnode)
            # in order to customize the flag name, you can use the parser_flag keyword
            # if parser_flag keyword is not set and an existing flag name is in self._parameters[parameter_name].parser_flag
            # then the existing flag value will be respected
            # otherwise, a default flag will be created
            if parser_flag is None:
                parser_flag = self._parameters[parameter_name].parser_flag
            if parser_flag == None:
                concatenated_flag_prefix, concatenated_flag_suffix = self.get_concatenated_flag_affixes()
                parser_flag = f'{concatenated_flag_prefix}{parameter_name}{concatenated_flag_suffix}'
                self._parameters[parameter_name].parser_flag = parser_flag

            flag_dash=''
            if optional:
                flag_dash='--'

            self._parameters[parameter_name].parser_action = parser_group.add_argument(
                f'{flag_dash}{parser_flag}', *args, **kwargs)

    def modify_parser_argument(self, parameter_name, attribute, value):
        """
        :param parameter_name: Name of parameter to modify (first argument given to :func:`CFMMParserArguments.add_parser_argument` and the keyword in self._parameters).
        :param attribute: argparse action attribute to modify (eg. default, help, etc)
        :param value: value to give argparse action attribute
        """
        if attribute == 'flag':
            # not sure how well this will work, might need to argparse.SUPPRESS existing parameter and create a new one
            setattr(self._parameters[parameter_name].parser_action, 'option_strings', [f'--{value}'])
            setattr(self._parameters[parameter_name].parser_action, 'dest', value)
        else:
            setattr(self._parameters[parameter_name].parser_action, attribute, value)

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
        self._parameters[parameter_name].parser_action.help = argparse.SUPPRESS

    def populate_parameters(self, parsed_args_dict):
        for parameter_name,parameter in self._parameters.items():
            if parameter_name not in self.exclude_list:
                parameter.populate_user_value(parsed_args_dict=parsed_args_dict)

    def get_parameter(self,parameter):
        return self._parameters[parameter]

    def validate_parameters(self):
        """
        To be implemented by subclass. Give warning and errors to the user if commandline options are
        mutually exclusive.
        """
        pass
    def get_inputnode(self, extra_fields=None):
        """
        Create a nipype IdentityInterface named inputnode for the argument group. The inputnode inputs are the
        argument names (first argument given to :func:`CFMMParserArguments.add_parser_argument`). The inputs
        attributes are set to the CFMMFlagValuePair.user_value. This can be considered the default value for the
        inputnode that can be overridden by an upstream nipype workflow connection.
        :param extra_fields: additional list of (field_name, default_value) tuples to be included in the inputnode.
        :return:inputnode
        """
        inputnode_fields = []
        for parameter_name, parameter in self._parameters.items():
            if parameter.add_to_inputnode:
                inputnode_fields.append(parameter_name)

        if extra_fields is not None:
            for extra_field in extra_fields:
                parameter_name = extra_field[0]
                inputnode_fields.append(parameter_name)

        inputnode = pe.Node(niu.IdentityInterface(fields=inputnode_fields), name='inputnode')

        # if inputnode isn't connected to an upstream parent workflow, the node should be set by command line parameters
        # if any of the inputnode's inputs are connected upstream, it's the parent's job to either use this object's
        # command line parameters to set inputnode or to hide this object's command line parameters from the parser.
        # If the parent hides any of the the object's parameters from the parser, it becomes responsible for
        # performing relevant validate_parameters() checks.
        for parameter_name, parameter in self._parameters.items():
            if parameter.add_to_inputnode:
                setattr(inputnode.inputs, parameter_name, parameter.user_value)
        if extra_fields is not None:
            for extra_field in extra_fields:
                parameter_name, default_value = extra_field
                setattr(inputnode.inputs, parameter_name, default_value)
        self.inputnode = inputnode
        return inputnode


class convert_argparse_using_eval():
    def __init__(self, trait_type):
        self.trait_type = trait_type

    def convert(self, argparse_value):
        # this is annoying for the user, but low maitenance
        # since users will mostly use config files, the annoying strings are acceptable
        # we can put in logic for strings outside of lists to behave differently (ie. enum and string traits)
        # if trait is string, don't do eval()
        # if trait is enum, '' converts to None, and the casting should be done by the enum.values ignoring a None enum.value if it exists
        # but then None in enum and None in a list or tuple is input differently.
        try:
            trait_value = eval(argparse_value)
        except Exception as e:
            raise argparse.ArgumentTypeError(
                f'input "{argparse_value}" must be a valid input for python\'s eval(). Did you forget quotes around a string? eg. for a string input use "\'string_1\'" and for a list input use "[\'string_1\',\'string_2\']"')

        class dummy(HasTraits):
            trait_argument = self.trait_type

        # perhaps we need to traverse the trait_type to do appropriate casting?
        try:
            self.trait_type.validate(dummy(), 'trait_argument', trait_value)
        except Exception as e:
            raise argparse.ArgumentTypeError(
                str(e).replace("'trait_argument' trait of a dummy instance", f'input "{str(trait_value)}"'))
        return trait_value


class CFMMInterface(CFMMParserArguments):
    def __init__(self, nipype_interface, *args, **kwargs):
        self.interface = nipype_interface
        #parameter_names = list(nipype_interface().inputs.trait_get().keys())
        #parameter_names.sort()
        super().__init__(*args, **kwargs)

    def convert_trait_to_argument(self, parameter, trait):
        convert_obj = convert_argparse_using_eval(trait.trait_type)
        default = Undefined
        if trait.usedefault:
            default = trait.default
        if type(default) is str:
            # prepare string defaults for eval() inside convert_argparse_using_eval.convert()
            if default == '' or not ((default[0] == default[-1] == "'") or (default[0] == default[-1] == '"')):
                default = '"' + default + '"'
        self.add_parser_argument(parameter,
                                 default=default,
                                 type=convert_obj.convert,
                                 help=trait.desc)

    def add_parser_arguments(self):
        # add parser arguments
        parameter_names = list(self.interface().inputs.trait_get().keys())
        parameter_names.sort()
        trait_dict = self.interface.input_spec().traits()
        for parameter in parameter_names:
            self.convert_trait_to_argument(parameter, trait_dict[parameter])

    def get_interface(self, name='CFMMNode', arg_dict=None):
        if arg_dict is not None:
            self.populate_parameters(arg_dict)
        nipype_interface = self.interface()
        for parameter in self._parameters.keys():
            user_value = self._parameters[parameter].user_value
            setattr(nipype_interface.inputs, parameter, user_value)
        return nipype_interface




class CFMMWorkflow(CFMMParserArguments):
    """
    Class for managing a nipype workflow with commandline arguments.
    """
    def __init__(self, subcomponents, *args, pipeline_name=None, pipeline_version=None,**kwargs):
        for proposed_subcomponent in subcomponents:
            self.add_subcomponent(proposed_subcomponent)
        if pipeline_name is None:
            pipeline_name = self.__class__.__name__
        self.pipeline_name = pipeline_name
        self.pipeline_short_desc = ''

        if pipeline_version is None:
            import inspect
            from git import Repo
            subclass_dir = inspect.getfile(self.__class__)
            subclass_git_repo = Repo(subclass_dir, search_parent_directories=True)
            self.pipeline_version = subclass_git_repo.head.commit.hexsha
        super().__init__(*args, **kwargs)

    def set_parser(self, parser):
        # location of super().set_parser() call determines order of argparser help groups
        # putting super() at the beginning means the nested group arguments are shown below their owners
        super().set_parser(parser)
        for subcomponent in self.subcomponents:
            if not subcomponent.parent == self:
                self.parser = parser
                group_name = self.get_argument_group_name()
                if group_name == '':
                    self.parser_group = parser
                else:
                    self.parser_group = parser.add_argument_group(group_name)
                continue
            subcomponent.set_parser(parser)

    def add_parser_argument(self, parameter_name, *args, override_parameters = None, **kwargs):
        super().add_parser_argument(parameter_name, *args,**kwargs)
        if override_parameters is not None:
            if self._parameters[parameter_name].subordinate_parameters is None:
                self._parameters[parameter_name].subordinate_parameters = []
            for subordinate_parameter_name, subordinate_subcomponent_name in override_parameters:
                subordinate_subcomponent = self.get_subcomponent(subordinate_subcomponent_name)
                subordinate_subcomponent.hide_parser_argument(subordinate_parameter_name)
                subordinate_parameter = subordinate_subcomponent.get_parameter(subordinate_parameter_name)
                self.get_parameter(parameter_name).subordinate_parameters.append((subordinate_parameter_name,subordinate_subcomponent))
                subordinate_parameter.superior_parameter = (parameter_name,self)

    def replace_default_values(self,superior,subordinate):
        for parameter_name in superior._parameters.keys():
            subordinate.modify_parser_argument(parameter_name, 'default', argparse.SUPPRESS)
            subordinate.get_parameter(parameter_name).superior_parameter = (parameter_name, superior)
            subordinate.get_parameter(parameter_name).listening_mode = ParameterListeningMode.REPLACE_DEFAULT



    def add_parser_arguments(self):
        for subcomponent in self.subcomponents:
            if not subcomponent.parent == self:
                continue
            subcomponent.add_parser_arguments()

    def populate_parameters(self, arg_dict):
        for subcomponent in self.subcomponents:
            if not subcomponent.parent == self:
                continue
            subcomponent.populate_parameters(arg_dict)
        super().populate_parameters(arg_dict)

    def validate_parameters(self):
        for subcomponent in self.subcomponents:
            if not subcomponent.parent == self:
                continue
            subcomponent.validate_parameters()

    def add_subcomponent(self,subcomponent):
        if not hasattr(self,'subcomponents'):
            self.subcomponents = []
        proposed_subcomponents_name = subcomponent.group_name
        for existing_subcomponent in self.subcomponents:
            if proposed_subcomponents_name == existing_subcomponent.group_name:
                raise ValueError(f'Cannot add subcomponent with group_name {proposed_subcomponents_name}, a subcomponent with that name already exists. Subcomponent group_name must be unique.')
        subcomponent.parent = self
        self.subcomponents.append(subcomponent)

    def get_subcomponent(self,group_names):
        if type(group_names) == str:
            group_names = group_names.split(os.sep)
        current_subcomponent = self
        # go down the chain to get desired subcomponent
        for group_name in group_names:
            search_hit = False
            for subcomponent in current_subcomponent.subcomponents:
                if subcomponent.group_name == group_name:
                    current_subcomponent = subcomponent
                    search_hit = True
                    break
            if not search_hit:
                return None

        return current_subcomponent

    def get_parameter(self,parameter,subcomponent_chain=None):
        if subcomponent_chain is None:
            return super().get_parameter(parameter)
        else:
            subcomponent = self.get_subcomponent(subcomponent_chain)
        return subcomponent.get_parameter(parameter)

    def replace_subcomponent(self,subcomponent_name, new_subcomponent, retain_control = False):
        # DEPRECATED
        # currently this function must be called BEFORE add_parser_arguments()

        # if we want to allow this function to be called after add_parser_arguments, we must be able to remove
        # existing argparse parameters. https://bugs.python.org/issue19462#msg251739
        # we can't simply suppress the existing argparse parameters because the new parameter will probably want to use
        # the same flag name.
        # although it might be possible to make use of argparse.ArgumentParser(conflict_handler='resolve')

        # use retain_control=True if you want this workflow to be in control of:
        # adding the argparse parameter and deciding the argparse flag name
        # deciding the command line group help name
        # populating the subcomponent with user command line inputs

        # the common use case of replace_subcomponent is to have the parent workflow take control of the subcomponent,
        # but allow a nested child workflow to access the interface/node. In this case you want the child workflow
        # to relinquish control.

        self.subcomponents[subcomponent_name] = new_subcomponent
        if retain_control:
            # assign the subcomponent to yourself (takes it away from component's current parent)
            new_subcomponent.parent = self

    def get_inputnode(self,*args,**kwargs):
        bids_fields=[]
        bids_subcomponent = self.get_subcomponent('BIDS Arguments')
        if bids_subcomponent is not None:
            for parameter_name, parameter in bids_subcomponent._parameters.items():
                if parameter.add_to_inputnode:
                    bids_fields.append((parameter_name,parameter.user_value))
        self.inputnode = super().get_inputnode(extra_fields=bids_fields)
        return self.inputnode


    def get_workflow(self, arg_dict=None):
        # use workflow_parameters to create nodes and return a workflow
        raise NotImplementedError('Subclass must define get_workflow function.')

    def get_base_workflow(self):
        print(self,id(self.inputnode))
        stop
        wf = pe.Workflow(self.pipeline_name)
        for parameter_name, parameter in self._parameters.items():
            if parameter.subordinate_parameters is not None:
                for subordinate_parameter_name,subordinate_component in parameter.subordinate_parameters:
                    subordinate_parameter = subordinate_component.get_parameter(subordinate_parameter_name)
                    if parameter.add_to_inputnode and subordinate_parameter.add_to_inputnode:
                        wf.connect([
                            (self.inputnode,subordinate_component.inputnode,[(parameter_name,subordinate_parameter_name)])
                        ])
        return wf





    def get_bids_derivatives_description(self):
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

