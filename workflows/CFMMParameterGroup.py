import configargparse as argparse
import os

class CFMMParserGroups():
    def __init__(self, parser):
        self.parser = parser
        self.parser_groups = {}



class CFMMCommandlineParameter():
    def __init__(self, add_argument_inputs, flagname=None, groupname=None, optional=True, default_provider=None,
                 action=None):
        self.flagname = flagname
        self.groupname = groupname
        self.optional = optional
        # self.default_provider overrides the default in self.add_argument_inputs
        self.default_provider = default_provider
        # we should reinstate this and figure out errors
        # eval necessary to have python None and string "None" input from command line
        # eval allows dictionary inputs from command line
        # eval keeps us consistent with the Nipype node parameter syntax
        # if 'type' not in add_argument_inputs.keys() and 'action' not in add_argument_inputs.keys():
        #     add_argument_inputs['type'] = eval
        self.add_argument_inputs = add_argument_inputs
        self.action = action
        self.user_value = None

    def populate_user_value(self, parsed_args_dict):
        if self.flagname in parsed_args_dict.keys():
            self.user_value = parsed_args_dict[self.flagname]
        else:
            self.user_value = self.default_provider.populate_user_value(parsed_args_dict)
        return self.user_value

    def __str__(self):
        return f'({self.flagname},{self.user_value},{"Action present" if self.action else None})'

    def __repr__(self):
        return f'({self.flagname},{self.user_value},{"Action present" if self.action else None})'


class CFMMParameterGroup():
    """
    Base class for grouping a number of related argparse arguments.

    :ivar groupname: initial value:
    :ivar flag_prefix: initial value:
    :ivar flag_suffix: initial value:
    :ivar owner: initial value:
    :ivar exclude_list: initial value:
    :ivar _parameters: initial value:
    :ivar parser: initial value:
    :ivar parser_group: initial value:

    """

    def __init__(self, groupname=None, owner=None, exclude_list=[], flag_prefix=None,
                 flag_suffix=None, replaced_parameters={}):
        """
        :param groupname: Name used in parser's argument group. See :func:`CFMMParserArguments.set_parser`.
        :param owner: If the current instance is a subcomponent, owner stores the owner.
        :param parser: The parser instance that arguments will be added to.
        :param exclude_list: List of names of arguments to exclude from the group.
        :param flag_prefix: Prefix to add to flag names to make unique.
        :param flag_suffix: Suffix to add to flag names to make unique.
        """
        if groupname is not None:
            self.group_name = groupname
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
            self.flag_prefix = ''

        if flag_suffix is not None:
            self.flag_suffix = flag_suffix
        elif hasattr(self, 'flag_suffix'):
            pass
        else:
            self.flag_suffix = ''

        self.owner = None
        if owner:
            owner.add_subcomponent(self)

        self.exclude_parameters(exclude_list)

        self._parameters = {}
        if replaced_parameters:
            self._parameters = replaced_parameters
        self._add_parameters()

    def add_subcomponent(self, subcomponent):
        pass
    def exclude_parameters(self,exclude_list):
        if not hasattr(self,'exclude_list') or self.exclude_list is None:
            self.exclude_list = []
        self.exclude_list.extend(exclude_list)

    def get_toplevel_owner(self):
        """
        Traverse the composition chain and return first instance without a owner.
        :return: toplevel_owner
        """

        current_component = self
        current_owner = current_component.owner
        while current_owner is not None:
            current_component = current_owner
            current_owner = current_owner.owner
        return current_component

    def get_concatenated_flag_affixes(self):
        """
        Traverse the composition chain and concatenate all flag affixes to obtain a unique prefix and suffix for the flag.
        :return: concatenated_flag_prefix, concatenated_flag_suffix
        """
        current_owner = self.owner
        flag_prefix_list = []
        flag_suffix_list = []
        if self.flag_prefix is not None:
            flag_prefix_list.insert(0, self.flag_prefix)
        if self.flag_suffix is not None:
            flag_suffix_list.insert(0, self.flag_suffix)

        while current_owner is not None:
            if current_owner.flag_prefix is not None:
                flag_prefix_list.insert(0, current_owner.flag_prefix)
            if current_owner.flag_suffix is not None:
                flag_suffix_list.insert(0, current_owner.flag_suffix)
            current_owner = current_owner.owner

        # the [1:] index gets ride of the topmost workflow flag (prefixes are only necessary if you're a subworkflow)
        concatenated_flag_prefix = ''.join(flag_prefix_list[1:])
        concatenated_flag_suffix = ''.join(flag_suffix_list[1:])
        return concatenated_flag_prefix, concatenated_flag_suffix

    def get_nested_groupnames(self):
        """
        Traverse the composition chain and append all groupnames to a list.
        :return: groupnames
        """
        groupnames = []
        current_component = self
        while current_component.owner is not None:
            if current_component.group_name is not None and current_component.group_name != '':
                groupnames.append(current_component.group_name)
            current_component = current_component.owner
        if current_component.group_name is not None and current_component.group_name != '':
            groupnames.append(current_component.group_name)
        groupnames.reverse()
        return groupnames

    def join_nested_groupnames(self, groupname_list):
        return os.sep.join(groupname_list)

    def get_nested_groupname_str(self):
        """
        Traverse the composition chain and append all group_names to a list.
        :return: concatenated_group_name
        """

        return self.join_nested_groupnames(self.get_nested_groupnames())

    def _add_parameter(self, parameter_name, flagname=None, nested_flag_prefixes=None,
                       nested_flag_suffixes=None, groupname=None, nested_groupnames=None, optional=True, **kwargs):
        """
        Helper class for :func:`CFMMParserArguments.add_parser_arguments`.
        :param parameter_name: Name of parameter to be added to argument_group
        :param args: arguments for :func:`ArgumentParser.add_argument`
        :param flagname: Optional flag that will override automated flag name
        :param optional: If true add as keyword commandline argument, if false add as positional commandline argument
        :param add_to_inputnode: If true, will be added as field of the identity interface returned by :func:`CFMMParserArguments.get_inputnode`
        :param kwargs: keyword arguments for :func:`ArgumentParser.add_argument`
        """

        if parameter_name not in self.exclude_list:
            if parameter_name not in self._parameters.keys():
                nested_affixes = self.get_concatenated_flag_affixes()
                if flagname is None:
                    flagname = parameter_name
                if nested_flag_prefixes is None:
                    nested_flag_prefixes = nested_affixes[0]
                if nested_flag_suffixes is None:
                    nested_flag_suffixes = nested_affixes[1]
                full_flagname = f'{nested_flag_prefixes}{flagname}{nested_flag_suffixes}'

                if groupname is None:
                    groupname = self.group_name
                if nested_groupnames is None:
                    nested_groupnames = self.get_nested_groupnames()[:-1]

                if groupname is not None and groupname != '':
                    full_groupname = self.join_nested_groupnames(nested_groupnames + [groupname])
                else:
                    full_groupname = self.join_nested_groupnames(nested_groupnames)

                self._parameters[parameter_name] = CFMMCommandlineParameter(kwargs,
                                                                            flagname=full_flagname,
                                                                            groupname=full_groupname,
                                                                            optional=optional)

    def _modify_parameter(self, parameter_name, attribute, value):
        """
        :param parameter_name: Name of parameter to modify (first argument given to :func:`CFMMParserArguments.add_parser_argument` and the keyword in self._parameters).
        :param attribute: argparse action attribute to modify (eg. default, help, etc)
        :param value: value to give argparse action attribute
        """
        if parameter_name not in self.exclude_list:
            self.get_parameter(parameter_name).add_argument_inputs[attribute] = value

    @classmethod
    def provide_node_defaults(cls,providing_node,receiving_nodes):
        if type(receiving_nodes) != list:
            receiving_nodes = [receiving_nodes]
        for receiving_node in receiving_nodes:
            for parameter_name,parameter in receiving_node._parameters.items():
                parameter.default_provider = providing_node._parameters[parameter_name]

    def _add_parameters(self):
        """
        To be implemented by subclass. Customize commandline arguments to add to the parser and store in
        self._parameters. Use helper function `CFMMParserArguments.add_parser_argument`.
        """
        raise NotImplementedError('Subclass must define _add_parameters function.')

    def get_parameter(self, parameter_name):
        """
        :param parameter_name:
        :return: CFMMFlagValuePair object stored in self._parameters
        """
        return self._parameters[parameter_name]

    def populate_parser_groups(self, cfmm_parser_groups):
        for parameter_name, parameter in self._parameters.items():
            if parameter.action is None and parameter.add_argument_inputs is not None:

                if parameter.groupname not in cfmm_parser_groups.parser_groups.keys():
                    cfmm_parser_groups.parser_groups[
                        parameter.groupname] = cfmm_parser_groups.parser.add_argument_group(
                        parameter.groupname)

                flag_dash = ''
                if parameter.optional:
                    flag_dash = '--'
                if parameter.default_provider:
                    parameter.add_argument_inputs['default'] = argparse.SUPPRESS

                # if type isn't explicitly provided, use eval for consistency between user defined parameters and
                # parameters from nipype nodes. This means users always have to surround their parameters in double
                # quotes.  Eg. a command input for a string will be "'my_string'", a command input for a float would be
                # "3.14" and a command for a list will be "['str1','str2']"
                if 'action' not in parameter.add_argument_inputs.keys() \
                        and 'type' not in parameter.add_argument_inputs.keys()\
                        and not ('default' in parameter.add_argument_inputs.keys() and parameter.add_argument_inputs['default'] == argparse.SUPPRESS):
                    if 0:
                        # useful for debugging
                        #if 'default' in parameter.add_argument_inputs.keys():
                        print(parameter.flagname)
                        print(parameter.add_argument_inputs)
                        print('adding eval type!')
                        print('')
                    #ACTUALLY THIS SHOULD BE A CUSTOM FUNCTION THAT USES EVAL BUT GIVES A NICE USER ERROR IF IT FAILS
                    # REMINDING THEM OF HOW THINGS NEED TO BE PUT ON THE COMMAND LINE. PYTHON OBJECTS INSIDE DOUBLE QUOTES.
                    # STRING: "'STRING'", DOUBLE: "3.14", NONE: "NONE", STRING NONE: "'NONE'", DICTIONARY: "{'ONE':1,'TWO':2}"
                    parameter.add_argument_inputs['type'] = eval

                parameter.action = cfmm_parser_groups.parser_groups[parameter.groupname].add_argument(
                    f'{flag_dash}{parameter.flagname}', **parameter.add_argument_inputs)

    def populate_parameters(self, parsed_args_dict):
        """
        Automatically populate each parameter's user_value.
        :param parsed_args_dict: Dictionary returned by :func:`ArgumentParser.parse_args`
        """
        for parameter_name, parameter in self._parameters.items():
            if parameter_name not in self.exclude_list:
                parameter.populate_user_value(parsed_args_dict=parsed_args_dict)

    def validate_parameters(self):
        """
        To be implemented by subclass. Give warning and errors to the user if commandline options are
        mutually exclusive.
        """
        pass