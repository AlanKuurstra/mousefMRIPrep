import os
from functools import lru_cache

from cfmm.commandline.commandline_parameter import CommandlineParameter


class ParameterGroup:
    """
    Base class for hierarchical groups of CommandlineParameters.
    Facilitates adding a group of parameters to an ArgumentParser() for commandline representation.
    Hierarchical concatenation of flag_prefix and flag_suffix help keep parameter names unique between groups.
    Hierarchical concatenation of group_name provides a heading for the argparse help sections.
    """

    def __init__(self,
                 group_name=None,
                 owner=None,
                 flag_prefix=None,
                 flag_suffix=None,
                 exclude_list=None,
                 replaced_parameters=None):
        """
        :param group_name: Name used in parser's argument group. See :func:`CFMMParserArguments.set_parser`.
        :param owner: If the current instance is a subcomponent, owner stores the owner.
        :param parser: The parser instance that arguments will be added to.
        :param exclude_list: List of names of arguments to exclude from the group. (perhaps defined manually)
        :param flag_prefix: Prefix to add to flag names to make unique.
        :param flag_suffix: Suffix to add to flag names to make unique.
        """
        if group_name is not None:
            self.group_name = group_name
        # hasattr accommodates subclasses which make group_name a class variable
        elif hasattr(self, 'group_name'):
            pass
        else:
            self.group_name = self.__class__.__name__

        # prefix and suffix facilitate differentiating each group's flags
        # eg. differentiate node parameters when node is used more than once
        if flag_prefix is not None:
            self.flag_prefix = flag_prefix
        # hasattr accommodates BIDS mixin
        elif hasattr(self, 'flag_prefix'):
            pass
        else:
            self.flag_prefix = ''

        if flag_suffix is not None:
            self.flag_suffix = flag_suffix
        # hasattr accommodates BIDS mixin
        elif hasattr(self, 'flag_suffix'):
            pass
        else:
            self.flag_suffix = ''

        self.owner = None
        if owner:
            owner.add_subcomponent(self)

        self.exclude_list = exclude_list if exclude_list is not None else []
        self._parameters = replaced_parameters if replaced_parameters is not None else {}
        self._add_parameters()

    @lru_cache(maxsize=1)
    def get_toplevel_owner(self):
        """
        Traverse the hierarchy and return first instance without a owner.
        :return: toplevel_owner
        """

        current_component = self
        current_owner = current_component.owner
        while current_owner is not None:
            current_component = current_owner
            current_owner = current_owner.owner
        return current_component

    def get_nested_groupnames(self):
        """
        Traverse the hierarchy and append all group_names to a list.
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

    @lru_cache(maxsize=1)
    def get_nested_groupname_str(self):
        """
        Traverse the hierarchy and generate a concatenated string of group_names.
        :return: concatenated_group_name
        """

        return self.join_nested_groupnames(self.get_nested_groupnames())

    @lru_cache(maxsize=1)
    def get_concatenated_flag_affixes(self):
        """
        Traverse the hierarchy and concatenate all group flag affixes to obtain unique flag
        affixes for all parameters in this group. Solves parameter naming conflicts between groups.
        :return: concatenated_flag_prefix, concatenated_flag_suffix
        """
        current_owner = self.owner
        # what's the difference between a prefix of None vs ''
        flag_prefix_list = [self.flag_prefix] if self.flag_prefix is not None else []
        flag_suffix_list = [self.flag_suffix] if self.flag_suffix is not None else []
        while current_owner is not None:
            if current_owner.flag_prefix is not None:
                flag_prefix_list.append(current_owner.flag_prefix)
            if current_owner.flag_suffix is not None:
                flag_suffix_list.append(current_owner.flag_suffix)
            current_owner = current_owner.owner
        # the [-2::-1] removes the topmost group flag, which is the program name
        # it doesn't need to be added to every subgroup
        concatenated_flag_prefix = ''.join(flag_prefix_list[-2::-1])
        concatenated_flag_suffix = ''.join(flag_suffix_list[-2::-1])
        return concatenated_flag_prefix, concatenated_flag_suffix

    @property
    def concatenated_flag_affixes(self):
        return self.get_concatenated_flag_affixes()

    @property
    def nested_groupname(self):
        return self.get_nested_groupname_str()

    def _add_parameter(self,
                       parameter_name,
                       optional=True,
                       **kwargs):
        """
        Helper class for :func:`ParameterGroup._add_parameters`.
        :param parameter_name: Name of parameter to be added to argument_group
        :param optional: If true add as keyword commandline argument, if false add as positional commandline argument
        :param kwargs: keyword arguments for :func:`ArgumentParser.add_argument`
        """
        if parameter_name not in self.exclude_list:
            if parameter_name not in self._parameters.keys():
                # flag_prefix, flag_suffix = self.get_concatenated_flag_affixes()
                flag_prefix, flag_suffix = self.concatenated_flag_affixes
                full_flagname = f'{flag_prefix}{parameter_name}{flag_suffix}'
                # full_groupname = self.get_nested_groupname_str()
                full_groupname = self.nested_groupname
                self._parameters[parameter_name] = CommandlineParameter(flagname=full_flagname,
                                                                        groupname=full_groupname,
                                                                        optional=optional,
                                                                        **kwargs,
                                                                        )

    def _add_parameters(self):
        """
        To be implemented by subclass. Customize commandline arguments to add to the parser and store in
        self._parameters. Use helper function `CFMMParserArguments.add_parser_argument`.
        """
        raise NotImplementedError('Subclass must define _add_parameters function.')

    def populate_parser(self, parser):
        """
        Each ParameterGroup in the hierarchy uses this function to add its _parameters to the same parser.

        :param parser:
        :return:
        """
        for parameter in self._parameters.values():
            parameter.add_to_parser(parser)

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

    def exclude_parameters(self, exclude_list):
        self.exclude_list.extend(exclude_list)

    def _modify_parameter(self, parameter_name, attribute, value):
        """
        Modify a parameter attribute.
        (eg. the default value or help description of a parameter automatically added by a nipype interface)
        :param parameter_name: Name of parameter to modify (first argument given to :func:`CFMMParserArguments.add_parser_argument` and the keyword in self._parameters).
        :param attribute: argparse action attribute to modify (eg. default, help, etc)
        :param value: value to give argparse action attribute
        """
        if parameter_name not in self.exclude_list:
            self.get_parameter(parameter_name).add_argument_kwargs[attribute] = value

    @classmethod
    def copy_node_defaults(cls, providing_node, receiving_nodes):
        if type(receiving_nodes) != list:
            receiving_nodes = [receiving_nodes]
        for receiving_node in receiving_nodes:
            for parameter_name, parameter in receiving_node._parameters.items():
                parameter.default_provider = providing_node._parameters[parameter_name]

    def get_parameter(self, parameter_name):
        """
        :param parameter_name:
        :return: CFMMFlagValuePair object stored in self._parameters
        """
        return self._parameters[parameter_name]

    def add_subcomponent(self, subcomponent):
        pass
