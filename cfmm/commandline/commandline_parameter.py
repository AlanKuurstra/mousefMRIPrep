import configargparse as argparse

from cfmm.commandline.argparse_type_functions import eval_with_handling


class CommandlineParameter:
    """
    Base class for commandline parameters.
    Facilitates getting the user input value from parsed argument dictionary.
    Defines the order of precedence for default values when no user input provided.
    """

    def __init__(self,
                 flagname=None,
                 groupname=None,
                 optional=True,
                 default_provider=None,
                 **add_argument_kwargs,
                 ):
        self.flagname = flagname
        self.groupname = groupname
        self.optional = optional

        # default_provider allows the default value to be provided by another CommandlineParameter instance.
        # self.default_provider overrides the default in self.add_argument_inputs['default']
        self.default_provider = default_provider
        # the keyword arguments to give parser add_argument()
        self.add_argument_kwargs = add_argument_kwargs
        # the return value from parser add_argument()
        self.action = None
        self.user_value = None

    @property
    def added_to_parser(self):
        return not self.action is None

    def add_to_parser(self, parser):
        if not self.added_to_parser:  # and self.add_argument_kwargs is not None:
            flag_dash = '--' if self.optional else ''
            # self.default_provider (CommandlineParameter instance) overrides parser default
            if self.default_provider:
                self.add_argument_kwargs['default'] = argparse.SUPPRESS

            # if type isn't explicitly provided and
            # and the parameter being added isn't an incompatible action (eg. store_true action)
            # and there's no existing default value that could be made invalid if a type was set,
            # then use eval for the type which gives consistency between user defined
            # parameters and parameters from nipype Interface() groups.
            #
            # This means users will have to surround their parameters in double quotes.
            # Eg. a commandline input for a string will be "'my_string'", a commandline input for a float would
            # be "3.14" and a commandline input for a list will be "['str1','str2']"
            if 'type' not in self.add_argument_kwargs.keys() \
                    and 'action' not in self.add_argument_kwargs.keys() \
                    and not ('default' in self.add_argument_kwargs.keys() and self.add_argument_kwargs[
                'default'] == argparse.SUPPRESS):
                if 0:
                    # useful for debugging
                    # if 'default' in self.add_argument_inputs.keys():
                    print(self.flagname)
                    print(self.add_argument_kwargs)
                    print('adding eval type!')
                    print('')
                self.add_argument_kwargs['type'] = eval_with_handling

            if self.groupname not in parser.argument_groups.keys():
                # must give title as a keyword (not positional) argument to be stored in parser.argument_groups
                parser.add_argument_group(title=self.groupname)
            try:
                self.action = parser.argument_groups[self.groupname].add_argument(f'{flag_dash}{self.flagname}',
                                                                                  **self.add_argument_kwargs)
            except Exception as e:
                print(e)

    def populate_user_value(self, parsed_args_dict):
        if self.flagname in parsed_args_dict.keys():
            self.user_value = parsed_args_dict[self.flagname]
        else:
            try:
                self.user_value = self.default_provider.populate_user_value(parsed_args_dict)
            except Exception as e:
                print(e)
        return self.user_value

    def __str__(self):
        return f'({self.flagname},{self.user_value},{"Action present" if self.action else None})'

    def __repr__(self):
        return f'({self.flagname},{self.user_value},{"Action present" if self.action else None})'
