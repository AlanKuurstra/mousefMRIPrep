import argparse
from configargparse import *
from configargparse import _ENV_VAR_SOURCE_KEY, _COMMAND_LINE_SOURCE_KEY, _CONFIG_FILE_SOURCE_KEY, _DEFAULTS_SOURCE_KEY
from workflows.CFMMBase import convert_argparse_using_eval
import inspect

# NO INVESTIGATION HAS BEEN DONE WITH _ENV_VAR_SOURCE_KEY

def fix_values_for_eval(value):
    if type(value) == list:
        # fix for configarparse

        # if this is not used, a list of strings ['str1','str2'] will be stored as the string [str1,str2] under the
        # assumption that the string will be read from the config file and once again stored as the list of strings
        # ['mm','mm'].  However, we are tweaking lists from actions belonging to nipype nodes - and if we encounter
        # a list from a nipype action, we read it in as a string of a list (ie. "[mm,mm]"). But this is incorrect and
        # will fail when it's passed to the action.type conversion which uses the eval().
        # We need to read it in as the string "['mm','mm']", where the string quotes of mm have been left.
        # For this reason we can't use the list to string conversion provdied by configargparse.  Instead we convert
        # the list to a string ourselves so we can ensure it will be valid for eval() and then
        # configargparse treats it as a string value (ie. it does nothing) rather than a list (ie. incorrectly converts)
        value = repr(value)
    elif type(value) == str:
        # fix for eval() conversion
        value = f"'{value}'"
    return value

def serialize_convert_argparse_using_eval(action,value):
    # The only way to reliably use "config files" is to pickle the parsed namespace,
    # But pickling is not human readable. The user would be saving command line options as
    # defaults, but would not be able to easily modify the config file after it's saved.
    # To change the config file, the user would have to enter all the same command line
    # arguments while modifying the specific ones they want to change and and then use
    # the --write_config_file to pickle the parsed namespace again.

    conversion_function = action.type
    if inspect.ismethod(conversion_function):
        for cls in inspect.getmro(conversion_function.__self__.__class__):
            if cls == convert_argparse_using_eval:
                value = fix_values_for_eval(value)
    elif conversion_function==eval:
        value = fix_values_for_eval(value)
    return value

class CFMMArgumentParser(ArgumentParser):
    def get_items_for_config_file_output(self, source_to_settings,
                                         parsed_namespace):
        """Converts the given settings back to a dictionary that can be passed
        to ConfigFormatParser.serialize(..).

        Args:
            source_to_settings: the dictionary described in parse_known_args()
            parsed_namespace: namespace object created within parse_known_args()
        Returns:
            an OrderedDict where keys are strings and values are either strings
            or lists
        """
        config_file_items = OrderedDict()
        for source, settings in source_to_settings.items():
            if source == _COMMAND_LINE_SOURCE_KEY:
                _, existing_command_line_args = settings['']
                for action in self._actions:
                    config_file_keys = self.get_possible_config_keys(action)
                    if config_file_keys and not action.is_positional_arg and \
                        already_on_command_line(existing_command_line_args,
                                                action.option_strings,
                                                self.prefix_chars):
                        value = getattr(parsed_namespace, action.dest, None)
                        if value is not None:
                            if isinstance(value, bool):
                                value = str(value).lower()
                            # ****************************************************************************
                            # AK: ensure nipype lists and strings are saved properly
                            value = serialize_convert_argparse_using_eval(action,value)
                            # ****************************************************************************
                            config_file_items[config_file_keys[0]] = value

            elif source == _ENV_VAR_SOURCE_KEY:
                for key, (action, value) in settings.items():
                    config_file_keys = self.get_possible_config_keys(action)
                    if config_file_keys:
                        value = getattr(parsed_namespace, action.dest, None)
                        if value is not None:
                            config_file_items[config_file_keys[0]] = value
            elif source.startswith(_CONFIG_FILE_SOURCE_KEY):
                # ****************************************************************************
                # AK: all values from the config file are strings (they are not converted through action.type yet)
                # do not put through serialize_convert_argparse_using_eval, or all arguments will receive unnecessary
                # string ""
                # ****************************************************************************
                for key, (action, value) in settings.items():
                    config_file_items[key] = value
            elif source == _DEFAULTS_SOURCE_KEY:
                for key, (action, value) in settings.items():
                    config_file_keys = self.get_possible_config_keys(action)
                    if config_file_keys:
                        value = getattr(parsed_namespace, action.dest, None)
                        if value is not None:
                            # ****************************************************************************
                            # AK: ensure nipype lists and strings are saved properly
                            value = serialize_convert_argparse_using_eval(action,value)
                            # ****************************************************************************
                            config_file_items[config_file_keys[0]] = value
        return config_file_items

    def parse_known_args(self, args = None, namespace = None,
                         config_file_contents = None, env_vars = os.environ):
        """Supports all the same args as the ArgumentParser.parse_args(..),
        as well as the following additional args.

        Additional Args:
            args: a list of args as in argparse, or a string (eg. "-x -y bla")
            config_file_contents: String. Used for testing.
            env_vars: Dictionary. Used for testing.
        """
        if args is None:
            args = sys.argv[1:]
        elif isinstance(args, str):
            args = args.split()
        else:
            args = list(args)

        for a in self._actions:
            a.is_positional_arg = not a.option_strings

        # maps a string describing the source (eg. env var) to a settings dict
        # to keep track of where values came from (used by print_values()).
        # The settings dicts for env vars and config files will then map
        # the config key to an (argparse Action obj, string value) 2-tuple.
        self._source_to_settings = OrderedDict()
        if args:
            a_v_pair = (None, list(args))  # copy args list to isolate changes
            self._source_to_settings[_COMMAND_LINE_SOURCE_KEY] = {'': a_v_pair}

        # handle auto_env_var_prefix __init__ arg by setting a.env_var as needed
        if self._auto_env_var_prefix is not None:
            for a in self._actions:
                config_file_keys = self.get_possible_config_keys(a)
                if config_file_keys and not (a.env_var or a.is_positional_arg
                    or a.is_config_file_arg or a.is_write_out_config_file_arg or
                    isinstance(a, argparse._VersionAction) or
                    isinstance(a, argparse._HelpAction)):
                    stripped_config_file_key = config_file_keys[0].strip(
                        self.prefix_chars)
                    a.env_var = (self._auto_env_var_prefix +
                                 stripped_config_file_key).replace('-', '_').upper()

        # add env var settings to the commandline that aren't there already
        env_var_args = []
        nargs = False
        actions_with_env_var_values = [a for a in self._actions
            if not a.is_positional_arg and a.env_var and a.env_var in env_vars
                and not already_on_command_line(args, a.option_strings, self.prefix_chars)]
        for action in actions_with_env_var_values:
            key = action.env_var
            value = env_vars[key]
            # Make list-string into list.
            if action.nargs or isinstance(action, argparse._AppendAction):
                nargs = True
                element_capture = re.match(r'\[(.*)\]', value)
                if element_capture:
                    value = [val.strip() for val in element_capture.group(1).split(',') if val.strip()]
            env_var_args += self.convert_item_to_command_line_arg(
                action, key, value)

        if nargs:
            args = args + env_var_args
        else:
            args = env_var_args + args

        if env_var_args:
            self._source_to_settings[_ENV_VAR_SOURCE_KEY] = OrderedDict(
                [(a.env_var, (a, env_vars[a.env_var]))
                    for a in actions_with_env_var_values])

        # before parsing any config files, check if -h was specified.
        supports_help_arg = any(
            a for a in self._actions if isinstance(a, argparse._HelpAction))
        skip_config_file_parsing = supports_help_arg and (
            "-h" in args or "--help" in args)

        # prepare for reading config file(s)
        known_config_keys = {config_key: action for action in self._actions
            for config_key in self.get_possible_config_keys(action)}

        # open the config file(s)
        config_streams = []
        if config_file_contents is not None:
            stream = StringIO(config_file_contents)
            stream.name = "method arg"
            config_streams = [stream]
        elif not skip_config_file_parsing:
            config_streams = self._open_config_files(args)

        # parse each config file
        for stream in reversed(config_streams):
            try:
                config_items = self._config_file_parser.parse(stream)
            except ConfigFileParserException as e:
                self.error(e)
            finally:
                if hasattr(stream, "close"):
                    stream.close()

            # add each config item to the commandline unless it's there already
            config_args = []
            nargs = False
            for key, value in config_items.items():
                if key in known_config_keys:
                    action = known_config_keys[key]
                    discard_this_key = already_on_command_line(
                        args, action.option_strings, self.prefix_chars)
                else:
                    action = None
                    discard_this_key = self._ignore_unknown_config_file_keys or \
                        already_on_command_line(
                            args,
                            [self.get_command_line_key_for_unknown_config_file_setting(key)],
                            self.prefix_chars)

                if not discard_this_key:
                    # *******************************************************************************************
                    # AK: could probably do this in convert_item_to_command_line_arg function,
                    # but I'm not sure yet if it will be necessary with env var settings env_var_args
                    conversion_function = action.type
                    if inspect.ismethod(conversion_function):
                        for cls in inspect.getmro(conversion_function.__self__.__class__):
                            if cls == convert_argparse_using_eval:
                                if type(value) == list:
                                    value = '['+','.join(value)+']'
                    # *******************************************************************************************

                    config_args += self.convert_item_to_command_line_arg(
                        action, key, value)
                    source_key = "%s|%s" %(_CONFIG_FILE_SOURCE_KEY, stream.name)
                    if source_key not in self._source_to_settings:
                        self._source_to_settings[source_key] = OrderedDict()
                    self._source_to_settings[source_key][key] = (action, value)
                    if (action and action.nargs or
                        isinstance(action, argparse._AppendAction)):
                        nargs = True

            if nargs:
                args = args + config_args
            else:
                args = config_args + args

        # save default settings for use by print_values()
        default_settings = OrderedDict()
        for action in self._actions:
            cares_about_default_value = (not action.is_positional_arg or
                action.nargs in [OPTIONAL, ZERO_OR_MORE])
            if (already_on_command_line(args, action.option_strings, self.prefix_chars) or
                    not cares_about_default_value or
                    action.default is None or
                    action.default == SUPPRESS or
                    isinstance(action, ACTION_TYPES_THAT_DONT_NEED_A_VALUE)):
                continue
            else:
                if action.option_strings:
                    key = action.option_strings[-1]
                else:
                    key = action.dest
                default_settings[key] = (action, str(action.default))

        if default_settings:
            self._source_to_settings[_DEFAULTS_SOURCE_KEY] = default_settings

        # parse all args (including commandline, config file, and env var)
        namespace, unknown_args = argparse.ArgumentParser.parse_known_args(
            self, args=args, namespace=namespace)
        # handle any args that have is_write_out_config_file_arg set to true
        # check if the user specified this arg on the commandline
        output_file_paths = [getattr(namespace, a.dest, None) for a in self._actions
                             if getattr(a, "is_write_out_config_file_arg", False)]
        output_file_paths = [a for a in output_file_paths if a is not None]
        self.write_config_file(namespace, output_file_paths, exit_after=True)
        return namespace, unknown_args

