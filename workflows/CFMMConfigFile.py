from workflows.CFMMParameterGroup import CFMMParameterGroup
import os
from workflows.CFMMLogging import NipypeLogger as logger

class CFMMConfig(CFMMParameterGroup):
    group_name = "Config File Options"
    def _add_parameters(self):
        self._add_parameter('config_file',
                                 is_config_file=True,
                                 help='Path to config config file for argument default values. Command line arguments override config file.')

        self._add_parameter('write_config_file',
                            const='config.txt',
                            nargs='?',
                            help='Write a config file storing the current command line arguments and exit.')
    def parse_args(self,parser_groups, *args,**kwargs):
        arg_namespace = parser_groups.parser.parse_args(*args,**kwargs)
        self.populate_parameters(vars(arg_namespace))
        write_config_file = self.get_parameter('write_config_file')
        if write_config_file.user_value:
            config_path = os.path.abspath(write_config_file.user_value)
            logger.info(f'Writing config file to {config_path}.')

            if hasattr(arg_namespace,'write_config_file'):
                delattr(arg_namespace, 'write_config_file')
            if hasattr(arg_namespace,'config_file'):
                delattr(arg_namespace, 'config_file')
            self.parser.write_config_file(arg_namespace,
                                     [config_path],
                                     exit_after=True)
        return arg_namespace