from workflows.CFMMBase import CFMMParserArguments
import os
from workflows.CFMMLogging import NipypeLogger as logger

class CFMMConfig(CFMMParserArguments):
    group_name = "Config File Options"
    def add_parser_arguments(self):
        self.add_parser_argument('config_file',
                                 is_config_file=True,
                                 help='Path to config config file for argument default values. Command line arguments override config file.')

        self.add_parser_argument('write_config_file',
                            const='config.txt',
                            nargs='?',
                            help='Write a config file storing the current command line arguments and exit.')
    def parse_args(self,*args,**kwargs):
        arg_namespace = self.parser.parse_args(*args,**kwargs)
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