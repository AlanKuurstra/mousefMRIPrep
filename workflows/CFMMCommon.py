from workflows.CFMMBase import CFMMParserArguments
from nipype import config
import tempfile
import os
from datetime import datetime
from nipype.pipeline import engine as pe
from nipype.interfaces.utility import Function
import logging
from nipype import logging as nipype_logging

class NipypeLoggerClass():
    def __init__(self):
        self.stdout_handler = logging.getLogger('nipype').handlers[0]
        self.workflow_logger = logging.getLogger('nipype.workflow')
    def info(self,msg):
        prev_level = self.stdout_handler.level
        self.stdout_handler.setLevel('INFO')
        self.workflow_logger.info(msg)
        self.stdout_handler.setLevel(prev_level)
    def warning(self,msg):
        self.workflow_logger.warning(msg)
    def error(self,msg):
        self.workflow_logger.error(msg)

NipypeLogger = NipypeLoggerClass()

# if a connection is made on one of the inputs, but no upstream value is passed along, then the None value is still
# included in the list. If list_length is not provided, can only guess the list length is equal to the the index
# of the last input with a value (we will not know if there were more inputs that had no upstream value and didn't
# pass a value to the function)
def inputs_to_list(
        input1,
        input2=None,
        input3=None,
        input4=None,
        input5=None,
        input6=None,
        input7=None,
        input8=None,
        input9=None,
        input10=None,
        list_length=None,
):
    # get inputs before cluttering the namespace
    locals_copy = locals().copy()

    parameters = list(locals_copy.values())
    # locals dict keys are in reverse order
    parameters.reverse()

    if list_length is None:
        index = 1
        for input in parameters:
            if input is not None:
                list_length = index
            index += 1
    elif list_length == 0:
        return []
    return_list = parameters[:list_length]
    return return_list


def get_node_inputs_to_list(name='inputs_to_list'):
    node = pe.Node(
        Function(input_names=[
            "input1",
            "input2",
            "input3",
            "input4",
            "input5",
            "input6",
            "input7",
            "input8",
            "input9",
            "input10",
            "list_length",
        ],
            output_names=["return_list"],
            function=inputs_to_list), name=name)
    return node


# if a connection is made on one of the inputs, but no upstream value is passed along, then the None value is NOT
# included in the list
def existing_inputs_to_list(input1,
                            input2=None,
                            input3=None,
                            input4=None,
                            input5=None,
                            input6=None,
                            input7=None,
                            input8=None,
                            input9=None,
                            input10=None,
                            ):
    # get inputs before cluttering the namespace
    locals_copy = locals().copy()

    parameters = list(locals_copy.values())
    # locals dict keys are in reverse order
    parameters.reverse()

    return_list=[]
    for input in parameters:
        if input is not None:
            return_list.append(input)

    # # if the order of the locals dictionary changes, then we can always sort it
    # argskwargs = list(locals_copy.items())
    # import re
    # def atoi(text):
    #     return int(text) if text.isdigit() else text
    # def natural_keys(text):
    #     return [atoi(c) for c in re.split(r'(\d+)', text)]
    # return_list = []
    # for _,input in sorted(argskwargs,key=lambda x:natural_keys(x[0])):
    #     if input is not None:
    #         return_list.append(input)
    #     else:
    #         break
    # return return_list
    return return_list


def get_node_existing_inputs_to_list(name='existing_inputs_to_list'):
    node = pe.Node(
        Function(input_names=[
            "input1",
            "input2",
            "input3",
            "input4",
            "input5",
            "input6",
            "input7",
            "input8",
            "input9",
            "input10",
        ],
            output_names=["return_list"],
            function=existing_inputs_to_list), name=name)
    return node


class CFMMConfig(CFMMParserArguments):
    group_name = "Config Options"
    def add_parser_arguments(self):
        parser.add_argument('--config_file',
                            help='Use a config file for argument default values. Command line arguments override config file.')

        parser.add_argument('--write_config_file',
                            const='config.txt',
                            nargs='?',
                            help='Write a config file storing the current command line arguments and exit.')


class NipypeRunArguments(CFMMParserArguments):
    group_name = "Nipype Run Arguments"
    def add_parser_arguments(self):
        self.add_parser_argument('nipype_processing_dir',
                                 help='Directory where intermediate images, logs, and crash files should be stored.')

        self.add_parser_argument('log_dir',
                                 help="Nipype output log directory. Defaults to <nipype_processing_dir>/log")

        self.add_parser_argument('crash_dir',
                                 help="Nipype crash dump directory. Defaults to <nipype_processing_dir>/crash_dump")

        self.add_parser_argument('plugin',
                                 default='Linear',
                                 help="Nipype run plugin")

        self.add_parser_argument('plugin_args',
                                 help="Nipype run plugin arguments")

        self.add_parser_argument('keep_unnecessary_outputs',
                                 action='store_true', default=False,
                                 help="Keep all nipype node outputs, even if unused by downstream nodes.")

    def populate_parameters(self, arg_dict):
        super().populate_parameters(arg_dict)

        # set nipype's stdout handler to ERROR to clean up commandline (leave module loggers' file handlers at
        # default level of INFO)
        nipype_logger = logging.getLogger('nipype')
        nipype_logger.handlers[0].setLevel('ERROR')

        # setup scratch, logging, crash dirs
        nipype_dir = self._parameters['nipype_processing_dir'].user_value
        log_dir = self._parameters['log_dir'].user_value
        crash_dir = self._parameters['crash_dir'].user_value

        if nipype_dir:
            nipype_dir = os.path.abspath(nipype_dir)
        else:
            nipype_dir = tempfile.TemporaryDirectory().name
        arg_dict[self._parameters['nipype_processing_dir'].parser_flag] = nipype_dir
        if log_dir:
            log_dir = os.path.abspath(log_dir)
        else:
            tmp = datetime.now().strftime('nipype_log_%Y_%m_%d_%H_%M.log')

            # if participant label in arg_dict, add to log name
            # if session label, add to log name
            # + "_".join(subject) + '-' + "_".join(func_sessions)
            # tmp = tmp.replace(".*", "all").replace("*", "star")
            log_dir = os.path.join(nipype_dir, 'logs', tmp)
        if crash_dir:
            crash_dir = os.path.abspath(crash_dir)
        else:
            crash_dir = os.path.join(nipype_dir, 'crash_dump')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        config.update_config({'logging': {
            'log_directory': log_dir,
            'log_to_file': True,
        },
            'execution': {
                'crashdump_dir': crash_dir,
                'crashfile_format': 'txt',
            }})
        nipype_logging.update_logging(config)

    def run_workflow(self, wf):
        wf.config['execution']['remove_unnecessary_outputs'] = not self._parameters[
            'keep_unnecessary_outputs'].user_value
        plugin = self._parameters['plugin'].user_value
        plugin_args = self._parameters['plugin_args'].user_value

        if plugin_args:
            execGraph = wf.run(plugin, plugin_args=eval(plugin_args))
        else:
            execGraph = wf.run(plugin)

        return execGraph


class NipypeWorkflowArguments(CFMMParserArguments):
    group_name = "Nipype Workflow Arguments"
    def add_parser_arguments(self):
        # base_dir only has an effect for the toplevel workflow so we want to exclude it if we're not toplevel
        # we can exclude it in __init__ because we don't know if we have parents in the initialization (parents are set
        # afterword. So we set the exclude list during add_parser_arguments.
        #if not self.parent == self.get_toplevel_parent():
        if self.parent is not None:
            if self.parent.parent is not None:
                self.exclude_list.append('base_dir')

        self.add_parser_argument('nthreads_node',
                                 type=int,
                                 default=-1,
                                 help="Number of threads for a single node. The default, -1, is to have as many threads as available cpus.")
        self.add_parser_argument('nthreads_mapnode',
                                 type=int,
                                 default=-1,
                                 help="Number of threads in every node of a mapnode. The default, -1, is to divide the available cpus between the number of running mapnode nodes.")
        self.add_parser_argument('mem_gb_mapnode',
                                 default=10,
                                 type=float,
                                 help="Maximum memory required by a mapnode.")

        pipeline_name = self.get_toplevel_parent().pipeline_name
        self.add_parser_argument('base_dir',
                                 help=f"Nipype base dir for storing intermediate results. Defaults to <nipype_processing_dir>/{pipeline_name}_workdir'",
                                 # inputnode is only used for parameters that are useful when a workflow is a
                                 # subworkflow. If a workflow is a subworkflow, then it's base_dir is ignored so it
                                 # is not useful for inputnode
                                 add_to_inputnode=False)

    def populate_parameters(self, arg_dict):
        super().populate_parameters(arg_dict)
        if 'base_dir' not in self.exclude_list:
            if self._parameters['base_dir'].user_value:
                self._parameters['base_dir'].user_value = os.path.abspath(self._parameters['base_dir'].user_value)
            elif arg_dict['nipype_processing_dir']:
                nipype_dir = os.path.abspath(arg_dict['nipype_processing_dir'])
                pipeline_name = self.get_toplevel_parent().pipeline_name
                self._parameters['base_dir'].user_value = os.path.join(nipype_dir, f'{pipeline_name}_workdir')
            else:
                self._parameters['base_dir'].user_value = tempfile.TemporaryDirectory().name
