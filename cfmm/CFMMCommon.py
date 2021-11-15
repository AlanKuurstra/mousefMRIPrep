from cfmm.commandline.parameter_group import ParameterGroup
from nipype import config
import tempfile
import os
from datetime import datetime
from nipype.pipeline import engine as pe
from nipype.interfaces.utility import Function
import logging
from nipype import logging as nipype_logging
from cfmm.logging import NipypeLogger as logger
from cfmm.mapnode import MapNode
from nipype.pipeline.engine import Node
from inspect import signature
from nipype.interfaces.base import Undefined
from multiprocessing import cpu_count

def get_fn_interface(fn, output_names, imports=None):
    input_names = signature(fn).parameters.keys()
    interface = Function(
        input_names=input_names,
        output_names=output_names,
        function=fn,
        imports=imports
    )
    return interface


def get_fn_node(fn, output_names, *args, imports=None, mapnode=False, name=None, **kwargs):
    interface = get_fn_interface(fn, output_names, imports)
    if name is None:
        name = fn.__name__
    if mapnode:
        node = MapNode(*args,
                       interface=interface,
                       name=name,
                       **kwargs
                       )
    else:
        node = Node(*args,
                    interface=interface,
                    name=name,
                    **kwargs
                    )
    return node


def listify(possible_list):
    return [possible_list] if type(possible_list) != list else possible_list

def delistify(input_list, length_0_return = Undefined):
    if len(input_list) == 0:
        # maybe the user wants [] or None instead of Undefined
        return length_0_return
    elif len(input_list) == 1:
        return input_list[0]
    else:
        return input_list

def get_node_delistify(name='delistify'):
    return get_fn_node(delistify,['output'],name=name ,imports=['from nipype.interfaces.base import Undefined'])

# if a connection is made on one of the inputs, but no upstream value is passed along, then the None value is still
# included in the list. If list_length is not provided, can only guess the list length is equal to the the index
# of the last input with a value (we will not know if there were more inputs that had no upstream value and didn't
# pass a value to the function)
def inputs_to_list(
        input1=None,
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
        list_length = 0
        index = 1
        for input in parameters:
            if input is not None:
                list_length = index
            index += 1
    if list_length == 0:
        return []
    return_list = parameters[:list_length]
    return return_list


# performs what a mapnode of inputs_to_list should do
# the problem with making inputs_to_list a mapnode is that you can't
# have a variable number of inputs - once all the inputs are marked as iterators
# nipype expects all the inputs to be present and you can't leave any of the inputs undefined
def zip_inputs_to_list(
        input1=None,
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

    from cfmm.CFMMCommon import listify
    # locals dict keys are in reverse order, and might not be a list
    parameters_reversed=[]
    for param in reversed(parameters):
        param = listify(param) if param is not None else param
        parameters_reversed.append(param)
    parameters = parameters_reversed

    if list_length is None:
        list_length = 0
        index = 1
        for input in parameters:
            if input is not None:
                list_length = index
            index += 1
    if list_length == 0:
        return []

    # how many elements in a single input? (get number of iterations for mapnode)

    input_length=0
    for input in parameters[:list_length]:
        if input is not None and len(input)>input_length:
            input_length = len(input)

    for index in range(list_length):
        if parameters[index] is None:
            parameters[index] = [None]*input_length
        assert len(parameters[index])==input_length

    return_list = [x for x in zip(*parameters[:list_length])]
    return return_list


def get_node_inputs_to_list(name='inputs_to_list', mapnode=False):
    interface = Function(input_names=[
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
        function=inputs_to_list
    )
    if mapnode:
        # node = CFMMMapNode(interface, name=name,
        #                    iterfield=["input1",
        #                               "input2",
        #                               "input3",
        #                               "input4",
        #                               "input5",
        #                               "input6",
        #                               "input7",
        #                               "input8",
        #                               "input9",
        #                               "input10",
        #                               "list_length", ])
        node = pe.Node(Function(input_names=[
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
        function=zip_inputs_to_list
    ), name=name)
    else:
        node = pe.Node(interface, name=name)
    return node


# if a connection is made on one of the inputs, but no upstream value is passed along, then the None value is NOT
# included in the list
def existing_inputs_to_list(input1=None,
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

    return_list = []
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





class NipypeRunEngine(ParameterGroup):
    group_name = "Nipype Run Arguments"

    def _add_parameters(self):
        self._add_parameter('nipype_processing_dir',
                            help='Directory where intermediate images, logs, and crash files should be stored.')

        self._add_parameter('base_dir',
                            help=f"Nipype base dir for storing intermediate results. Defaults to <nipype_processing_dir>/<pipeline_name>_workdir",
                            )

        self._add_parameter('log_dir',
                            help="Nipype output log directory. Defaults to <nipype_processing_dir>/log")

        self._add_parameter('crash_dir',
                            help="Nipype crash dump directory. Defaults to <nipype_processing_dir>/crash_dump")

        self._add_parameter('plugin',
                            default="'Linear'",
                            help="Nipype run plugin")

        self._add_parameter('plugin_args',
                            help="Nipype run plugin arguments")

        self._add_parameter('keep_unnecessary_outputs',
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
        arg_dict[self._parameters['nipype_processing_dir'].flagname] = nipype_dir
        self.nipype_dir = nipype_dir

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
                # iterables filenames become too long with parametrized dirs
                'parameterize_dirs': False,
            }})

        nipype_logging.update_logging(config)

    def run_workflow(self, wf):
        if wf is None:
            logger.error("The workflow does not exist. A common reason is that the programmer forgot the return statement in "
                         "the pipeline's create_workflow() function.")
        wf.config['execution']['remove_unnecessary_outputs'] = not self._parameters['keep_unnecessary_outputs'].user_value
        plugin = self._parameters['plugin'].user_value
        plugin_args = self._parameters['plugin_args'].user_value

        # base_dir only has an effect for the toplevel workflow, so there's no need to allow subworkflows to
        # have their own base_dir value
        base_dir = self._parameters['base_dir'].user_value
        if base_dir:
            base_dir = os.path.abspath(base_dir)
        else:
            pipeline_name = wf.name
            base_dir = os.path.join(self.nipype_dir, f'{pipeline_name}_workdir')
        wf.base_dir = base_dir
        wf.write_graph(dotfilename=os.path.join(self.nipype_dir, f'{pipeline_name}_graph'), graph2use='flat')

        logger.info(f'Starting pipeline {wf.name} in {base_dir}')


        if plugin_args:
            execGraph = wf.run(plugin, plugin_args=plugin_args)
        else:
            execGraph = wf.run(plugin)
        logger.info(f'Finished pipeline {wf.name}.')

        return execGraph


#
# class NipypeWorkflowArguments(ParameterGroup):
#     group_name = "Nipype Workflow Arguments"
#     flag_prefix = "nipype_"
#
#     def _add_parameters(self):
#         # base_dir only has an effect for the toplevel workflow so we want to exclude it if we're not toplevel
#         # we can exclude it in __init__ because we don't know if we have owners in the initialization (owners are set
#         # afterword. So we set the exclude list during add_parser_arguments.
#         # if not self.owner == self.get_toplevel_owner():
#         if self.owner is not None:
#             if self.owner.owner is not None:
#                 self.exclude_list.append('base_dir')
#
#         self._add_parameter('nthreads_node',
#                             type=int,
#                             default=-1,
#                             help="Number of threads for a single node. The default, -1, is to have as many threads as available cpus.")
#         self._add_parameter('nthreads_mapnode',
#                             type=int,
#                             default=-1,
#                             help="Number of threads in every node of a mapnode. The default, -1, is to divide the available cpus between the number of running mapnode nodes.")
#         self._add_parameter('mem_gb_mapnode',
#                             default=10,
#                             type=float,
#                             help="Maximum memory required by a mapnode.")
#
#         pipeline_name = self.get_toplevel_owner().pipeline_name
#         self._add_parameter('base_dir',
#                             help=f"Nipype base dir for storing intermediate results. Defaults to <nipype_processing_dir>/{pipeline_name}_workdir'",
#                             )

def int_neg_gives_max_cpu(value):
    value = int(value)
    return_value = value
    if value is None or value < 1:
        return_value = cpu_count()
    return return_value


class NipypeWorkflowArguments(ParameterGroup):
    group_name = "Nipype Workflow Arguments"
    flag_prefix = "nipype_"

    def _add_parameters(self):
        # https://nipype.readthedocs.io/en/0.12.1/users/resource_sched_profiler.html
        self._add_parameter('nthreads_node',
                            type=int_neg_gives_max_cpu,
                            default=-1,
                            help=f"Number of threads required for a single node. When nipype is running nodes in "
                                 f"parallel, any node in '{self.owner.group_name}' getting their resource estimation "
                                 f"from this parameter will only be started if it can be supported by the currently "
                                 f"available unused resources. The default, -1, is to start the node and use as many "
                                 f"threads as available.")
        self._add_parameter('nthreads_mapnode',
                            type=int_neg_gives_max_cpu,
                            default=-1,
                            help=f"Number of threads required in every node of a mapnode. When nipype is running nodes "
                                 f"in parallel, any mapnode in '{self.owner.group_name}' getting their resource "
                                 f"estimation from this parameter will only be able to start one of its child nodes if "
                                 f"it can be supported by the currently available unused resources. The default, -1, "
                                 f"is to run all child nodes and divide the available threads between them.")
        self._add_parameter('mem_gb_mapnode',
                            default=3,
                            type=float,
                            help=f"The amount of memory required by every node of a mapnode. When nipype is running "
                                 f"nodes in parallel, any mapnode in '{self.owner.group_name}' getting their resource "
                                 f"estimation from this parameter will only be able to start one of its child nodes "
                                 f"if it can be supported by the currently available unused resources. The default "
                                 f"is 10Gb.")

        self._add_parameter('gzip_large_images',
                            action='store_true',
                            help=f"If true, any node in '{self.owner.group_name}' getting their extension from this "
                                 f"parameter will gzip the output. Gzip saves space but I/O operations take longer.")

