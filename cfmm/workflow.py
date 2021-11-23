"""
Workflow Goals:
workflows contain an input node and output node
input node fields of an existing workflow can be easily converted to use bids
input node fields can be made iterable to take advantage of nipype multiprocessing
consider synchronized (eg. file and mask) interations vs permutations (best parameter search)
input node fields can be easily connected into other workflows' outputnode
workflows can be used stand alone or as a subworkflow in larger workflow
"""
import os
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from cfmm.logging import NipypeLogger as logger
from cfmm.configfile import Config
from cfmm.CFMMCommon import NipypeRunEngine
from cfmm.commandline.parameter_group import HierarchicalParameterGroup
#from cfmm.parameter_group import ParserGroups
from cfmm.commandline.argument_parser import ArgumentParser

class InputnodeField():
    def __init__(self,field_name,default_value=None,iterable=False,default_value_from_commandline=False):
        # default_value only works if default_value_from_commandline is false
        # it can be used to set the default value of inputnode fields that are not directly from a command line parameter
        # default_value is what will be set external to the pipeline on the inputnode (it's the default value that can
        # be overridden by a pipeline connection)
        # it's a misnomer in the case of iterables - because iterables will override a pipeline connection.
        # in a niterable it is a permanent value.
        self.field_name=field_name
        self.default_value = default_value
        self.iterable = iterable
        self.default_value_from_commandline = default_value_from_commandline

class Workflow(HierarchicalParameterGroup):
    """
    Class for managing a nipype workflow with commandline arguments. Manages a list, self.subcomponents, of
    ParameterGroup subclasses (eg. a Workflow used as a subworkflow or an Interface) which will have their
    commandline arguments displayed in this workflow's help.
    """

    def __init__(self, *args, pipeline_name=None, pipeline_version=None, **kwargs):
        """
        :param subcomponents: List of CFMMParserArguments subclasses
        :param args:
        :param pipeline_name:
        :param pipeline_version:
        :param kwargs:
        """

        self.pipeline_name = self.__class__.__name__ if pipeline_name is None else pipeline_name
        self.pipeline_short_desc = ''

        if not hasattr(self, 'extra_inputnode_fields'):
            self.extra_inputnode_fields = []
        if not hasattr(self, 'outputs'):
            self.outputs = []

        self._inputnode_fields = {}
        self.inputnode = None
        self.outputnode = None
        self.workflow = None

        if pipeline_version == None:
            pipeline_version = '1.0'
        self.pipeline_version = pipeline_version
        super().__init__(*args, **kwargs)

    # rename parameter subgroups to workflow subcomponents
    @property
    def subcomponents(self):
        return self.subgroups.values()
    def get_subcomponent(self, name):
        return self.get_subgroup(name)
    def remove_subcomponent(self, subcomponent):
        return self.remove_subgroup(subcomponent)
    def _remove_subcomponent_attribute(self, attribute_name):
        if not hasattr(self,attribute_name):
            logger.error(f"attribute '{attribute_name}' does not exist in any of the following classes: "
                         f"{[c.__name__ for c in self.__class__.__bases__]}.")
        self.remove_subcomponent(getattr(self, attribute_name))


    def _add_parameter(self, parameter_name, *args, add_to_inputnode=True, iterable=False,**kwargs):
        """
        Helper function for :func:`ParameterGroup._add_parameter`. Allows a workflow parameter to hide
        and override parameters from its subcomponents.
        :param parameter_name: Name of parameter to be added to argument_group
        :param args:
        :param override_parameters: Subcomponent parameters to be overridden by current parameter.
        :param kwargs:
        """
        super()._add_parameter(parameter_name, *args, **kwargs)
        # the init exclude_list stops a parameter from showing up on the commandline, but it should not stop the
        # parameter from being added to the inputnode - the reason is because the inputnode field is probably being
        # used in the workflow. Also note that a parameter can only be an iterable if it is on the commandline.
        # An upstream workflow can take over an iterable parameter's values by adding it to the exclude_list (removing
        # it from the commadline and disabling the iteration) and then hooking it's own value into the inputnode
        if add_to_inputnode:
            if parameter_name in self.exclude_list:
                self._inputnode_fields[parameter_name] = InputnodeField(parameter_name)
            else:
                self._inputnode_fields[parameter_name] = InputnodeField(parameter_name,
                                                             default_value_from_commandline=True,
                                                             iterable=iterable)


    def add_inputnode_field(self, field_name, default_value = None, iterable = False):
        #this function should be used before self.get_inputnode()
        self._inputnode_fields[field_name]=(InputnodeField(field_name,
                                                           default_value=default_value,
                                                           iterable=iterable))

    def get_inputnode_field(self,field_name):
        return self._inputnode_fields[field_name]


    def get_inputnode_iterables_index(self, inputnode, field_name):
        for index in range(len(inputnode.iterables)):
            if inputnode.iterables[index][0] == field_name:
                return index
        return None

    def set_inputnode_iterable(self,inputnode,field_name,iterable_list):
        if iterable_list == []:
            return

        if inputnode.iterables is None:
            inputnode.iterables=[]
            inputnode.synchronize = True

        if type(iterable_list) != list:
            iterable_list = [iterable_list]

        index = self.get_inputnode_iterables_index(inputnode,field_name)
        if index is None:
            inputnode.iterables.append((field_name, iterable_list))
        else:
            inputnode.iterables[index] = (field_name, iterable_list)


    def get_inputnode(self):
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

        inputnode_fieldnames = [x.field_name for x in self._inputnode_fields.values()]
        inputnode = pe.Node(niu.IdentityInterface(fields=inputnode_fieldnames), name='inputnode')

        # if inputnode isn't connected to an upstream parent workflow, the node should be set by command line parameters
        # if any of the inputnode's inputs are connected upstream, it's the parent's job to either use this object's
        # command line parameters to set its own inputnode or to hide this object's command line parameters from
        # the parser. If the parent hides any of the the object's parameters from the parser, it becomes responsible
        # for performing relevant validate_parameters() checks.
        for field in self._inputnode_fields.values():
            if field.default_value_from_commandline:
                parameter_name = field.field_name
                parameter = self.get_parameter(parameter_name)
                # iterables override node connections!! node connections will be ignored.
                # therefore, you must disable (with exclude_list not replace) inputs that you want to override
                # one might think they need the parameter to exist even though they are overriding it - for example, in
                # AntsBrainExtractionBIDS one might think they can't disable in_file because the bids part of it
                # in_file_entities_labels_string is a requirement for in_file_mask, template, and template_probability_mask.
                # The correct way to solve this is to create a new parameter that all those other parameters depend on including
                # in_file. For instance, use self._add_parameter to add something like "input_entities_labels_string", and then
                # have in_file, in_file_mask, template, and template_probability_mask all set their existing_entities_labels_string
                # option to it. Then in_file can be disabled without affecting in_file_mask, template, or template_probability mask.
                if field.iterable:
                    self.set_inputnode_iterable(inputnode,parameter_name,parameter.user_value)
                else:
                    setattr(inputnode.inputs, parameter_name, parameter.user_value)
            else:
                if field.iterable:
                    self.set_inputnode_iterable(inputnode,field.field_name,field.default_value)
                else:
                    setattr(inputnode.inputs, field.field_name, field.default_value)
        self.inputnode = inputnode
        return inputnode

    def get_base_workflow(self):
        """
        Create a nipype workflow with same name as self.pipeline_name and store in self.workflow. If a
        NipypeWorkflowArguments subcomponent exists, set the workflow base_dir using the parameter value.
        If being called inside a subclass' :func:`Workflow.get_workflow` after super().get_workflow(), overwrite
        should be True. This ensures that the subclass call to get_base_workflow will overwrite the self.workflow
        stored by super().get_workflow()'s call to get_base_workflow.
        :param overwrite: Overwrite existing self.workflow
        :return: nipype workflow
        """
        if self.workflow is not None:
            return self.workflow
        workflow = pe.Workflow(self.pipeline_name)
        self.workflow = workflow
        return workflow

    def get_outputnode(self, extra_fields=None):
        # if self.outputnode[self.calling_subclass] is not None:
        #     return self.outputnode[self.calling_subclass]

        if self.outputnode is not None:
            return self.outputnode

        fields = self.outputs
        if extra_fields is not None:
            fields += extra_fields
        if fields != []:
            self.outputnode = pe.Node(niu.IdentityInterface(fields=fields), name='outputnode')
        return self.outputnode



    def get_io_and_workflow(self):
        """
        Helper function for self.get_inputnode, self.get_base_workflow, self.connect_overridden_inputnode, and
        self.connect_superclass_inputnode.
        :param extra_inputnode_fields: additional list of (field_name, default_value) tuples to be included in the inputnode.
        :param overridden_inputnode_exclude_list: List of parameter names that should not be connected to their overridden subcomponents
        :param get_superclass_workflow: Also return the superclass workflow
        :param superclass_inputnode_exclude_list: List of parameter names that should not be connected to the superclass inputnode
        :return:
        """
        inputnode = self.get_inputnode()
        workflow = self.get_base_workflow()
        outputnode = self.get_outputnode()
        return inputnode, outputnode, workflow

    def create_workflow(self, arg_dict=None):
        """
        To be implemented by subclass. Use subcomponents and other nipype components to connect and return
        a nipype workflow.
        """
        raise NotImplementedError('Subclass must define create_workflow function.')

    # def iterable_inputnode(self,wf,iteration_list):
    #     if iteration_list == []:
    #         return wf
    #     iterating_wf = Workflow(wf.name)
    #     #iterating_wf_inputnode = wf.get_node('inputnode').clone('inputnode')
    #     iterating_wf_inputnode = deepcopy(wf.get_node('inputnode'))
    #     for inputnode_dict in iteration_list:
    #         hash_id = hashlib.md5()
    #         hash_id.update(str(inputnode_dict).encode('utf-8'))
    #         hash = hash_id.hexdigest()
    #         #-----
    #         # workflow.clone calls workflow.reset_hierarchy which does some weird renaming of the
    #         # nested workflow hierarchy
    #         # let's just do the clone ourselves
    #         #wf_clone = wf.clone(f'{hash}')
    #         wf_clone = deepcopy(wf)
    #         wf_clone.name = hash
    #         wf_clone._id = hash
    #         for node in wf_clone._graph.nodes():
    #             if not isinstance(node, Workflow):
    #                 node._hierarchy = wf_clone.name
    #         # -----
    #         wf_clone_inputnode = wf_clone.get_node('inputnode')
    #
    #         iterable_fields=[]
    #         for field,value in inputnode_dict.items():
    #             if field not in self.exclude_list:
    #                 if not hasattr(wf_clone_inputnode.inputs, field):
    #                     logger.error(f"Trying to set an iterable '{field}', but it is not a field of {wf.name}.inputnode")
    #
    #                 setattr(wf_clone_inputnode.inputs,field,value)
    #                 iterable_fields.append(field)
    #         if len(iterable_fields)>0:
    #             iterating_wf.add_nodes([wf_clone])
    #             # connect non-iterable inputnode fields
    #             for field in iterating_wf_inputnode.inputs.copyable_trait_names():
    #                 if field not in iterable_fields:
    #                     iterating_wf.connect(iterating_wf_inputnode,field,wf_clone,f'inputnode.{field}')
    #
    #     if len(iterating_wf.list_node_names()) == 0:
    #         return wf
    #     else:
    #         return iterating_wf

    def replace_srcnode_connections(self,srcnode,srcnode_output_name,new_srcnode,new_srcnode_output_name):
        connected = []
        wf = self.workflow
        for dstnode in wf._graph.successors(srcnode):
            if dstnode != new_srcnode:
                for edge in wf._graph.get_edge_data(srcnode, dstnode)["connect"]:
                    if edge[0] == srcnode_output_name:
                        connected.append((dstnode, edge[1]))
        wf.disconnect([(srcnode, x[0], [(srcnode_output_name, x[1])]) for x in connected])
        wf.connect([(new_srcnode, x[0], [(new_srcnode_output_name, x[1])]) for x in connected])

    def outputnode_field_connected(self,field):
        wf = self.workflow
        outputnode = wf.get_node('outputnode')
        for srcnode in wf._graph.predecessors(outputnode):
            for edge in wf._graph.get_edge_data(srcnode,outputnode)["connect"]:
                if edge[1] == field:
                    return True
        return False

    def run_setup(self, dbg_args=None):
        parser = ArgumentParser()

        config_file_obj = Config()
        config_file_obj.populate_parser(parser)

        nipype_run_engine = NipypeRunEngine()
        nipype_run_engine.populate_parser(parser)

        self.populate_parser(parser)

        parsed_namespace = config_file_obj.parse_args(parser, dbg_args)
        parsed_dict = vars(parsed_namespace)

        nipype_run_engine.populate_user_value(parsed_dict)
        self.populate_user_value(parsed_dict)
        self.validate_parameters()

        wf = self.create_workflow()
        Config.write_config_file(parser, parsed_namespace,
                                 os.path.join(nipype_run_engine.nipype_dir, f'{wf.name}.config'))
        return nipype_run_engine, wf
    def run(self, dbg_args=None):
        if hasattr(self,'bids'):
            logger.warning('This pipeline has bids components. You should use run_bids() instead of run().')
        nipype_run_engine, wf = self.run_setup(dbg_args)
        nipype_run_engine.run_workflow(wf)