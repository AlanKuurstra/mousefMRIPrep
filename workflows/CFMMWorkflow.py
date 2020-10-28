import os
from nipype.pipeline import engine as pe
from nipype.pipeline.engine import Workflow
import hashlib
from copy import deepcopy
from nipype.interfaces import utility as niu
from workflows.CFMMLogging import NipypeLogger as logger
from workflows.CFMMConfigFile import CFMMConfig
from workflows.CFMMCommon import NipypeRunArguments
import configargparse
from workflows.CFMMParameterGroup import CFMMParameterGroup
from workflows.CFMMParameterGroup import CFMMParserGroups

class inputnode_field():
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

class CFMMWorkflow(CFMMParameterGroup):
    """
    Class for managing a nipype workflow with commandline arguments. Manages a list self.subcomponents of
    CFMMParserArguments subclasses (eg. a CFMMInterface or a CFMMWorkflow used as a subworkflow) which will have their
    commandline arguments displayed in this workflow's help.

    :ivar pipeline_name: initial value:
    :ivar _inputnode_params: initial value:
    :ivar _param_subordinates: initial value:
    :ivar inputnode: initial value:
    :ivar outputnode: initial value:
    :ivar workflow: initial value:
    :ivar pipeline_version: initial value:
    """

    def __init__(self, *args, pipeline_name=None, pipeline_version=None, **kwargs):
        """
        :param subcomponents: List of CFMMParserArguments subclasses
        :param args:
        :param pipeline_name:
        :param pipeline_version:
        :param kwargs:
        """

        if pipeline_name is None:
            pipeline_name = self.__class__.__name__
        self.pipeline_name = pipeline_name
        self.pipeline_short_desc = ''

        if not hasattr(self, 'subcomponents'):
            self.subcomponents = []
        if not hasattr(self, 'extra_inputnode_fields'):
            self.extra_inputnode_fields = []
        if not hasattr(self,'outputs'):
            self.outputs=[]

        self._inputnode_field_info = []
        self.inputnode = None
        self.outputnode = None
        self.workflow = None

        if pipeline_version == None:
            pipeline_version = '1.0'
        self.pipeline_version = pipeline_version
        super().__init__(*args, **kwargs)

    def get_toplevel_owner(self):
        """
        Traverse the composition chain and return first instance without a owner.
        :return: toplevel_owner
        """

        current_child = self
        current_owner = current_child.owner
        while current_owner is not None:
            current_child = current_owner
            current_owner = current_owner.owner
        return current_child

    def _add_parameter(self, parameter_name, *args, add_to_inputnode=True, iterable=False,**kwargs):
        """
        Helper function for :func:`CFMMParserArguments.add_parser_argument`. Allows a workflow parameter to hide
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
                self._inputnode_field_info.append(inputnode_field(parameter_name))
            else:
                self._inputnode_field_info.append(inputnode_field(parameter_name,
                                                              default_value_from_commandline=True,
                                                              iterable=iterable))

    def _add_parameters(self):
        """
        Adds parser arguments from all subcomponents. This function should be called from a user redefined
        add_parser_arguments using super().add_parser_arguments().
        """
        for subcomponent in self.subcomponents:
            if not subcomponent.owner == self:
                continue
            subcomponent._add_parameters()

    def add_subcomponent(self, subcomponent):
        """
        Add a subcomponent to this workflow and assign self as owner.
        :param subcomponent:
        """
        if not hasattr(self, 'subcomponents'):
            self.subcomponents = []
        # maybe should still use a dictionary _subcomponents for unique keys
        # no need for item to be unique if we don't use get_subcomponent and use class attributes instead
        proposed_subcomponents_name = subcomponent.group_name
        for existing_subcomponent in self.subcomponents:
            if proposed_subcomponents_name == existing_subcomponent.group_name:
                raise ValueError(
                    f"Cannot add subcomponent with group_name '{proposed_subcomponents_name}', a subcomponent with that name already exists. Subcomponent group_name must be unique.")
        subcomponent.owner = self
        # # disable all bids derivatives for subcomponents
        # since this function is called during superclass init, and super().__init__ is always called before the rest
        # of the subclass init,
        # the subclass outputs attribute will not be set yet when this function is being called!
        # # toplevel workflow can connect the two workflows outputnodes together to put a result in their own folder
        # # or give a desc name to subworkflow.output[key] which will make the subworkflow save it is a subdirectory
        # # of the owner's derivatives folder
        # if hasattr(subcomponent, 'outputs') and type(subcomponent.outputs) == dict:
        #     for k in subcomponent.outputs: subcomponent.outputs[k] = None
        self.subcomponents.append(subcomponent)

    def _remove_subcomponent(self, attribute_name):
        if not hasattr(self,attribute_name):
            logger.error(f'Trying to replace attribute {attribute_name}, but it does not exist in the superclass.')
        self.subcomponents.remove(getattr(self,attribute_name))
        delattr(self,attribute_name)

    def get_subcomponent(self, groupnames):
        """
        Get a nested subcomponent using the nested group names.
        :param groupnames: Either list of nested group names or the string returned by
        :func:`CFMMParserArguments.get_group_name_chain`
        :return: The desired subcomponent instance
        """
        if type(groupnames) == str:
            groupnames = groupnames.split(os.sep)
        current_subcomponent = self
        # go down the chain to get desired subcomponent
        for group_name in groupnames:
            # maybe should still use a dictionary _subcomponents for its hash table
            for subcomponent in current_subcomponent.subcomponents:
                if subcomponent.group_name == group_name:
                    current_subcomponent = subcomponent
                    break
            else:
                return None
        return current_subcomponent

    def get_parameter(self, parameter_name, subcomponent_chain=None):
        """
        Return a parameter from this workflow or a subcomponent's parameter.
        :param parameter_name:
        :param subcomponent_chain: Either list of nested group names or the string returned by
        :func:`CFMMParserArguments.get_group_name_chain`
        :return: CFMMFlagValuePair object stored in _parameters
        """
        if subcomponent_chain is None:
            return super().get_parameter(parameter_name)
        else:
            subcomponent = self.get_subcomponent(subcomponent_chain)
        return subcomponent.get_parameter(parameter_name)

    def populate_parser_groups(self, cfmm_parser_groups):
        super().populate_parser_groups(cfmm_parser_groups)
        for subcomponent in self.subcomponents:
            subcomponent.populate_parser_groups(cfmm_parser_groups)

    def populate_parameters(self, parsed_args_dict):
        """
        Automatically populate the parameters from all subcomponents.
        :param parsed_args_dict: Dictionary returned by :func:`ArgumentParser.parse_args`
        """
        super().populate_parameters(parsed_args_dict)
        for subcomponent in self.subcomponents:
            if not subcomponent.owner == self:
                continue
            subcomponent.populate_parameters(parsed_args_dict)
        #super().populate_parameters(parsed_args_dict) #ak. oct 16, 2020


    # ----------------------------------------------------
    # add_parameters saves information about how the commandline parameter will appear on the inputnode
    # what's the name, what's the default value, should it be iterable\
    # class vs list vs dictionary for storing info?????

    # inputnode_field is a class to store information about the field that will be added to inputnode
    # the field can come from a commandline parameter, or can be a userdefined field that is not directly from cmdline param
    # get_inputnode uses a list of these objects to build the inputnode

    # add_ipnutnode_field_info saves the custom objects to a list
    # get_inputnode_field_info_index gets the list index for a given field name
    # get inputnode_field gets the list value for the given fieldname using the list and get_inputnode_field_info_index
    # would this be better as a dictionary??

    # note that inputnode iterables are meant to be set by get_inputnode.
    # get_inputnode sets the iterable using the commandline value or the given default value
    # this means that all the iterable information should be figured out before you get_inputnode
    # a bids search could set its search result in populate_parameters.
    # the inputnode_field object's default value, or it could populate the cmdline_param.user_value
    # or during add_bids_to_workflow you could overwrite the existing iterable using set_inputnode_iterable

    # in a nested workflow scenario with iterables - it's important to note that the iterables override connections
    # from upstream. This means that ALL iterables must be put in the exclude list to avoid overriding.
    # if you only override one iterable - then you'll get some weird permutations. if you only override func but not
    # func_mask, for every subject you'll get a result paired with every possible subject mask.

    # if a workflow has iterables, all iterables must be overriden by uptsream users

    # this is the requirement of adding iterables.

    # we are removing iteration from the code.

    def add_inputnode_field_info(self, field_name, default_value = None, iterable = False):
        # check to see if inpunode has already been instatiated
        #this function should only be used before self.get_inputnode()
        self._inputnode_field_info.append(inputnode_field(field_name,default_value=default_value,iterable=iterable))

    #is this function actually required?
    def get_inputnode_field_info_index(self, field_name):
        for index in range(len(self._inputnode_field_info)):
            if self._inputnode_field_info[index].field_name == field_name:
                return index
        return None

    def get_inputnode_field(self,field_name):
        index = self.get_inputnode_field_info_index(field_name)
        if index is None:
            return None
        else:
            return self._inputnode_field_info[index]
    # ----------------------------------------------------

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

        inputnode_fieldnames = [x.field_name for x in self._inputnode_field_info]
        inputnode = pe.Node(niu.IdentityInterface(fields=inputnode_fieldnames), name='inputnode')

        # if inputnode isn't connected to an upstream parent workflow, the node should be set by command line parameters
        # if any of the inputnode's inputs are connected upstream, it's the parent's job to either use this object's
        # command line parameters to set its own inputnode or to hide this object's command line parameters from
        # the parser. If the parent hides any of the the object's parameters from the parser, it becomes responsible
        # for performing relevant validate_parameters() checks.
        for field in self._inputnode_field_info:
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
        If being called inside a subclass' :func:`CFMMWorkflow.get_workflow` after super().get_workflow(), overwrite
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

    def validate_parameters(self):
        """
        Validate user inputs for arguments in all subcomponents.
        """
        for subcomponent in self.subcomponents:
            if not subcomponent.owner == self:
                continue
            subcomponent.validate_parameters()

    def run(self,dbg_args=None):
        parser_groups = CFMMParserGroups(configargparse.ArgumentParser())

        config_file_obj = CFMMConfig()
        config_file_obj.populate_parser_groups(parser_groups)

        nipype_run_arguments = NipypeRunArguments()
        nipype_run_arguments.populate_parser_groups(parser_groups)


        self.populate_parser_groups(parser_groups)
        # parser_groups.parser.print_help()

        parsed_namespace = config_file_obj.parse_args(parser_groups, dbg_args)
        parsed_dict = vars(parsed_namespace)

        nipype_run_arguments.populate_parameters(parsed_dict)
        self.populate_parameters(parsed_dict)
        self.validate_parameters()

        wf = self.create_workflow()
        #wf.write_graph(graph2use='flat')

        nipype_run_arguments.run_workflow(wf)