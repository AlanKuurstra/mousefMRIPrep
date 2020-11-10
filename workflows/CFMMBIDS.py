from bids import BIDSLayout
import tempfile
import os
from workflows.CFMMParameterGroup import CFMMCommandlineParameter
from nipype.pipeline.engine import Node, Workflow
from nipype_interfaces.DerivativesDatasink import get_node_derivatives_datasink
from workflows.CFMMCommon import get_node_inputs_to_list, get_node_existing_inputs_to_list, get_fn_node, listify, \
    delistify
from workflows.CFMMLogging import NipypeLogger as logger
from workflows.CFMMWorkflow import inputnode_field
from nipype_interfaces.DerivativesDatasink import get_derivatives_entities
from workflows.CFMMParameterGroup import CFMMParameterGroup
from workflows.argparse_conversion_functions import label_eval

# sentinel values for bids entities:
# the file cannot have the entity
NOT_PRESENT = None
# the file should get the entity's label from a command line argument
CMDLINE_VALUE = object()


# the default value for a cmdline argument for an entity label
# LEAVE_EXISTING = object()
# the file can have any label for the entity
# IRRELEVANT = object()

class IrrelevantClass():
    def __repr__(self):
        # this sentinel is used as a label for an entity
        # if a label is of None type, then the filename is not allowed to have that entity in it
        # this sentinel allows for the case when you don't care if the filename has the entity or what the label is
        # this string will be used for nipype hashing
        # the random ending is to avoid a clash with a potential string label 'IRRELEVANT' (eg. if any image ever
        # had _desc-IRRELEVANT and we switched from this sentinel to the string IRRELEVANT, nipype's hash for the node
        # would be the same even though an input had changed
        # return 'IRRELEVANT_;a89quj;;askdjoipqwr3jeklanjdflknsdlcvn.xz,mcv.laesj'
        # actually __repr__ must match the commandline string used with the argparse action.type to implement it
        return 'IRRELEVANT'
IRRELEVANT = IrrelevantClass()


class LeaveExistingClass():
    def __repr__(self):
        #return 'LEAVE_EXISTING_;a89quj;;askdjoipqwr3jeklanjdflknsdlcvn.xz,mcv.laesj'
        return 'LEAVE_EXISTING'
LEAVE_EXISTING = LeaveExistingClass()


class BIDSLayoutDB():
    def __init__(self, bids_dir=None, derivatives_dirs=None, layout_db=None):
        self.set_directories(bids_dir, derivatives_dirs, layout_db)

    def set_directories(self, bids_dir, derivatives_dirs, layout_db):
        if bids_dir is None:
            self.bids_dir = bids_dir
        else:
            self.bids_dir = os.path.abspath(bids_dir)
        if derivatives_dirs is None:
            self.derivatives_dirs = derivatives_dirs
        else:
            self.derivatives_dirs = [os.path.abspath(derivatives_dir) for derivatives_dir in derivatives_dirs]
        if layout_db is None:
            self.layout_db = layout_db
        else:
            self.layout_db = os.path.abspath(layout_db)

    def get_layout(self, reset_database=False):
        # reset database for now until we can find if there's a better way to add new files to the database
        # we also need to do this when we write derivatives files
        layout = BIDSLayout(self.bids_dir, database_path=self.layout_db, reset_database=reset_database)
        for derivatives_dir in self.derivatives_dirs:
            layout.add_derivatives(derivatives_dir, parent_database_path=self.layout_db, reset_database=reset_database)
        return layout

    def __repr__(self):
        # when nipype hashes a python object, it hashes it's string. By default the string prints the memory location
        # nipype also copies nodes to achieve iterables (probably deep copy). Therefore, the memory address of the
        # object is always changing, and then the string to be hashed is always changing, and nodes using this object
        # are always being rerun.  We want the hash to be related to the files, so let's have the print string relay
        # that information. See nipype's Node.is_cached() fn

        # note that this will check that you're using the same database file, but will not check the datetime of the
        # last time the database was updated
        return str(self.__dict__)


class BIDSAppArguments(CFMMParameterGroup):
    group_name = "BIDS Arguments"

    def __init__(self, *args, **kwargs):
        self.bids_layout_db = BIDSLayoutDB(None, None, None)
        super().__init__(*args, **kwargs)

    def _add_parameter(self, parameter_name, *args, **kwargs):
        # only add positional arguments to argparse parser if you're the top level bids workflow
        if self.owner == self.get_toplevel_owner():
            super()._add_parameter(parameter_name, *args, **kwargs)
        else:
            if parameter_name not in self.exclude_list + list(self._parameters.keys()):
                self._parameters[parameter_name] = CFMMCommandlineParameter(None)

    def _modify_parameter(self, *args, **kwargs):
        if self.owner == self.get_toplevel_owner():
            super()._modify_parameter(*args, **kwargs)

    def _add_parameters(self):
        self._add_parameter('bids_dir',
                            optional=False,
                            help='Data directory formatted according to BIDS standard.',
                            type=eval)

        self._add_parameter('output_derivatives_dir',
                            optional=False,
                            help='Directory where processed output files are to be stored in bids derivatives format.',
                            type=eval)

        self._add_parameter('analysis_level',
                            optional=False,
                            help='Level of the analysis that will be performed.',
                            choices=['participant', 'group'],
                            type=eval)

        self._add_parameter('input_derivatives_dirs',
                            help='List of additional bids derivatives dirs used for searching.',
                            type=eval, )

        self._add_parameter('bids_layout_db',
                            help='Path to database for storing indexing of bids_dir and input_derivatives_dirs',
                            type=eval
                            )

        self._add_parameter('reset_db',
                            action='store_true',
                            help='Reset the database to index any files added to the bids '
                                 'directories since the last reset.',
                            )
        self._add_parameter('ignore_derivatives_cache',
                            action='store_true',
                            help='Run the pipeline even if derivatives already exist.',
                            )

    def populate_parameters(self, arg_dict):
        # only one workflow can have positional bids arguments for bids app
        # populate bids parameters using top level parent bids arguments

        current_component = self.owner
        toplevel_bids_app_arguments = self
        while current_component.owner is not None:
            current_component = current_component.owner
            for subcomponent in current_component.subcomponents:
                if type(subcomponent) == type(self):
                    toplevel_bids_app_arguments = subcomponent

        for parameter_name in self._parameters.keys():
            toplevel_parameter = toplevel_bids_app_arguments.get_parameter(parameter_name)
            if toplevel_parameter.flagname in arg_dict.keys():
                self.get_parameter(parameter_name).user_value = arg_dict[toplevel_parameter.flagname]

        if self == toplevel_bids_app_arguments:
            # bidslayout uses in memory db by default, but nipype needs to pickle anything it passes to nodes
            # here we specify a file db for bidslayout to use so that it can be pickled to other nodes and this way
            # indexing only happens once
            user_db_location = arg_dict[self.get_parameter('bids_layout_db').flagname]
            if user_db_location is None:
                db_location = tempfile.TemporaryDirectory().name
            else:
                db_location = os.path.abspath(user_db_location)

            self.bids_layout_db.set_directories(self.get_parameter('bids_dir').user_value,
                                                self.get_parameter('input_derivatives_dirs').user_value,
                                                db_location)

            # regenerate the database once at the beginning of the program
            # this should be an option because there can be large databases
            if self.get_parameter('reset_db').user_value:
                if user_db_location is not None:
                    logger.info(f'Resetting database {os.path.abspath(user_db_location)}.')
                    self.bids_layout_db.get_layout(reset_database=True)
            elif user_db_location is not None:
                logger.info(f"Reusing old database {os.path.abspath(user_db_location)}. Any images added to the BIDS "
                            f"directories after the creation of the old database will be ignored. "
                            f"See --{self.get_parameter('reset_db').flagname} to update database.")
        else:
            self.bids_layout_db = toplevel_bids_app_arguments.bids_layout_db

        # slightly hacky
        self.get_parameter('bids_layout_db').user_value = self.bids_layout_db

def lists_to_dict(keys_list, values_list):
    return dict(zip(keys_list, values_list))


def overwrite_dict(existing_dict, keys_list, values_list):
    if keys_list is not None and values_list is not None:
        overwrite_dict = dict(zip(keys_list, values_list))
        existing_dict.update(overwrite_dict)
    return existing_dict


def extend_dict(existing_dict, keys_list, values_list):
    if keys_list is not None and values_list is not None:
        for key, value in zip(keys_list, values_list):
            if key in existing_dict.keys():
                if type(existing_dict[key]) == list:
                    existing_dict[key].extend(value)
                else:
                    existing_dict[key] = [existing_dict[key], value]
            else:
                existing_dict[key] = [value]
    return existing_dict


def update_dict(existing_dict, update_dict):
    existing_dict.update(update_dict)
    return existing_dict


def zip_lists(list1, list2):
    return list(zip(list1, list2))


def parse_file_entities(file, entity_list=None):
    from bids.layout.layout import parse_file_entities
    import os
    ent_vals = parse_file_entities(os.path.join(os.sep, file))

    # # same as derivatives
    # from tools.split_exts import split_exts
    # if 'desc' in ent_vals.keys() and 'suffix' in ent_vals.keys():
    #     if ent_vals['desc'] == ent_vals['suffix']:
    #         del ent_vals['suffix']
    # if 'subject' not in ent_vals.keys():
    #     filename, _ = split_exts(os.path.basename(file))
    #     # bidsify the filename
    #     filename = filename.replace('-', '').replace('_', '').replace('.', '')
    #     ent_vals = {'subject': filename}

    if entity_list:
        ent_vals = {k: v for k, v in ent_vals.items() if k in entity_list}
    return ent_vals


def bids_search(bids_layout_db,
                base_entities_dict,
                entities_to_remove,
                entities_to_overwrite,
                entities_to_extend, ):
    # order of operation: remove, overwrite, extend
    from workflows.CFMMLogging import NipypeLogger as logger
    from workflows.CFMMBIDS import IRRELEVANT, extend_dict
    # by default we want to find an original file, not a derivative
    entities_labels_dict = {'desc': None}
    entities_labels_dict.update(base_entities_dict)

    # list (consider making a set) 
    for entity in entities_to_remove:
        if entity in entities_labels_dict.keys():
            del entities_labels_dict[entity]

    # dict, should only have an entity entered once if we're overwriting
    for entity, label in entities_to_overwrite.items():
        # using the type of the sentinel allows it to be used across different processes
        if type(label) == type(IRRELEVANT):
            if entity in entities_labels_dict.keys():
                del entities_labels_dict[entity]
        else:
            entities_labels_dict[entity] = label

    # list of tuples, can have an entity entered more than once
    entity_list = [x[0] for x in entities_to_extend]
    label_list = [x[1] for x in entities_to_extend]
    entities_labels_dict = extend_dict(entities_labels_dict, entity_list, label_list)

    layout = bids_layout_db.get_layout()
    bids_search = layout.get(**entities_labels_dict, invalid_filters='allow')
    bids_search = [x.path for x in bids_search]
    input_length = len(bids_search)
    # if input_length == 0:
    #     logger.warning(
    #         f'BIDS search for {entities_labels_dict} found {len(bids_search)} images: {bids_search}')
    if input_length == 1:
        bids_search = bids_search[0]
    return bids_search


def bids_search_override(input_parameter,
                         input_parameter_original_file,
                         bids_layout_db,
                         base_entities_dict,
                         entities_to_remove,
                         entities_to_overwrite,
                         entities_to_extend,
                         ):
    from workflows.CFMMBIDS import bids_search
    # only do bids search if input_parameter is not given by upstream

    if input_parameter is None and input_parameter_original_file is None:
        bids_search_result = bids_search(bids_layout_db, base_entities_dict,
                                         entities_to_remove, entities_to_overwrite, entities_to_extend)
        return bids_search_result, bids_search_result
    return input_parameter, input_parameter_original_file


def get_node_parse_file_entities(name='parse_file_entities'):
    return get_fn_node(parse_file_entities, output_names=['ent_vals'], name=name)


def get_node_update_dict(name='update_dict'):
    return get_fn_node(update_dict, output_names=['updated_dict'], name=name)


def get_node_lists_to_dict(name=None):
    return get_fn_node(lists_to_dict, output_names=['output_dict'], name=name)


def get_node_zip_lists(name=None):
    return get_fn_node(zip_lists, output_names=['list_of_tuples'], name=name)


def get_node_bids_search_override(name=None, output_names=('input_parameter', 'input_parameter_original_file')):
    # even though the database file location and the bids directories have not changed, the database may have been
    # updated to include more files. Need to rerun this node every time. overwrite=True
    return get_fn_node(bids_search_override, output_names=output_names, name=name, overwrite=True)


class CFMMBIDSInput():
    # this is a subcomponent that doesn't subclass CFMMParameterGroup
    # instead, it adds parameters (for the bids search) to the owner's parameter group
    # it mimics a CMMParameterGroup
    def _add_parameters(self):
        input_parameter = self.input_parameter
        owner = self.owner
        # add to the inputnode but don't make a cmdline parameter
        # use the setter
        owner._inputnode_field_info.append(inputnode_field(f'{input_parameter}_original_file',
                                                           default_value=None,
                                                           iterable=owner.get_inputnode_field(input_parameter).iterable))

        input_parameter_flagname = owner.get_parameter(input_parameter).flagname if input_parameter not in owner.exclude_list else 'DOES NOT EXIST'
        if self.create_base_bids_string:
            base_cmdline_param = f'{input_parameter}_base_bids_string'
            owner._add_parameter(base_cmdline_param,
                                 help=f'The base BIDS entity-label search string for {input_parameter}. The bids '
                                      f'search can be overridden by --{input_parameter_flagname}.',
                                 add_to_inputnode=False,
                                 default="''",
                                 )

        for entity, label in list(self.entities_to_overwrite.items()) + self.entities_to_extend:
            if label is CMDLINE_VALUE:
                owner._add_parameter(f'{input_parameter}_{entity}',
                                     help=f"The label(s) of the BIDS entity '{entity}' ({entity}-<{entity}_label>) "
                                          f"used to search for {input_parameter} . Overridden by "
                                          f"--{input_parameter_flagname}.",
                                     default=LEAVE_EXISTING,
                                     type=label_eval,
                                     add_to_inputnode=False)

    def __init__(self,
                 owner,
                 input_parameter,
                 create_base_bids_string=True,
                 entities_to_remove=[],
                 entities_to_overwrite={},
                 entities_to_extend=[],
                 output_derivatives=None,
                 derivatives_mapnode=False,
                 disable_derivatives=False):

        if input_parameter in owner.exclude_list:
            # if corresponding parameter in the owner is in the exclude_list, the add all the related bids parameters
            # to the exclude list too!
            bids_exclude_list = []
            bids_exclude_list.append(f'{input_parameter}_base_bids_string')
            for entity, label in list(entities_to_overwrite.items()) + entities_to_extend:
                if label is CMDLINE_VALUE:
                    bids_exclude_list.append(f'{input_parameter}_{entity}')
            owner.exclude_list.extend(bids_exclude_list)

        self.owner = owner
        self.input_parameter = input_parameter
        self.create_base_bids_string = create_base_bids_string
        self.entities_to_remove = entities_to_remove
        self.entities_to_overwrite = entities_to_overwrite
        self.entities_to_extend = entities_to_extend
        self.output_derivatives = output_derivatives
        self.derivatives_mapnode = derivatives_mapnode
        self.disable_derivatives = disable_derivatives

        self.owner_wf = None
        # putting derivatives nodes and bids search nodes in their own respective workflows
        # cleans up the working directory (puts the bids node caches in their own directory)
        self.bids_search_wf = None
        self.derivatives_wf = None
        # either inputnode or bids_search node depending on if
        self.bids_search_node = None

        self.group_name = f'{input_parameter}_bids_input'
        owner.add_subcomponent(self)
        self._add_parameters()

    def populate_parameters(self, parsed_args_dict):
        owner = self.owner
        input_parameter = self.input_parameter

        if input_parameter not in owner.exclude_list:
            entities_to_overwrite = self.entities_to_overwrite
            entities_to_extend = self.entities_to_extend

            # populate CMDLINE_VALUE and LEAVE_EXISTING sentinels
            entities_to_overwrite_populated = {}
            for entity, label in entities_to_overwrite.items():
                if label is CMDLINE_VALUE:
                    label = owner.get_parameter(f'{input_parameter}_{entity}').user_value
                if label is LEAVE_EXISTING:
                    continue
                else:
                    entities_to_overwrite_populated[entity] = label

            entities_to_extend_populated = []
            for entity, label in entities_to_extend:
                if label is CMDLINE_VALUE:
                    label = owner.get_parameter(f'{input_parameter}_{entity}').user_value
                if label is LEAVE_EXISTING:
                    continue
                else:
                    entities_to_extend_populated.append((entity, label))

            self.entities_to_overwrite = entities_to_overwrite_populated
            self.entities_to_extend = entities_to_extend_populated

    def add_derivatives(self):
        owner_wf = self.owner_wf
        derivatives_wf = self.derivatives_wf
        bids_search_wf = self.bids_search_wf
        inputnode = owner_wf.get_node('inputnode')
        outputnode = owner_wf.get_node('outputnode')

        #if input_parameter not in self.owner.exclude_list and
        if not self.disable_derivatives:
            input_parameter = self.input_parameter
            derivatives_info = self.output_derivatives
            mapnode = self.derivatives_mapnode
            if derivatives_info is not None:
                pipeline_name = self.owner.get_toplevel_owner().pipeline_name
                pipeline_dataset_desc = self.owner.get_toplevel_owner().get_bids_derivatives_description()
                # pipeline_nested_path = os.path.join(pipeline_name, os.sep.join(self.get_nested_groupnames()[1:]))
                pipeline_nested_path = pipeline_name

                datasink_name = f'{input_parameter}_derivatives_datasink'
                enabled_outputs = {k: v for k, v in derivatives_info.items() if v is not None}

                # note: we can't put the derivatives_datasink into the BIDS workflow because it would cause a cyclical
                # graph for wf.  wf.inputnode connects into BIDS.bidssearch, then BIDS.bidssearch passes stuff along
                # to the rest of wf, we can't now have wf.outputnode connect back into BIDS.derivatives
                derivatives_datasink = get_node_derivatives_datasink(name=datasink_name, mapnode=mapnode)
                pipeline_output_list = get_node_inputs_to_list(f'{input_parameter}_derivatives_inputs_list',
                                                               mapnode=mapnode)

                pipeline_output_list.inputs.list_length = len(enabled_outputs)
                derivatives_datasink.inputs.dataset_description_dict = pipeline_dataset_desc
                derivatives_datasink.inputs.pipeline_name = pipeline_nested_path

                # the derivative files need to be a mapnode if the input is a mapnode
                # if there are a list of inputs for a field
                # every derivative of that field should have a list
                # however the descriptions stay the same - they don't need to be a mapnode
                # we might have 5 input images, which gives us 5 output images, but the description of those
                # 5 output images is always the same.
                derivatives_desc_list = get_node_inputs_to_list(f'{input_parameter}_derivatives_desc_list', )
                pipeline_output_list.inputs.list_length = len(enabled_outputs)
                index = 1
                for derivatives_desc in list(enabled_outputs.values()):
                    if type(derivatives_desc) == str:
                        setattr(derivatives_desc_list.inputs, f'input{index}', derivatives_desc)
                    elif isinstance(derivatives_desc, Node):
                        node = derivatives_desc
                        derivatives_wf.connect(node, 'desc', derivatives_desc_list, f'input{index}')

                    index += 1

                index = 1
                for field in enabled_outputs.keys():
                    if field not in outputnode.inputs.trait_get().keys():
                        logger.error(f"Trying to connect the requested derivatives for input '{input_parameter}' but "
                                     f"derivative '{field}' is not on the outputnode.")
                    # wf.connect(outputnode, field, pipeline_output_list, f'input{index}')
                    if pipeline_output_list not in derivatives_wf._get_all_nodes():
                        derivatives_wf.add_nodes([pipeline_output_list])
                    owner_wf.connect(outputnode, field,
                                     derivatives_wf, f'{pipeline_output_list.name}.input{index}')
                    index += 1

                derivatives_wf.connect([
                    (pipeline_output_list, derivatives_datasink, [('return_list', 'derivatives_files_list')]),
                    (derivatives_desc_list, derivatives_datasink, [('return_list', 'derivatives_description_list')]),
                ])
                derivatives_datasink.inputs.derivatives_dir = self.owner.bids.get_parameter(
                    'output_derivatives_dir').user_value

                if isinstance(self, BIDSInputExternalSearch):
                    owner_wf.connect(inputnode,
                                     f'{input_parameter}_original_file',
                                     derivatives_wf,
                                     f'{derivatives_datasink.name}.original_bids_file')
                else:
                    owner_wf.connect(bids_search_wf,
                                     f'{self.bids_search_node.name}.{input_parameter}_original_file',
                                     derivatives_wf,
                                     f'{derivatives_datasink.name}.original_bids_file')

    def populate_parser_groups(self, cfmm_parser_groups):
        pass

    def validate_parameters(self):
        pass


class BIDSInputExternalSearch(CFMMBIDSInput):
    def __init__(self, input_parameter, *args, dependent_search=None, dependent_entities=[], **kwargs):
        super().__init__(input_parameter, *args, **kwargs)
        self.dependent_search = dependent_search
        self.dependent_entities = dependent_entities

    def _search(self, dependent_file=None):
        # search is performed external to pipeline
        # no Nodes allowed in entities_to_overwrite or entities_to_extend
        owner = self.owner
        input_parameter = self.input_parameter
        if input_parameter in self.owner.exclude_list:
            return []
        commandline_parameter = owner.get_parameter(input_parameter)
        if commandline_parameter.user_value:
            return listify(commandline_parameter.user_value)

        entities_to_remove = self.entities_to_remove
        entities_to_overwrite = self.entities_to_overwrite
        entities_to_extend = self.entities_to_extend

        if self.create_base_bids_string:
            base_entities_string = owner.get_parameter(f'{input_parameter}_base_bids_string').user_value
            cmdline_dict = parse_file_entities(base_entities_string)
        else:
            cmdline_dict = {}

        if dependent_file is not None:
            dependent_file_dict = {k: v for k, v in parse_file_entities(dependent_file).items() if
                                   k in self.dependent_entities}
        else:
            dependent_file_dict = {}

        base_dict = {}
        base_dict.update(cmdline_dict)
        base_dict.update(dependent_file_dict)

        for entity, label in entities_to_overwrite.items():
            if isinstance(label, Node):
                raise Exception(
                    f"{input_parameter} uses a nipype Node in entities_to_overwrite to overwrite the entity "
                    f"'{entity}'. However, iterables must perform the BIDS search external to the pipeline and "
                    f"cannot use Node outputs.")
        for entity, label in entities_to_extend:
            if isinstance(label, Node):
                raise Exception(
                    f"{input_parameter} uses a nipype Node in entities_to_extend to extend the entity "
                    f"'{entity}'. However, iterables must perform the BIDS search external to the pipeline and "
                    f"cannot use Node outputs.")

        return listify(
            bids_search(owner.bids.bids_layout_db, base_dict, entities_to_remove, entities_to_overwrite,
                        entities_to_extend))

    def traverse_nested_files(self, nested_file_list):
        # set perform _search() for every nested file in nested_file_list
        newlist = []
        for elem in nested_file_list:
            if type(elem) == list:
                newlist.append(self.traverse_nested_files(elem))
            else:
                newlist.append(self._search(elem))
        return newlist

    def search(self):
        # uses self.dependent_iterable's search to create a nested list of bids search results
        owner = self.owner
        input_parameter = self.input_parameter
        if input_parameter in self.owner.exclude_list:
            return []
        commandline_parameter = owner.get_parameter(input_parameter)
        if commandline_parameter.user_value:
            return listify(commandline_parameter.user_value)
        if self.dependent_search:
            dependent_files = self.dependent_search.search()
            return self.traverse_nested_files(dependent_files)
        else:
            return self._search()


class BIDSInputWorkflow(CFMMBIDSInput):
    def __init__(self, *args, base_input=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_input = base_input

    def _populate_entity_label_list_nodes(self, entity_label_pairs, entity_list_node_name, label_list_node_name, wf):
        # create an entity list and a label list and populate them with node connections, static values
        entity_list_from_pipeline = get_node_existing_inputs_to_list(name=entity_list_node_name)
        label_list_from_pipeline = get_node_existing_inputs_to_list(name=label_list_node_name)
        pipeline_entity_index = 1
        for entity, label in entity_label_pairs:
            input_index = f'input{pipeline_entity_index}'
            if isinstance(label, Node):
                node = label
                node_output = entity
                setattr(entity_list_from_pipeline.inputs, input_index, entity)
                wf.connect(node, node_output, label_list_from_pipeline, input_index)
            else:
                setattr(entity_list_from_pipeline.inputs, input_index, entity)
                setattr(label_list_from_pipeline.inputs, input_index, label)
            pipeline_entity_index += 1
        return entity_list_from_pipeline, label_list_from_pipeline

    def inject_bids_search_wf(self):
        owner = self.owner
        input_parameter = self.input_parameter

        if input_parameter in self.owner.exclude_list:
            pass
            # return the searchnode with the inputnode hooked into it and nothing else

        owner_wf = self.owner_wf
        inputnode = owner_wf.get_node('inputnode')
        bids_search_wf = self.bids_search_wf

        base_cmdline_entity_dict = get_node_parse_file_entities(name=f'{input_parameter}_base_cmdline_entity_dict')

        if self.create_base_bids_string:
            base_cmdline_entity_dict.inputs.file = owner.get_parameter(f'{input_parameter}_base_bids_string').user_value
        else:
            base_cmdline_entity_dict.inputs.file = ''

        base_input_entity_dict = get_node_parse_file_entities(name=f'{input_parameter}_base_input_entity_dict')
        if self.base_input:
            # better return just 1 result
            # if the base_input is a BIDSInputWorkflow then node f'{self.base_input[0]}_bids_search' will exist
            # if it's BIDSIterable, then we should hook up to inputnode
            base_input_search_node = bids_search_wf.get_node(f'{self.base_input[0]}_bids_search')
            if base_input_search_node:
                bids_search_wf.connect(base_input_search_node, f'{self.base_input[0]}_original_file',
                                       base_input_entity_dict, 'file')
            else:
                bids_search_wf.add_nodes([base_input_entity_dict])
                owner_wf.connect(inputnode, f'{self.base_input[0]}_original_file',
                                 bids_search_wf, f'{base_input_entity_dict.name}.file')
            # which entities we should extract from the base input
            base_input_entity_dict.inputs.entity_list = self.base_input[1]
        else:
            base_input_entity_dict.inputs.file = ''

        base_dict = get_node_update_dict(name=f'{input_parameter}_base_dict')
        bids_search_wf.connect([
            [base_cmdline_entity_dict, base_dict, [('ent_vals', 'existing_dict')]],
            [base_input_entity_dict, base_dict, [('ent_vals', 'update_dict')]],
        ])

        # turn entities_to_overwrite dict into a node output
        entity_overwrite_list_from_pipeline, label_overwrite_list_from_pipeline = \
            self._populate_entity_label_list_nodes(self.entities_to_overwrite.items(),
                                                   f'{input_parameter}_entities_to_overwrite',
                                                   f'{input_parameter}_labels_to_overwrite',
                                                   bids_search_wf)
        entities_to_overwrite_node = get_node_lists_to_dict(
            name=f'{input_parameter}_entities_to_overwrite_dict')
        bids_search_wf.connect(entity_overwrite_list_from_pipeline, 'return_list',
                               entities_to_overwrite_node, 'keys_list')
        bids_search_wf.connect(label_overwrite_list_from_pipeline, 'return_list',
                               entities_to_overwrite_node, 'values_list')

        # turn input_parameter list of tuples into a node output
        entity_extend_list_from_pipeline, label_extend_list_from_pipeline = \
            self._populate_entity_label_list_nodes(self.entities_to_extend,
                                                   f'{input_parameter}_entities_to_extend',
                                                   f'{input_parameter}_labels_to_extend',
                                                   bids_search_wf)
        entities_to_extend_node = get_node_zip_lists(
            name=f'{input_parameter}_entities_to_extend_list_of_tuples')
        bids_search_wf.connect(entity_extend_list_from_pipeline, 'return_list',
                               entities_to_extend_node, 'list1')
        bids_search_wf.connect(label_extend_list_from_pipeline, 'return_list',
                               entities_to_extend_node, 'list2')

        # connect inputs to bids_search node
        bids_search_node = get_node_bids_search_override(name=f'{input_parameter}_bids_search',
                                                         output_names=[input_parameter,
                                                                       f'{input_parameter}_original_file'])

        bids_search_node.inputs.bids_layout_db = self.owner.bids.bids_layout_db
        bids_search_wf.connect(base_dict, 'updated_dict', bids_search_node, 'base_entities_dict')
        bids_search_node.inputs.entities_to_remove = self.entities_to_remove
        bids_search_wf.connect(entities_to_overwrite_node, 'output_dict', bids_search_node,
                               'entities_to_overwrite')
        bids_search_wf.connect(entities_to_extend_node, 'list_of_tuples', bids_search_node,
                               'entities_to_extend')

        # inputnode.input_parameter is automatically set by get_inputnode().
        # On the other hand, input_parameter_original_file is on the inputnode but not on the commandline
        # this means it is not set by get_inputnode() and we must set it manually.

        # setattr(inputnode.inputs, f'{input_parameter}_original_file', commandline_parameter.user_value)
        setattr(inputnode.inputs, f'{input_parameter}_original_file', getattr(inputnode.inputs, input_parameter))
        owner_wf.connect(inputnode, input_parameter, bids_search_wf, f'{bids_search_node.name}.input_parameter')
        owner_wf.connect(inputnode, f'{input_parameter}_original_file', bids_search_wf,
                         f'{bids_search_node.name}.input_parameter_original_file')

        # inject the search workflow
        self.owner.replace_srcnode_connections(inputnode, input_parameter,
                                               bids_search_wf, f'{bids_search_node.name}.{input_parameter}')
        self.owner.replace_srcnode_connections(inputnode, f'{input_parameter}_original_file',
                                               bids_search_wf,
                                               f'{bids_search_node.name}.{input_parameter}_original_file')


from nipype_interfaces.DerivativesDatasink import get_node_get_derivatives_entities


class BIDSDerivativesInputWorkflow(CFMMBIDSInput):
    def __init__(self,
                 owner,
                 input_parameter,
                 base_input,  # get original file
                 base_input_derivative_desc,  # get derivative
                 base_input_derivative_extension=LEAVE_EXISTING,
                 output_derivatives=None,
                 derivatives_mapnode=False,
                 disable_derivatives=False
                 ):
        super().__init__(owner=owner, input_parameter=input_parameter, create_base_bids_string=False,
                         output_derivatives=output_derivatives, derivatives_mapnode=derivatives_mapnode,
                         disable_derivatives=disable_derivatives)
        self.base_input = base_input
        self.base_input_derivative_desc = base_input_derivative_desc
        self.base_input_derivative_extension = base_input_derivative_extension

    def inject_bids_search_wf(self):
        owner = self.owner
        input_parameter = self.input_parameter

        if input_parameter in self.owner.exclude_list:
            pass
            # return the searchnode with the inputnode hooked into it and nothing else

        owner_wf = self.owner_wf
        inputnode = owner_wf.get_node('inputnode')
        bids_search_wf = self.bids_search_wf

        derivative_entities = get_node_get_derivatives_entities(name=f'{input_parameter}_entities')

        # derivative's original file
        base_input_search_node = bids_search_wf.get_node(f'{self.base_input}_bids_search')
        if base_input_search_node:
            bids_search_wf.connect(base_input_search_node, f'{self.base_input}_original_file',
                                   derivative_entities, 'original_bids_file')
        else:
            bids_search_wf.add_nodes([derivative_entities])
            owner_wf.connect(inputnode, f'{self.base_input}_original_file',
                             bids_search_wf, f'{derivative_entities.name}.original_bids_file')

        # derivative description
        if isinstance(self.base_input_derivative_desc, Node):
            node = self.base_input_derivative_desc
            bids_search_wf.connect(node, 'desc', derivative_entities, 'derivatives_description')
        else:
            setattr(derivative_entities.inputs, 'derivatives_description', self.base_input_derivative_desc)

        # bids search for derivative
        bids_search_node = get_node_bids_search_override(name=f'{input_parameter}_bids_search',
                                                         output_names=[input_parameter,
                                                                       f'{input_parameter}_original_file'])
        bids_search_node.inputs.bids_layout_db = self.owner.bids.bids_layout_db

        bids_search_wf.connect(derivative_entities, 'ent_vals', bids_search_node, 'base_entities_dict')
        bids_search_node.inputs.entities_to_remove = []
        bids_search_node.inputs.entities_to_extend = []
        # derivative extension
        if self.base_input_derivative_extension is not LEAVE_EXISTING:
            bids_search_node.inputs.entities_to_overwrite = {'extension': self.base_input_derivative_extension}

        # inject
        setattr(inputnode.inputs, f'{input_parameter}_original_file', getattr(inputnode.inputs, input_parameter))
        owner_wf.connect(inputnode, input_parameter, bids_search_wf, f'{bids_search_node.name}.input_parameter')
        owner_wf.connect(inputnode, f'{input_parameter}_original_file', bids_search_wf,
                         f'{bids_search_node.name}.input_parameter_original_file')

        # inject the search workflow
        self.owner.replace_srcnode_connections(inputnode, input_parameter,
                                               bids_search_wf, f'{bids_search_node.name}.{input_parameter}')
        self.owner.replace_srcnode_connections(inputnode, f'{input_parameter}_original_file',
                                               bids_search_wf,
                                               f'{bids_search_node.name}.{input_parameter}_original_file')


























class CFMMBIDSWorkflowMixin():
    def add_bids_parameter_group(self):
        # ensures the attribute name is always self.bids and we can access it in helper functions
        self.bids = BIDSAppArguments(owner=self)

    def add_bids_to_workflow(self, wf):
        # set CFMMBIDSInput workflow for adding bids related nodes to workflow
        # adding derivatives node
        # inserting bids searches into the workflow if some of the search parameters come from results in the workflow
        for subcomponent in self.subcomponents:
            if isinstance(subcomponent, CFMMBIDSInput):
                subcomponent.owner_wf = wf
        self.inject_bids_search()
        self.add_derivatives()

    def inject_bids_search(self):
        # puts all subcomponents bids search nodes in the same directory
        bids_search_wf = Workflow(name='BIDSInputSearches')
        for subcomponent in self.subcomponents:
            if isinstance(subcomponent, (BIDSInputWorkflow, BIDSDerivativesInputWorkflow)):
                subcomponent.bids_search_wf = bids_search_wf
                subcomponent.inject_bids_search_wf()

    def add_derivatives(self):
        # puts all subcomponents bids derivatives nodes in the same directory
        derivatives_wf = Workflow(name='BIDSDerivatives')
        for subcomponent in self.subcomponents:
            if isinstance(subcomponent, CFMMBIDSInput):
                subcomponent.derivatives_wf = derivatives_wf
                subcomponent.add_derivatives()

    def output_derivative_exists(self, original_file, derivative_description):
        # use the derivatives module to perform an appropriate search to see if derivative already exists
        derivative_ent_vals = get_derivatives_entities(original_file, derivative_description)
        layout = self.bids.bids_layout_db.get_layout()
        derivative_file = layout.get(**derivative_ent_vals)
        return len(derivative_file)

    def synchronize_iterables(self, bids_iterables, single_iteration=None, synchronized_iterables=None):
        # iterables from bids searches can be nested lists if one iterable depends on another or there is a
        # dependencing chain
        # eg. func = [f1,f2,f3]
        #     anat = [[a11,a12],[a21],[a31,a32,a33]]
        #     anat_mask = [[[am11],[am12]],[[am21]],[[am31],[am32],[am33]]]

        # we need to flatten these lists to make synchronized iterable lists:
        # func = [f1, f1, f2, f3, f3, f3]
        # anat = [a11,a12,a21,a31,a32,a33]
        # anat_mask = [am11,am12,am21,am31,am32,am33]

        if single_iteration is None:
            single_iteration = dict.fromkeys(bids_iterables.keys())
        if synchronized_iterables is None:
            synchronized_iterables = {}
        input_names = bids_iterables.keys()

        from itertools import zip_longest
        # def zip_equal(*iterables):
        #     sentinel = object()
        #     for combo in zip_longest(*iterables, fillvalue=sentinel):
        #         if sentinel in combo:
        #             iterables_str = ''
        #             for x in bids_iterables.items():
        #                 iterables_str+=str(x)+'\n'
        #             raise ValueError(f'Iterables have different lengths. \n'+iterables_str)
        #         yield combo
        # elements_to_synchronize = zip_equal(*bids_iterables.values())
        from nipype.interfaces.base import Undefined
        # if all bids_iterables.values() are an empty list, then every single_iteration key-value should be
        # set to undefined. But zip_longest just skips them and the iterable inputs do not get set.
        # this is only needed if we do zip_longest instead of zip_equal
        if all(x == [] for x in bids_iterables.values()):
            elements_to_synchronize = [[Undefined]*len(input_names)]
        else:
            elements_to_synchronize = zip_longest(*bids_iterables.values(),fillvalue=Undefined)
        for elem_to_sync in elements_to_synchronize:
            remaining_iterables = {}
            for input_name, value in zip(input_names, elem_to_sync):
                if type(value) != list:
                    single_iteration[input_name] = value
                else:
                    remaining_iterables[input_name] = value
            if len(remaining_iterables) > 0:
                self.synchronize_iterables(remaining_iterables, single_iteration, synchronized_iterables)
            else:
                for k, v in single_iteration.items():
                    synchronized_iterables.setdefault(k, []).append(v)
        return synchronized_iterables


    def image_cached(self, inputnode_field, image):
        for subcomponent in self.subcomponents:
            if isinstance(subcomponent, BIDSInputExternalSearch) and \
                    subcomponent.input_parameter not in self.exclude_list and \
                    subcomponent.input_parameter == inputnode_field and \
                    not subcomponent.disable_derivatives and \
                    subcomponent.output_derivatives is not None:
                for outputnode_field, derivative_desc in subcomponent.output_derivatives.items():
                    if self.outputnode_field_connected(outputnode_field) and \
                            self.output_derivative_exists(image, derivative_desc) < 1:
                        # Counter(bids_derivatives[inputnode_field_name].values())
                        return False
        return True


    def check_bids_cache(self):
        bids_iterables = {}
        bids_non_iterables = {}

        def flatten(l,output=None):
            output = [] if output is None else output
            for i in l:
                if type(i) == list:
                    flatten(i,output)
                else:
                    output.append(i)
            return output

        for subcomponent in self.subcomponents:
            if isinstance(subcomponent,
                          BIDSInputExternalSearch) and subcomponent.input_parameter not in self.exclude_list:
                bids_search_results = subcomponent.search()
                if self.get_inputnode_field(subcomponent.input_parameter).iterable:
                    bids_iterables[subcomponent.input_parameter] = bids_search_results
                else:
                    bids_non_iterables[subcomponent.input_parameter] = flatten(bids_search_results)


        synchronized_iterables = self.synchronize_iterables(bids_iterables)
        inputnode_field_names = list(synchronized_iterables.keys())
        iterable_lists = list(synchronized_iterables.values())

        # full iterables is the synchronized list with original_file fields added in
        full_iterables = {}
        for images in zip(*iterable_lists):
            for inputnode_field_name, image in zip(inputnode_field_names, images):
                full_iterables.setdefault(inputnode_field_name, []).append(image)
                full_iterables.setdefault(f'{inputnode_field_name}_original_file', []).append(image)

        # remove cached results from iteration list
        reduced_iterables = {}
        for images in zip(*iterable_lists):
            for inputnode_field_name, image in zip(inputnode_field_names, images):
                if not self.image_cached(inputnode_field_name,image):
                    # if one image in the iterable isn't cached, all images should be added to the reduced iterables
                    for inputnode_field_name_add, image_add in zip(inputnode_field_names, images):
                        reduced_iterables.setdefault(inputnode_field_name_add, []).append(image_add)
                        reduced_iterables.setdefault(f'{inputnode_field_name_add}_original_file', []).append(image_add)
                    break
            else:
                logger.info(f"Derivatives found for inputs {[f'{f}:{i}' for f,i in zip(inputnode_field_names,images)]}."
                            f"\n\t Skipping.")

        # if all iterables are cached, check if all non-iterables are cached too
        if reduced_iterables == {}:
            is_cached = True
            for inputnode_field_name, images in bids_non_iterables.items():
                for image in listify(images):
                    if not self.image_cached(inputnode_field_name, image):
                        break
                else:
                    continue
                break
            else:
                return True, bids_non_iterables, full_iterables, reduced_iterables

        return False, bids_non_iterables, full_iterables, reduced_iterables

    def run_bids(self, dbg_args=None):
        nipype_run_engine, wf = self.run_setup(dbg_args)


        is_cached, bids_non_iterables, bids_full_iterables, bids_reduced_iterables = self.check_bids_cache()

        if self.bids.get_parameter('ignore_derivatives_cache').user_value:
            is_cached = False
            bids_iterables = bids_full_iterables
        else:
            bids_iterables = bids_reduced_iterables

        if is_cached:
            logger.info(f'Nothing to run, finished {wf.name}')
        else:
            # set results form bids search
            inputnode = wf.get_node('inputnode')
            for field, iterable_list in bids_iterables.items():
                self.set_inputnode_iterable(inputnode, field, iterable_list)
            for field, non_iterable in bids_non_iterables.items():
                setattr(inputnode.inputs, field, delistify(non_iterable))
                setattr(inputnode.inputs, f'{field}_original_file', delistify(non_iterable))
            nipype_run_engine.run_workflow(wf)

    def connect_dynamic_derivatives_desc(self, wf, srcnode, srcnode_output_name, dynamic_node_names,
                                         dynamic_node_input):
        # helper function to connect your derivative node with a node that provides the derivative description from
        # a result inside the pipeline
        nodes_already_connected = []
        for input_parameter, bids_derivatives_info in self._bids_derivatives_info.items():
            derivatives_info, _ = bids_derivatives_info
            if derivatives_info is not None:
                for outputnode_field, derivatives_desc in derivatives_info.items():
                    if isinstance(derivatives_desc, Node):
                        node = derivatives_desc
                        if node.name in dynamic_node_names and node not in nodes_already_connected:
                            bids_derivatives = wf.get_node('BIDSDerivatives')
                            wf.connect(srcnode, srcnode_output_name, bids_derivatives,
                                       f'{node.name}.{dynamic_node_input}')
                            nodes_already_connected.append(node)

    def get_bids_derivatives_description(self):
        """
        Bare bones derivatives description for derivatives datasinks. Should be redefined by subclass to provide a
        more detailed description.
        """
        # how to automate finding bids version?
        bids_version = '1.1.1'

        dataset_desc = {
            'Name': f'{self.pipeline_name} - {self.pipeline_short_desc}',
            'BIDSVersion': bids_version,
            'PipelineDescription': {
                'Name': self.pipeline_name,
                'Version': self.pipeline_version,
                'CodeURL': 'unknown',
            },
        }
        return dataset_desc



    # @classmethod
    # def get_node_replacing_derivatives_datasink_original_bids_file(cls, wf):
    #     dstnode = wf.get_node('derivatives_datasink')
    #     dstnode_output_name = 'original_bids_file'
    #
    #     # new_srcnode_name = 'original_bids_file_replacement'
    #     new_srcnode_name = 'choose_in_file'
    #     while new_srcnode_name in wf.list_node_names():
    #         random_str = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(5)])
    #         new_srcnode_name = f'original_bids_file_replacement_{random_str}'
    #     new_srcnode_output_name = dstnode_output_name
    #     new_srcnode = pe.Node(niu.IdentityInterface(fields=[new_srcnode_output_name]), name=new_srcnode_name)
    #
    #     # find srcnode that supplies derivatives datasink
    #     srcnode = None
    #     for candidate in wf._graph.predecessors(dstnode):
    #         for edge in wf._graph.get_edge_data(candidate, dstnode)["connect"]:
    #             if edge[1] == dstnode_output_name:
    #                 srcnode = candidate
    #                 srcnode_output_name = edge[0]
    #     if srcnode is None:
    #         wf.connect(new_srcnode, new_srcnode_output_name, dstnode, dstnode_output_name)
    #     # reconnect all derivatives nodes that the srcnode supplies
    #     else:
    #         connected_dstnodes = []
    #         for succesor in wf._graph.successors(srcnode):
    #             for edge in wf._graph.get_edge_data(srcnode, succesor)["connect"]:
    #                 if edge == (srcnode_output_name, dstnode_output_name):
    #                     connected_dstnodes.append(succesor)
    #         wf.disconnect([(srcnode, x, [(srcnode_output_name, dstnode_output_name)]) for x in connected_dstnodes])
    #         wf.connect([(new_srcnode, x, [(new_srcnode_output_name, dstnode_output_name)]) for x in connected_dstnodes])
    #     return new_srcnode
