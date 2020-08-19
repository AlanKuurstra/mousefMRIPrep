from workflows.CFMMBase import CFMMParserArguments
from bids import BIDSLayout
import tempfile
import os
from workflows.CFMMBase import CFMMFlagValuePair
from nipype.pipeline import engine as pe
from nipype.interfaces.utility import Function
import argparse

# depending on the values of a BIDS workflow's inputnode, we will either use the explicitly defined
# input file or we will do a bids search for the input image using input_bids_entities_string.
# we can't decide workflow connections based on conditional if statements involving the inputnode.inputs attributes
# because those attributes can be overridden at runtime by upstream workflow connections. Instead we must make
# a multiplexer node which takes the inputnode as input and provides a node output based on conditional
# statements inside the node.
def bids_file_multiplexer(input_file, entities_labels_dict, bids_layout_db):
    """
    output the input_file or, if none is provided, use entity-label pairs to do a BIDS search in the provided layout
    for an appropriate input

    :param input_file: input file that overrides a BIDS search
    :param entities_labels_dict: entity-label pairs for BIDS search
    :param bids_layout_db: BIDS layout for search
    :return: chosen input file
    """
    if input_file is None:
        layout = bids_layout_db.get_layout()
        chosen_file = layout.get(**entities_labels_dict, drop_invalid_filters=False)
        #chosen_file = [bidsfile.path for bidsfile in chosen_file]
        if len(chosen_file) != 1:
            print(f'BIDS search found {len(chosen_file)} input images')
        else:
            chosen_file = chosen_file[0]
    else:
        chosen_file = input_file
    return chosen_file

def get_node_bids_file_multiplexer(name='bids_file_multiplexer'):
    """
    returns a node created from the :func:`bids_file_multiplexer` function
    :param name: nipype name for node
    :return: nipype node
    """
    node = pe.Node(
        Function(input_names=["input_file", "entities_labels_dict", "bids_layout_db"],
             output_names=["chosen_file"],
             function=bids_file_multiplexer), name = name)
    return node


def get_input_file_entities_labels_dict(participant_label, session_labels, run_labels, entities_string):
    from bids.layout.layout import parse_file_entities
    entities_labels_dict = {'desc':None}
    if entities_string is not None:
        entities_labels_dict.update(parse_file_entities(entities_string))
    if participant_label is not None:
        entities_labels_dict['subject'] = participant_label
    if session_labels is not None:
        entities_labels_dict['session'] = session_labels
    if run_labels is not None:
        entities_labels_dict['run'] = run_labels
    return entities_labels_dict

def get_node_get_input_file_entities_labels_dict(name='get_input_file_entities_labels_dict'):
    node = pe.Node(
        Function(input_names=["participant_label", "session_labels", "run_labels", "entities_string"],
                 output_names=["entities_labels_dict"],
                 function=get_input_file_entities_labels_dict), name=name)
    return node

def update_entities_labels_dict(entities_labels_dict,entity,label):
    if type(entity) is list:
        assert(len(entity)==len(label))
        for entity_element,label_element in zip(entity,label):
            entities_labels_dict[entity_element] = label_element
    else:
        entities_labels_dict[entity] = label
    return entities_labels_dict

def get_node_update_entities_labels_dict(name='update_entities_labels_dict'):
    node = pe.Node(
        Function(input_names=["entities_labels_dict", "entity", "label"],
             output_names=["entities_labels_dict"],
             function=update_entities_labels_dict), name = name)
    return node

def batch_update_entities_labels_dict(entities_labels_dict,remove_entities_list=[],add_entities_labels_dict={}):
    for entity in remove_entities_list:
        if entity in entities_labels_dict.keys():
            del entities_labels_dict[entity]
    entities_labels_dict.update(add_entities_labels_dict)
    return entities_labels_dict

def get_node_batch_update_entities_labels_dict(name='batch_update_entities_labels_dict'):
    node = pe.Node(
        Function(input_names=["entities_labels_dict", "remove_entities_list", "add_entities_labels_dict"],
             output_names=["entities_labels_dict"],
             function=batch_update_entities_labels_dict), name = name)
    return node

class BIDSLayoutDB():
    def __init__(self, bids_dir=None, derivatives_dirs=None, layout_db=None):
        self.set_directories(bids_dir,derivatives_dirs,layout_db)
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

    def get_layout(self):
        layout = BIDSLayout(self.bids_dir, database_path=self.layout_db)
        for derivatives_dir in self.derivatives_dirs:
            layout.add_derivatives(derivatives_dir, parent_database_path=self.layout_db)
        return layout


class BIDSAppArguments(CFMMParserArguments):
    group_name = "BIDS Arguments"
    def __init__(self, *args, **kwargs):
        self.bids_layout_db = BIDSLayoutDB(None, None, None)
        super().__init__(*args, **kwargs)

    def add_parser_argument(self, parameter_name, *args, **kwargs):
        # only add positional arguments to argparse parser if you're the top level bids workflow
        if self.parent == self.get_toplevel_parent():
            super().add_parser_argument(parameter_name, *args, **kwargs)
        else:
            if parameter_name not in self.exclude_list + list(self._parameters.keys()):
                self._parameters[parameter_name] = CFMMFlagValuePair(None, None, None)

    def modify_parser_argument(self, *args, **kwargs):
        if self.parent == self.get_toplevel_parent():
            super().modify_parser_argument(*args, **kwargs)

    def hide_parser_argument(self, *args, **kwargs):
        if self.parent == self.get_toplevel_parent():
            super().hide_parser_argument(*args, **kwargs)

    def add_parser_arguments(self):
        self.add_parser_argument('bids_dir',
                                 optional=False,
                                 help='Data directory formatted according to BIDS standard.',)


        self.add_parser_argument('output_derivatives_dir',
                                 optional=False,
                                 help='Directory where processed output files are to be stored in bids derivatives format.')

        self.add_parser_argument('analysis_level',
                                 optional=False,
                                 help='Level of the analysis that will be performed.',
                                 choices=['participant', 'group'])

        self.add_parser_argument('input_derivatives_dirs',
                                 help='List of additional bids derivatives dirs used for searching.',
                                 type=eval,)
                                 #nargs="+",)

        self.add_parser_argument('bids_layout_db',
                                 help='Path to database for storing indexing of bids_dir and input_derivatives_dirs',
                                 )

        self.add_parser_argument('participant_label',
                                 help='The label(s) of the participant(s) that should be analyzed. The label '
                                      'corresponds to sub-<participant_label> from the BIDS spec '
                                      '(do not include prefix "sub-"). If this parameter is not '
                                      'provided all subjects will be analyzed. Multiple '
                                      'participants can be specified with a space separated list.')

        self.add_parser_argument('session_labels',
                                 help='The label(s) of the session(s) that should be analyzed. The label '
                                      'corresponds to ses-<session_label> from the BIDS spec '
                                      '(do not include prefix "ses-"). If this parameter is not '
                                      'provided all sessions will be analyzed. Multiple '
                                      'sessions can be specified with a space separated list.',
                                 nargs="+")

        self.add_parser_argument('run_labels',
                                 help='The label(s) of the run(s) that should be analyzed. The label '
                                      'corresponds to run-<run_label> from the BIDS spec '
                                      '(do not include prefix "run-"). If this parameter is not '
                                      'provided all runs will be analyzed. Multiple '
                                      'runs can be specified with a space separated list.',
                                 nargs="+")

    def populate_parameters(self, arg_dict):
        # only one workflow can have positional bids arguments for bids app
        # populate bids parameters using top level parent bids arguments

        current_component = self.parent
        toplevel_bids_app_arguments = self
        while current_component.parent is not None:
            current_component = current_component.parent
            for subcomponent in current_component.subcomponents:
                if type(subcomponent) == type(self):
                    toplevel_bids_app_arguments = subcomponent

        for parameter_name in self._parameters.keys():
            toplevel_parameter = toplevel_bids_app_arguments.get_parameter(parameter_name)
            if toplevel_parameter.parser_flag in arg_dict.keys():
                self.get_parameter(parameter_name).user_value = arg_dict[toplevel_parameter.parser_flag]

        if self == toplevel_bids_app_arguments:
            # bidslayout uses in memory db by default, but nipype needs to pickle anything it passes to nodes
            # here we specify a file db for bidslayout to use so that it can be pickled to other nodes and this way
            # indexing only happens once
            db_location = arg_dict[self.get_parameter('bids_layout_db').parser_flag]
            if db_location is None:
                db_location = tempfile.TemporaryDirectory().name
            else:
                db_location = os.path.abspath(db_location)

            self.bids_layout_db.set_directories(self.get_parameter('bids_dir').user_value,
                                                self.get_parameter('input_derivatives_dirs').user_value,
                                                db_location)
        else:
            self.bids_layout_db = toplevel_bids_app_arguments.bids_layout_db

        #slightly hacky
        self.get_parameter('bids_layout_db').user_value = self.bids_layout_db

class FunctionalBIDSAppArguments(BIDSAppArguments):
    group_name = "BIDS Functional Arguments"
    def __init__(self, *args, **kwargs):
        if 'exclude_list' in kwargs:
            exclude_list = kwargs.pop('exclude_list')
        else:
            exclude_list = []
        exclude_list = exclude_list + [
            'session_labels',
            'run_labels',
        ]
        self.layout = None
        super().__init__(*args, exclude_list=exclude_list, **kwargs)

    def add_parser_arguments(self):
        super().add_parser_arguments()

        self.add_parser_argument('func_session_labels',
                                 help='The label(s) of the session(s) that should be analyzed. The label '
                                      'corresponds to ses-<session_label> from the BIDS spec '
                                      '(do not include prefix "ses-"). If this parameter is not '
                                      'provided all sessions will be analyzed. Multiple '
                                      'sessions can be specified with a space separated list.',
                                 nargs="+")

        self.add_parser_argument('func_run_labels',
                                 help='The label(s) of the run(s) that should be analyzed. The label '
                                      'corresponds to run-<run_label> from the BIDS spec '
                                      '(do not include prefix "run-"). If this parameter is not '
                                      'provided all runs will be analyzed. Multiple '
                                      'runs can be specified with a space separated list.',
                                 nargs="+")

        self.add_parser_argument('anat_session_labels',
                                 help='The label(s) of the session(s) that should be analyzed. The label '
                                      'corresponds to ses-<session_label> from the BIDS spec '
                                      '(do not include prefix "ses-"). If this parameter is not '
                                      'provided all sessions will be analyzed. Multiple '
                                      'sessions can be specified with a space separated list.')

        self.add_parser_argument('anat_run_labels',
                                 help='The label(s) of the run(s) that should be analyzed. The label '
                                      'corresponds to run-<run_label> from the BIDS spec '
                                      '(do not include prefix "run-"). If this parameter is not '
                                      'provided all runs will be analyzed. Multiple '
                                      'runs can be specified with a space separated list.')

