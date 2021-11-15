from nipype.interfaces.base.traits_extension import Undefined
import configargparse as argparse
from traits.has_traits import HasTraits

def label_eval(label):
    from cfmm.bids_parameters import IRRELEVANT, NOT_PRESENT, LEAVE_EXISTING
    return eval(label)  # try locking down the globals and locals of eval

def eval_with_handling(argparse_value):
    try:
        value = eval(argparse_value)
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f'input "{argparse_value}" must be a valid input for python\'s eval(). Did you forget quotes around a string? eg. for a string input use "\'string_1\'" or for a list input use "[\'string_1\',\'string_2\']"')
    return value

class eval_with_trait_validation():
    """
    Class used by :func:`Interface._add_trait_as_parameter` and :func:`ArgumentParser.add_argument` to convert
    commandline text during argument parsing.  Python's eval is used to cast a commandline string to a python object
    which is then validated as a useable input for the desired trait.
    """

    def __init__(self, trait_type):
        """
        Save the trait so self.convert knows which trait to validate the argparse input for.
        :param trait_type:
        """
        self.trait_type = trait_type

    def convert(self, argparse_value):
        """
        The function provided to type argument of :func:`ArgumentParser.add_argument`. Python's eval is used to cast a
        commandline string to a python object which is then validated as a useable input for self.trait_type.
        :param argparse_value:
        :return:
        """
        # argparse does some processing on the commandline input before giving it to the "type" function.
        # therefore the following command inputs all are given as a string
        # --string_input 3
        # --string_input '3'
        # --string_input "3"
        # similarly the following command inputs are also all given as a string
        # --string_input three
        # --string_input 'three'
        # --string_input "three"

        # it's not possible to determine which arguments were meant as strings, and which were meant as numeric types.
        # to overcome this, we require the user to indicate strings using two sets of quotes:
        # --string_input '"mystringinput"' which evaluates to "mystringinput" and can be processed by eval()
        # or
        # --string_input "'mystringinput'"which evaluates to 'mystringinput' and can be processed by eval()

        # similarly, when using a dictionary or list the user should use
        # --my_dict_input "{'string_key':3}"
        # --my_list_input "['string1','string2']"

        # this is annoying for the user, but low maintenance
        # since users will mostly use config files, the annoying strings on the commandline are acceptable

        # with enum traits we have the additional problem of inputting a python None object.
        # some numerated types have both 'None' the string and None the python object
        # using eval we can differentiate the two with '"None"' for the string and 'None' for the python object
        # But without eval, this becomes difficult.  We could decide that '' will convert to the python None object.
        # And every other string input can be cast using the enum.values. But then when dealing with an enum we'd
        # use '' to get a python None object and dealing with a list we'd put "[None, 'three', 3]" which is inconsistent

        try:
            if argparse_value == "<undefined>":
                trait_value = Undefined
            else:
                trait_value = eval(argparse_value)
        except Exception as e:
            raise argparse.ArgumentTypeError(
                f'input "{argparse_value}" must be a valid input for python\'s eval(). Did you forget quotes around a string? eg. for a string input use "\'string_1\'" or for a list input use "[\'string_1\',\'string_2\']"')

        # validate:
        class dummy(HasTraits):
            trait_argument = self.trait_type
        try:
            # Although Undefined throws an error in the validate() function, it is used in a special way by
            # Nipype to set its own default values and therefore is automatically considered valid.
            if trait_value != Undefined:
                self.trait_type.validate(dummy(), 'trait_argument', trait_value)
        except Exception as e:
            raise argparse.ArgumentTypeError(
                str(e).replace("'trait_argument' trait of a dummy instance", f'input "{str(trait_value)}"'))
        return trait_value
