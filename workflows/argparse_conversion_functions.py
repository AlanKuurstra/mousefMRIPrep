from nipype.interfaces.base.traits_extension import Undefined
import configargparse as argparse
from traits.has_traits import HasTraits

def label_eval(label):
    from workflows.CFMMBIDS import IRRELEVANT, NOT_PRESENT, LEAVE_EXISTING
    return eval(label)  # try locking down the globals and locals of eval

def eval_with_handling(argparse_value):
    try:
        value = eval(argparse_value)
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f'input "{argparse_value}" must be a valid input for python\'s eval(). Did you forget quotes around a string? eg. for a string input use "\'string_1\'" or for a list input use "[\'string_1\',\'string_2\']"')
    return value

class convert_argparse_using_eval():
    """
    Class used by :func:`CFMMInterface.convert_trait_to_argument` and :func:`ArgumentParser.add_argument` to convert
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
        # everything that comes from the commandline will be put through python's eval()
        # this causes trouble for strings - every argument from the commandline is read in as a string
        # so it's hard to determine which arguments were meant as strings for eval() and which arguments were meant
        # as something else for eval

        # when a user is inputing a string for eval(), we require them to indicate it using double quotes
        # it's not good enough to use:
        # --string_input mystringinput
        # --string_input 'mystringinput'
        # --string_input "mystringinput"
        # because they will all evaluate to the string mystringinput which can't be processed by eval()

        # We require the user to put
        # --string_input '"mystringinput"' which evaluates to "mystringinput" and can be processed by eval()
        # or
        # --string_input "'mystringinput'"which evaluates to 'mystringinput' and can be processed by eval()

        # similarly, when using a dictionary or list the user should use
        # --my_dict_input "{'string_key':3}"
        # --my_list_input "['string1','string2']"

        # this is annoying for the user, but low maitenance
        # since users will mostly use config files, the annoying strings on the commandline are acceptable


        # we could put in logic for strings outside of lists to behave differently (ie. enum and string traits)

        # if trait is string, don't do eval() - but then a string in list and on it's own is input differently

        # with enum traits we have the additional problem of inputting a python None object.
        # some numerated types have both 'None' the string and None the python object
        # using eval we can differentiate the two with '"None"' for the string and 'None" for the python object
        # But without eval, this becomes difficult.  We could decide that '' will convert to the python None object.
        # And every other string input can be cast using the enum.values.
        # But then None in enum and None in a list or tuple is input differently on the commandline

        try:
            if argparse_value == "<undefined>":
                trait_value = Undefined
            else:
                trait_value = eval(argparse_value)
        except Exception as e:
            raise argparse.ArgumentTypeError(
                f'input "{argparse_value}" must be a valid input for python\'s eval(). Did you forget quotes around a string? eg. for a string input use "\'string_1\'" or for a list input use "[\'string_1\',\'string_2\']"')
        class dummy(HasTraits):
            trait_argument = self.trait_type

        try:
            # Undefined throws an error with this trait type check, but is actually a special type that is safe to
            # use when setting a trait value.
            if trait_value != Undefined:
                self.trait_type.validate(dummy(), 'trait_argument', trait_value)
        except Exception as e:
            raise argparse.ArgumentTypeError(
                str(e).replace("'trait_argument' trait of a dummy instance", f'input "{str(trait_value)}"'))
        return trait_value
