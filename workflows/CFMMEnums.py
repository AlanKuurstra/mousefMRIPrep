from enum import Enum
import argparse

class ParameterListeningMode(Enum):
    REPLACE_VALUE = 1
    REPLACE_DEFAULT = 2
    IGNORE = 3

class BrainExtractMethod(Enum):
    BRAINSUITE = 1
    REGISTRATION_WITH_INITIAL_MASK = 2
    REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK = 3
    REGISTRATION_NO_INITIAL_MASK = 4
    USER_PROVIDED_MASK = 5
    NO_BRAIN_EXTRACTION = 6

    @staticmethod
    def argparse_convert(s):
        if s in BrainExtractMethod.__members__:
            return BrainExtractMethod[s]
        else:
            # raise ValueError() # ugly, error message has function name in it "invalid from_string value"
            # argparse.ArgumentTypeError() # says the original choise was None instead of s
            # if we just return the string without converting it to the BrainExtractMethod type, the argparse will
            # throw its own type error indicating the s string as an invalid choice
            return s

    # overriding __str__ with just the name is done so that this Enum will work with argparse help
    def __str__(self):
        return self.name

    # overriding __repr__ with just the name is done so that this Enum will work with argparse and give a helpful
    # error message for an incorrect command line choice
    def __repr__(self):
        return self.name



