from workflows.CFMMBase import CFMMInterface
from nipype.interfaces.brainsuite import Bse

class CFMMBse(CFMMInterface):
    group_name = "BrainSuite BSE"
    flag_prefix = 'bse_'
    def __init__(self, *args, **kwargs):
        super().__init__(Bse, *args, **kwargs)
    def add_parser_arguments(self):
        super().add_parser_arguments()

class MouseBse(CFMMBse):
    """
    Wrapper class for CFMMBse with default parameter values suitable for mouse brains.
    """
    def add_parser_arguments(self):
        """
        Modifies default parameter values.
        """
        super().add_parser_arguments()
        self.modify_parser_argument('diffusionConstant', 'default', 30.0)
        self.modify_parser_argument('diffusionIterations', 'default', 3)
        self.modify_parser_argument('edgeDetectionConstant', 'default', 0.55)
        self.modify_parser_argument('radius', 'default', 2)
        self.modify_parser_argument('dilateFinalMask', 'default', True)
        self.modify_parser_argument('trim', 'default', False)
        self.modify_parser_argument('noRotate', 'default', True)