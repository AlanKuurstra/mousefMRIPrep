from workflows.CFMMInterface import CFMMInterface
from nipype.interfaces.fsl import TemporalFilter, SUSAN

class CFMMTemporalFilter(CFMMInterface):
    # uses fslmaths -bptf
    group_name = 'fslmaths -bptf'
    flag_prefix = 'tf_'
    def __init__(self, *args, **kwargs):
        super().__init__(TemporalFilter, *args, **kwargs)

class CFMMSpatialSmoothing(CFMMInterface):
    group_name = 'SUSAN'
    flag_prefix = 'smooth_'
    def __init__(self, *args, **kwargs):
        super().__init__(SUSAN, *args, **kwargs)
    def _add_parameters(self):
        super()._add_parameters()
        self._modify_parameter('usans',
                               'help',
                               self.get_parameter('usans').add_argument_inputs['help'].replace('%',' percent'))


if __name__ == '__main__':
    import configargparse
    from workflows.CFMMParameterGroup import CFMMParserGroups

    parser_groups = CFMMParserGroups(configargparse.ArgumentParser())
    tmp = CFMMSpatialSmoothing()
    tmp.populate_parser_groups(parser_groups)

    parser_groups.parser.print_help()
