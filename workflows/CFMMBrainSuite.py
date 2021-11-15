from cfmm.interface import Interface
from nipype.interfaces.brainsuite import Bse

class CFMMBse(Interface):
    group_name = "BrainSuite BSE"
    flag_prefix = 'bse_'
    def __init__(self, *args, **kwargs):
        super().__init__(Bse, *args, **kwargs)
    def _add_parameters(self):
        super()._add_parameters()
        # brainsuite autogenerates a filename with a new extension that causes problems: filename.mask_maths.nii.gz
        self._modify_parameter('outputMaskFile','default',"'mask.nii.gz'")
        self._modify_parameter('outputMRIVolume', 'default', "'bse.nii.gz'")

