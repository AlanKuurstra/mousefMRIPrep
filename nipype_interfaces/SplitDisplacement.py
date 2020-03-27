import nibabel as nib
import os
from nipype.interfaces.base import (
    CommandLineInputSpec,
    BaseInterface,
    TraitedSpec,
    File,
    OutputMultiPath,
)

from tools.split_exts import split_exts




class SplitDisplacementInputSpec(CommandLineInputSpec):
    displacement_img = File(desc="File", exists=True, mandatory=True, argstr="%s")


class SplitDisplacementOutputSpec(TraitedSpec):
    output_files = OutputMultiPath(File(exists=True), desc="Individual displacement volumes")


class SplitDisplacement(BaseInterface):
    input_spec = SplitDisplacementInputSpec
    output_spec = SplitDisplacementOutputSpec

    def _run_interface(self, runtime):
        displacement_img = self.inputs.displacement_img

        basename, exts = split_exts(displacement_img)
        displacement_obj = nib.load(displacement_img)
        displacement_header = displacement_obj.header
        for index in range(displacement_header['dim'][4]):
            displacement_single_vol = displacement_obj.slicer[:, :, :, index:index+1, :]
            displacement_single_vol_filename = f'{basename}_vol_{index + 1}' + exts
            displacement_single_vol.to_filename(displacement_single_vol_filename)
        return runtime

    def _list_outputs(self):
        # Define the file location of your output objects.
        # This helps the node enforce checks to make sure your _run_interface() actually does what you say it will do.
        # ie. The node will check you created the designated output files in the correct location.
        outputs = self.output_spec().get()

        displacement_img = self.inputs.displacement_img
        basename, exts = split_exts(displacement_img)
        displacement_obj = nib.load(displacement_img)
        displacement_header = displacement_obj.header
        output_files = []
        for index in range(displacement_header['dim'][4]):
            displacement_single_vol_filename = f'{basename}_vol_{index + 1}' + exts
            output_files.append(os.path.abspath(displacement_single_vol_filename))
        outputs['output_files'] = output_files
        return outputs


if __name__ == '__main__':
    tmp = SplitDisplacement()
    tmp.inputs.displacement_img = '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/bold_reference_wf/motion_correct/motcorrWarp.nii.gz'
    result = tmp.run()
    print(result.outputs)
