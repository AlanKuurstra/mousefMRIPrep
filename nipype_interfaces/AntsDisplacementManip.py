# only works for .nii.gz displacement transforms and not .h5 composite transforms

import nibabel as nib
import os
from nipype.interfaces.base import (
    CommandLineInputSpec,
    BaseInterface,
    TraitedSpec,
    File,
    OutputMultiPath,
    InputMultiPath,
    traits,
)
from tools.split_exts import split_exts
import numpy as np
import subprocess
from workflows.CFMMLogging import NipypeLogger as logger


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
            displacement_single_vol = displacement_obj.slicer[:, :, :, index:index + 1, :]
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


class MergeDisplacementInputSpec(CommandLineInputSpec):
    displacement_imgs = InputMultiPath(File(exists=True), mandatory=True, desc="Individual displacement volumes")
    tr = traits.Float(1.0, desc='TR of bold scan', mandatory=False, usedefault=True)


class MergeDisplacementOutputSpec(TraitedSpec):
    output_file = File(desc="File", exists=True)


class MergeDisplacement(BaseInterface):
    input_spec = MergeDisplacementInputSpec
    output_spec = MergeDisplacementOutputSpec

    def _run_interface(self, runtime):
        displacement_imgs = self.inputs.displacement_imgs
        output_filename = self._list_outputs()['output_file']

        tr = self.inputs.tr
        num_vols = len(displacement_imgs)

        first_vol = nib.load(displacement_imgs[0])
        first_vol_data = first_vol.get_data()
        first_vol_header = first_vol.header.copy()
        first_vol_header['pixdim'][4] = tr
        merge_shape = list(first_vol_data.shape)
        merge_shape[3] = num_vols
        merge_shape[4] += 1

        merged_img = np.empty(merge_shape, dtype=first_vol_data.dtype)
        merged_img[..., 0:1, :3] = first_vol_data

        for index in range(1, num_vols):
            merged_img[..., index:index + 1, :3] = nib.load(displacement_imgs[index]).get_data()
            merged_img[..., index, 3] = np.zeros(merge_shape[:3], dtype=first_vol_data.dtype)

        nifti_image = nib.Nifti1Image(merged_img, None, first_vol_header)
        nifti_image.to_filename(output_filename)

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_file'] = os.path.abspath('merged_displacement.nii.gz')
        return outputs


class ReplicateDisplacementInputSpec(CommandLineInputSpec):
    input_transform = File(desc="File", exists=True, mandatory=True)
    reps = traits.Int(desc='Number of repetitions to make', mandatory=True)
    tr = traits.Float(1.0, desc='TR of bold scan', mandatory=False, usedefault=True)
    output_file = File('replicated_transform.nii.gz', desc="File", mandatory=False, usedefault=True)


class ReplicateDisplacementOutputSpec(TraitedSpec):
    output_file = File(desc="Replicated Transform", exists=True)


class ReplicateDisplacement(BaseInterface):
    input_spec = ReplicateDisplacementInputSpec
    output_spec = ReplicateDisplacementOutputSpec

    def _run_interface(self, runtime):
        # get input images and values
        input_transform = self.inputs.input_transform
        nvolumes = str(self.inputs.reps)
        tr = str(self.inputs.tr)
        output_filename = self.inputs.output_file

        # process inputs
        logger.info(" ".join(
            ['ImageMath', '3', output_filename, 'ReplicateDisplacement', input_transform, nvolumes, tr, '0']))
        subprocess.call(
            ['ImageMath', '3', output_filename, 'ReplicateDisplacement', input_transform, nvolumes, tr, '0'])

        return runtime

    def _list_outputs(self):
        # Define the file location of your output objects.
        # This helps the node enforce checks to make sure your _run_interface() actually does what you say it will do.
        # ie. The node will check you created the designated output files in the correct location.
        outputs = self.output_spec().get()
        outputs['output_file'] = os.path.abspath(self.inputs.output_file)
        return outputs


class ReplicateImageInputSpec(CommandLineInputSpec):
    input_3d_image = File(desc="File", exists=True, mandatory=True)
    reps = traits.Int(desc='Number of repetitions to make', mandatory=True)
    tr = traits.Float(1.0, desc='TR of bold scan', mandatory=False, usedefault=True)
    output_file = File('replicated_image.nii.gz', desc="File", mandatory=False, usedefault=True)


class ReplicateImageOutputSpec(TraitedSpec):
    output_file = File(desc="Replicated Image", exists=True)


class ReplicateImage(BaseInterface):
    input_spec = ReplicateImageInputSpec
    output_spec = ReplicateImageOutputSpec

    def _run_interface(self, runtime):
        # get input images and values
        input_3d_image = self.inputs.input_3d_image
        nvolumes = str(self.inputs.reps)
        tr = str(self.inputs.tr)
        output_filename = self.inputs.output_file

        # process inputs
        logger.info(
            " ".join(['ImageMath', '3', output_filename, 'ReplicateImage', input_3d_image, nvolumes, tr, '0']))
        subprocess.call(
            ['ImageMath', '3', output_filename, 'ReplicateImage', input_3d_image, nvolumes, tr, '0'])

        return runtime

    def _list_outputs(self):
        # Define the file location of your output objects.
        # This helps the node enforce checks to make sure your _run_interface() actually does what you say it will do.
        # ie. The node will check you created the designated output files in the correct location.
        outputs = self.output_spec().get()
        outputs['output_file'] = os.path.abspath(self.inputs.output_file)
        return outputs