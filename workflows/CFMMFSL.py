from cfmm.interface import Interface
from nipype.interfaces.fsl import TemporalFilter, SUSAN, ImageStats
from cfmm.workflow import Workflow
from cfmm.CFMMCommon import get_fn_node

tmp = ImageStats()
tmp.inputs.traits()

class bptf(Interface):
    group_name = 'fslmaths -bptf'
    flag_prefix = 'bptf_'
    def __init__(self, *args, **kwargs):
        super().__init__(TemporalFilter, *args, **kwargs)

from nipype.interfaces.base import Undefined
def convert_fwhm_s_to_sigma_volumes(tr,lowpass_fwhm=None, highpass_fwhm=None):
    lowpass_sigma = lowpass_fwhm / float(tr) / 2 if lowpass_fwhm else Undefined
    highpass_sigma = highpass_fwhm/float(tr) / 2 if highpass_fwhm else Undefined
    return lowpass_sigma, highpass_sigma

class CFMMTemporalFilterPhysical(Workflow):
    group_name = 'Temporal Filter'
    flag_prefix = 'tf_'

    def _add_parameters(self):
        self._add_parameter('tr',
                            help='scan repetition time in seconds',
                            required = True)
        self._add_parameter('lowpass_fwhm',
                            default=-1,
                            help='lowpass filter fwhm in seconds',
                            )
        self._add_parameter('highpass_fwhm',
                            default=-1,
                            help='highpass filter fwhm in seconds',)

        dummy = bptf(exclude_list=self.parameters_to_calculate)
        for k,v in dummy._parameters.items():
            if k == 'output_type':
                # when the trait "output_type" had default type <undefined>. When this is the value of the trait, the
                # interface automatically populates NIFTI_GZ. However, if an <undefined> value is given via a nipype
                # connection, then the interface literally tries to find <undefined> as an output type and it doesn't
                # exist.  Since we are setting the value via a pipeline connection, we manually make the default value
                # NIFTI_GZ.
                v.add_argument_inputs['default'] = "'NIFTI_GZ'"
            self._add_parameter(k,**v.add_argument_inputs)

    def __init__(self, *args, **kwargs):
        self.parameters_to_calculate = ['lowpass_sigma', 'highpass_sigma']
        super().__init__(*args, **kwargs)
        exclude_list = list(bptf()._parameters.keys())
        self.tf = bptf(owner=self, exclude_list=exclude_list)
        self.outputs=['out_file']

    def create_workflow(self):
        inputnode,outputnode,wf = self.get_io_and_workflow()
        convert_units = get_fn_node(convert_fwhm_s_to_sigma_volumes,
                                    ['lowpass_sigma', 'highpass_sigma'],
                                    name='convert_units',
                                    imports=['from nipype.interfaces.base import Undefined'])
        tf = self.tf.get_node(name='tf')
        for parameter in self.tf.exclude_list:
            if parameter not in self.parameters_to_calculate:
                wf.connect(inputnode,parameter,tf,parameter)
        wf.connect([
            (inputnode, convert_units, [('tr', 'tr'),
                                        ('lowpass_fwhm', 'lowpass_fwhm'),
                                        ('highpass_fwhm', 'highpass_fwhm')]),
            (convert_units,tf,[('lowpass_sigma','lowpass_sigma'),
                               ('highpass_sigma','highpass_sigma')]),
            (tf,outputnode,[('out_file','out_file')]),
        ])
        return wf



# # this method sets the value for lowpass_sigma and highpass_sigma during populate_parameters
# # it happens outside the pipeline
# # it does not allow users to pass in tr,or fwhm, through pipeline connections
# # those parameters are only available from the commandline
# class CFMMTemporalFilter2(Interface):
#     # uses fslmaths -bptf
#     group_name = 'fslmaths -bptf'
#     flag_prefix = 'tf_'
#
#     def _add_parameters(self):
#         super()._add_parameters()
#         # hide the command line parameters that are based on volumes
#         self._modify_parameter('highpass_sigma', 'help', argparse.SUPPRESS)
#         self._modify_parameter('lowpass_sigma', 'help', argparse.SUPPRESS)
#
#         # add parameters for filtering based on physical units
#         self._add_parameter('tr',
#                             help='scan repetition time in seconds',
#                             required = True,
#                             )
#         self._add_parameter('highpass_fwhm',
#                             help='highpass filter fwhm in seconds',)
#         self._add_parameter('lowpass_fwhm',
#                             help='lowpass filter fwhm in seconds',)
#
#     def populate_parameters(self, parsed_args_dict):
#         super().populate_parameters(parsed_args_dict)
#         # use custom physical units parameters to calculate parameters based on volumes
#         # note: sigma = hwhm = fwhm/2 and sigma_in_volumes = fwhm_in_s / tr / 2
#         tr = self.get_parameter('tr').user_value
#         if self.get_parameter('highpass_fwhm_s').user_value:
#             self.get_parameter('highpass_sigma').user_value = self.get_parameter('highpass_fwhm_s').user_value / float(
#                 tr) / 2
#         if self.get_parameter('lowpass_fwhm_s').user_value:
#             self.get_parameter('lowpass_sigma').user_value = self.get_parameter('lowpass_fwhm_s').user_value / float(
#                 tr) / 2
#         # remove parameters so get_interface does not pass them to the interface
#         self._parameters.pop('tr')
#         self._parameters.pop('highpass_fwhm_s')
#         self._parameters.pop('lowpass_fwhm_s')
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(TemporalFilter, *args, **kwargs)



class CFMMSUSAN(Interface):
    group_name = 'SUSAN'
    flag_prefix = 'susan_'
    def __init__(self, *args, **kwargs):
        super().__init__(SUSAN, *args, **kwargs)
    def _add_parameters(self):
        super()._add_parameters()
        #this is if we want the default to be in-plane smoothing
        #self._modify_parameter('dimension', 'default', 2)
        try:
            help_str = self.get_parameter('usans').add_argument_inputs['help'].replace('%', ' percent')
        except:
            help_str = ''
        self._modify_parameter('usans','help',help_str)


def calculate_bt(bt=None, USAN_image=None,USAN_image_mask=None, bt_percentile=50, bt_fraction_of_percentile=1.0):
    if bt is None:
        assert USAN_image is not None
        import nibabel as nib
        import numpy as np
        usan_img_obj = nib.load(USAN_image)
        usan_img = usan_img_obj.get_data()
        if USAN_image_mask is not None:
            usan_img_mask_obj = nib.load(USAN_image_mask)

            # apply 90, 180, 270 deg rotations to align mask_qform and image_qform
            usan_img_qform = usan_img_obj.get_affine()[:3, :3]
            usan_img_mask_qform = usan_img_mask_obj.get_affine()[:3, :3]
            # R - rotation matrix, I - image, m - mask, d - data
            # Rm Im = Rd Id
            # [Rd^-1 Rm] Im = Id
            rot = np.dot(np.linalg.inv(usan_img_qform), usan_img_mask_qform)
            transpose_indices = np.abs(rot).argmax(axis=0)
            flip_indices = np.where(rot[transpose_indices, range(3)] < 0)
            usan_img_mask = usan_img_mask_obj.get_data().astype('bool')
            usan_img_mask = np.flip(usan_img_mask, flip_indices)
            usan_img_mask = usan_img_mask.transpose(transpose_indices)

            usan_img = usan_img[usan_img_mask]
        return bt_fraction_of_percentile*np.percentile(usan_img,bt_percentile)
    else:
        return bt
from cfmm.CFMMCommon import get_node_inputs_to_list
class CFMMSpatialSmoothing(Workflow):
    group_name = 'SUSAN'
    flag_prefix = 'smooth_'
    def _add_parameters(self):
        self._add_parameter('USAN_image',
                            help='Image to derive the USAN from.',
                            )
        self._add_parameter('USAN_image_mask',
                            help='Used to mask the calculation of the brightness threshold percentile.'
                            )
        self._add_parameter('bt_percentile',
                            default=50,
                            help='Set the brightness threshold using this percentile from the USAN image.',
                            )
        self._add_parameter('bt_fraction_of_percentile',
                            default=1.0,
                            help='Adjust the brightness threshold using this fraction of the percentile value.',)

        self._add_parameter('bt',
                            help=f"Explicitly set the USAN brightness threshold.  Should be greater "
                                 f"than noise level and less than contrast of edges to be preserved. Overrides "
                                 f"--{self.get_parameter('bt_percentile').flagname} and "
                                 f"--{self.get_parameter('bt_fraction_of_percentile').flagname}"
                            )
        dummy = CFMMSUSAN(exclude_list=self.parameters_to_calculate)
        for k,v in dummy._parameters.items():
            if k == 'output_type':
                v.add_argument_inputs['default'] = "'NIFTI_GZ'"
            self._add_parameter(k,**v.add_argument_inputs)


    def __init__(self, *args, **kwargs):
        self.parameters_to_calculate = ['brightness_threshold', 'usans']
        super().__init__(*args, **kwargs)
        exclude_list = list(CFMMSUSAN()._parameters.keys())
        self.susan = CFMMSUSAN(owner=self, exclude_list=exclude_list)
        self.outputs=['smoothed_file']

    def create_workflow(self):
        inputnode,outputnode,wf = self.get_io_and_workflow()

        bt = get_fn_node(calculate_bt,['bt'],name='bt')
        pack_usan = get_node_inputs_to_list(name='pack_usan')
        def _list_to_tuple_list(mylist):
            return [tuple(mylist)]
        susan = self.susan.get_node(name='susan')
        susan.inputs.brightness_threshold = 0.0

        for parameter in self.susan.exclude_list:
            if parameter not in self.parameters_to_calculate:
                wf.connect(inputnode,parameter,susan,parameter)
        wf.connect([
            (inputnode, bt, [('bt', 'bt'),
                             ('USAN_image', 'USAN_image'),
                             ('USAN_image_mask', 'USAN_image_mask'),
                             ('bt_percentile', 'bt_percentile'),
                             ('bt_fraction_of_percentile', 'bt_fraction_of_percentile')]),
            (inputnode,pack_usan,[('USAN_image','input1')]),
            (bt,pack_usan,[('bt','input2')]),
            (pack_usan, susan, [(('return_list',_list_to_tuple_list), 'usans')]),
            (susan,outputnode,[('smoothed_file','smoothed_file')]),
        ])
        return wf


# FSL Merge doesn't work with a large number of volumes (the shell does not allow large list inputs to commands)
# Therefore, we do the merge in python instead
# currenlty, we only implement merge in the temporal dimension
import nibabel as nib
import numpy as np
from nipype.interfaces.base import (
    CommandLineInputSpec,
    BaseInterface,
    TraitedSpec,
    File,
    InputMultiPath,
    traits,
)
import os
class MergeLargeInputSpec(CommandLineInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True, desc="Individual volumes to merge")
    tr = traits.Float(1.0, desc='TR of bold scan', mandatory=False, usedefault=True)
class MergeLargeOutputSpec(TraitedSpec):
    merged_file = File(desc="File", exists=True)
class MergeLarge(BaseInterface):
    input_spec = MergeLargeInputSpec
    output_spec = MergeLargeOutputSpec
    def _run_interface(self, runtime):
        in_files = self.inputs.in_files
        output_filename = self._list_outputs()['merged_file']
        tr = self.inputs.tr
        num_vols = len(in_files)
        first_vol = nib.load(in_files[0])
        first_vol_data = first_vol.get_data()
        first_vol_header = first_vol.header.copy()
        first_vol_header['pixdim'][4] = tr
        merge_shape = list(first_vol_data.shape)
        if len(merge_shape) < 4:
            merge_shape.append(num_vols)
        else:
            merge_shape[3] = num_vols
        merged_img = np.empty(merge_shape, dtype=first_vol_data.dtype)

        if len(first_vol_data.shape) < 4:
            merged_img[..., 0] = first_vol_data
        else:
            merged_img[..., 0:1] = first_vol_data

        for index in range(1, num_vols):
            data = nib.load(in_files[index]).get_data()
            if len(data.shape) < 4:
                merged_img[..., index] = data
            else:
                merged_img[..., index:index + 1] = data

        nifti_image = nib.Nifti1Image(merged_img, None, first_vol_header)
        nifti_image.to_filename(output_filename)
        return runtime
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['merged_file'] = os.path.abspath('merged.nii.gz')
        return outputs

if __name__ == '__main__':

    from cfmm.workflow import ParserGroups
    import configargparse


    cmd = [
        '--in_file',"'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/MouseFuncPreprocessingBIDS/sub-Nl311f9/ses-2020021001/func/sub-Nl311f9_ses-2020021001_task-rs_run-2_desc-Preproc_bold.nii.gz'",
        '--USAN_image',"'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/sub-Nl311f9/ses-2020021001/func/sub-Nl311f9_ses-2020021001_task-rs_run-01_bold.nii.gz'",
        '--fwhm',"1",
        '--dimension','2',
    ]

    parser_groups = ParserGroups(configargparse.ArgumentParser())
    tmp = CFMMSpatialSmoothing()
    tmp.populate_parser(parser_groups)
    parser_groups.parser.print_help()
    par_dict = vars(parser_groups.parser.parse_args(cmd))
    tmp.populate_parameters(par_dict)
    tmp2 = tmp.create_workflow()
    result = tmp2.run()


    if 0:
        cmd = [
            '--in_file',
            "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/sub-Nl311f9/ses-2020021001/func/sub-Nl311f9_ses-2020021001_task-rs_run-01_bold.nii.gz'",
            #'--lowpass_fwhm','3',
            '--highpass_fwhm', '100',
            '--tr','1.5'
        ]



        parser_groups = CFMMParserGroups(configargparse.ArgumentParser())
        tmp = CFMMTemporalFilterPhysical()
        tmp.populate_parser(parser_groups)
        parser_groups.parser.print_help()
        par_dict = vars(parser_groups.parser.parse_args(cmd))
        tmp.populate_parameters(par_dict)
        tmp2 = tmp.create_workflow()
        result = tmp2.run()


    if 0:
        import configargparse
        from cfmm.commandline.parameter_group import ParserGroups

        parser_groups = ParserGroups(configargparse.ArgumentParser())
        tmp = CFMMSpatialSmoothing()
        tmp.populate_parser(parser_groups)

        parser_groups.parser.print_help()
