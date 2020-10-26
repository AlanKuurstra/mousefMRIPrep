import nibabel as nib
import numpy as np
from workflows.CFMMCommon import get_fn_node
import pickle
import scipy.io as sio
import os
from nipype.interfaces.base import (
    CommandLineInputSpec,
    BaseInterface,
    TraitedSpec,
    File,
    traits,
)
from workflows.CFMMLogging import NipypeLogger as logger
from workflows.CFMMInterface import CFMMInterface

def read_label_mapping_file(label_mapping_file, downsample_mapping=None):
    import re
    labels_dict = {}
    with open(label_mapping_file, 'r', encoding='utf-8-sig') as f:
        for line in f.readlines():
            line=re.split('[ \t,;]+',line.rstrip())
            if (len(line) == 0):
                continue
            elif (len(line) % 3) != 0:
                assert "incorrect line"
            label_name = line[0]
            atlas_label = line[1]
            label_int = int(line[2])
            if downsample_mapping:
                atlas_label = downsample_mapping[atlas_label]
            labels_dict.setdefault(label_name, []).append((atlas_label, label_int))
    return labels_dict

def extract_label_mean_signal(fmri_volume, label_mapping):
    print(f'processing {fmri_volume}')
    fmri_volume = nib.load(fmri_volume).get_data()

    atlas_cache_dict = {}
    for atlas_list in label_mapping.values():
        for atlas,_ in atlas_list:
            if atlas not in atlas_cache_dict.keys():
                atlas_cache_dict[atlas] = nib.load(atlas).get_data()

    label_mean_signal_dict = {}
    for label_name, atlas_label_integer_pairs in label_mapping.items():
        # in numpy masked arrays, True means invalid
        mask = True
        for atlas_label, label_int in atlas_label_integer_pairs:
            mask &= (atlas_cache_dict[atlas_label] != int(label_int))
        mask = np.broadcast_to(mask[...,None],fmri_volume.shape)
        if mask.sum() == mask.size:
            logger.warning(f'Label {label_name} contains no voxels. Skipping.')
            continue
        volume_masked = np.ma.MaskedArray(data=fmri_volume, mask=mask)
        # specify dtype because volumes are often float32, but to avoid rounding errors we do calculations as float64
        label_mean_signal_dict[label_name] = volume_masked.mean(dtype='float64',axis=(0,1,2))
        # if the mask has 0 voxels for some reason, then mean() returns a masked array with data=0 mask=True
        if label_mean_signal_dict[label_name].mask.all():
            logger.warning(f'Label {label_name} contains no voxels. Extracted signal is NaN.')
            label_mean_signal_dict[label_name] = [np.NaN] * fmri_volume.shape[-1]
        else:
            label_mean_signal_dict[label_name] = label_mean_signal_dict[label_name].data

    names = ['label', 'avg_signal']
    formats = ['U50', np.ndarray]
    dtype = dict(names=names, formats=formats)
    label_signals = np.array(list(label_mean_signal_dict.items()), dtype=dtype)
    return label_signals


class ExractLabelMeansInputSpec(CommandLineInputSpec):
    fmri_volume = File(desc="List of volumes, typically an fmri split across time.", exists=True,mandatory=True)
    label_mapping = traits.Dict(traits.String, desc='xxx', exists=True,mandatory=True)
    output_name_without_extension = File('label_signals', desc="File", mandatory=False, usedefault=True)

class ExractLabelMeansOutputSpec(TraitedSpec):
    output_file_pkl = File(desc="Mean label signals stored in Python pickle file", exists=True)
    output_file_mat = File(desc="Mean label signals stored in Matlab .mat file", exists=True)

class ExractLabelMeans(BaseInterface):
    input_spec = ExractLabelMeansInputSpec
    output_spec = ExractLabelMeansOutputSpec

    def _run_interface(self, runtime):
        volume_loc = self.inputs.fmri_volume
        label_mapping = self.inputs.label_mapping
        output_file_pkl = self._list_outputs()['output_file_pkl']
        output_file_mat = self._list_outputs()['output_file_mat']

        label_signals = extract_label_mean_signal(volume_loc,label_mapping)

        with open(output_file_pkl, 'wb') as f:
            pickle.dump([label_signals,], f)

        sio.savemat(output_file_mat,
                    {
                        'label_signals': label_signals,
                    })

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()

        output_name_without_extension = self.inputs.output_name_without_extension
        outputs['output_file_pkl'] = os.path.abspath(output_name_without_extension + '.pkl')
        outputs['output_file_mat'] = os.path.abspath(output_name_without_extension + '.mat')
        return outputs



def get_node_read_label_mapping_file(name='read_label_mapping_file'):
    return get_fn_node(read_label_mapping_file,['label_mapping'],name=name)

# class CFMMExractLabelMeans(CFMMInterface):
#     group_name = 'Extract Label Mean Signal'
#     flag_prefix = 'sig_'
#     def __init__(self, *args, **kwargs):
#         super().__init__(ExractLabelMeans, *args, **kwargs)


if __name__ == '__main__':
    label_mapping_file = '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/Atlases/label_mapping_host.txt'
    # should downsample mapping come from the node copy or the derivative copy of the downsampled atlas???
    downsample_mapping={'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/Atlases/sub-AMBMCc57bl6_desc-BasalgangliaLabels15um.nii.gz': '/softdev/akuurstr/python/modules/mousefMRIPrep/workflows/test_downsample/DownsampleAtlasBIDS_workdir/DownsampleAtlasBIDS/b12a1931a8411401e5ce6a5c4f8eb8b400f8e828/downsample_atlas/sub-AMBMCc57bl6_desc-BasalgangliaLabels15um_bin_downsampled.nii.gz', '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/Atlases/sub-AMBMCc57bl6_desc-CerebellumLabels15um.nii.gz': '/softdev/akuurstr/python/modules/mousefMRIPrep/workflows/test_downsample/DownsampleAtlasBIDS_workdir/DownsampleAtlasBIDS/b12a1931a8411401e5ce6a5c4f8eb8b400f8e828/downsample_atlas/sub-AMBMCc57bl6_desc-CerebellumLabels15um_bin_downsampled.nii.gz', '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/Atlases/sub-AMBMCc57bl6_desc-CortexLabels15um.nii.gz': '/softdev/akuurstr/python/modules/mousefMRIPrep/workflows/test_downsample/DownsampleAtlasBIDS_workdir/DownsampleAtlasBIDS/b12a1931a8411401e5ce6a5c4f8eb8b400f8e828/downsample_atlas/sub-AMBMCc57bl6_desc-CortexLabels15um_bin_downsampled.nii.gz', '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives/Atlases/sub-AMBMCc57bl6_desc-HippocampusLabels15um.nii.gz': '/softdev/akuurstr/python/modules/mousefMRIPrep/workflows/test_downsample/DownsampleAtlasBIDS_workdir/DownsampleAtlasBIDS/b12a1931a8411401e5ce6a5c4f8eb8b400f8e828/downsample_atlas/sub-AMBMCc57bl6_desc-HippocampusLabels15um_bin_downsampled.nii.gz'}
    label_mapping = read_label_mapping_file(label_mapping_file, downsample_mapping)
    fmri_volume = '/softdev/akuurstr/python/modules/mousefMRIPrep/workflows/func_reg_test/MouseFuncToAtlas_workdir/MouseFuncToAtlas/register_func_to_atlas/sub-Nl311f9_ses-2020021001_task-rs_run-02_bold_tshift_xform_trans.nii.gz'

    # label_signals = extract_label_mean_signal(fmri_volume,label_mapping)
    # import matplotlib.pyplot as plt
    # for label,signal in label_average_dict.items():
    #     plt.figure()
    #     plt.title(label)
    #     plt.plot(signal)
    #     plt.show(block=True)

    tmp = ExractLabelMeans()
    tmp.inputs.fmri_volume = fmri_volume
    tmp.inputs.label_mapping = label_mapping
    result = tmp.run()
