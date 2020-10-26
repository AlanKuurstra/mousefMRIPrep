import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import scipy.io as sio
from nipype.interfaces.base import isdefined
import os
from nipype.interfaces.base import (
    CommandLineInputSpec,
    BaseInterface,
    TraitedSpec,
    File,
    traits,
)
from workflows.CFMMInterface import CFMMInterface
from workflows.CFMMLogging import NipypeLogger as logger

def comput_correlation_matrix(label_signals, shift_interval_s=None, max_shift_s=None, tr=1, search_for_neg_corr=False):
    if shift_interval_s is None:
        shift_interval_s = tr
    if max_shift_s is None:
        max_shift_s = 2*tr

    num_labels = len(label_signals['label'])
    num_timepoints = label_signals['avg_signal'][0].size

    original_index = np.arange(num_timepoints) * tr
    if max_shift_s > original_index[-2]:
        raise Exception(f'max shift of {max_shift_s} is too large. Mus be <= {original_index[-2]}')
    if max_shift_s < shift_interval_s:
        print('WARNING: shift interval > max shift. No shifts will be performed.')

    shifts_s = np.arange(0, max_shift_s + 10 * np.finfo(float).eps, shift_interval_s)
    shifts_s = np.concatenate((-shifts_s[:0:-1], shifts_s))

    correlation_mtx = np.empty((num_labels, num_labels))
    shift_mtx = np.empty((num_labels, num_labels))

    # this is involved since we want to allow shifting the signal
    for row_indx in range(num_labels):
        row_signal = label_signals['avg_signal'][row_indx]
        for col_indx in range(num_labels):
            col_signal = label_signals['avg_signal'][col_indx]
            if search_for_neg_corr:
                max_corrcoef = 0
            else:
                max_corrcoef = -np.inf
            max_corrcoef_shift = None

            for shift_s in shifts_s:
                shifted_index = original_index + shift_s
                begin_valid_boundary = (shifted_index < 0).sum()
                end_valid_boundary = original_index.size - (shifted_index > original_index[-1]).sum()
                row_signal_resampled = np.interp(shifted_index[begin_valid_boundary:end_valid_boundary],
                                                 original_index, row_signal)
                current_corrcoef = \
                np.corrcoef(row_signal_resampled, col_signal[begin_valid_boundary:end_valid_boundary])[0, 1]
                if search_for_neg_corr:
                    if np.abs(current_corrcoef) > np.abs(max_corrcoef):
                        max_corrcoef = current_corrcoef
                        max_corrcoef_shift = shift_s
                else:
                    if current_corrcoef > max_corrcoef:
                        max_corrcoef = current_corrcoef
                        max_corrcoef_shift = shift_s

            correlation_mtx[row_indx, col_indx] = max_corrcoef
            shift_mtx[row_indx, col_indx] = max_corrcoef_shift
    return correlation_mtx, shift_mtx

class ComputeCorrelationMatrixInputSpec(CommandLineInputSpec):
    label_signals_pkl = File(desc='File with list of labels to extract.', exists=True,mandatory=True)
    output_name_without_extension = File('correlation_matrix', desc="File", mandatory=False, usedefault=True)
    shift_interval_s = traits.Float(desc='Resolution of time shifts in seconds.', mandatory=False)
    max_shift_s = traits.Float(desc='Max allowed shift (in seconds) when searching for best correlation.', mandatory=False)
    tr = traits.Float(1.0, desc='Temporal resolution', mandatory=False, usedefault=True)
    search_for_neg_corr = traits.Bool(False, desc='If False, only searches for time shift with largest positive correlation. If True, searches for time shift with largest positive or negative correlation.', mandatory=False, usedefault=True)

class ComputeCorrelationMatrixOutputSpec(TraitedSpec):
    output_file_pkl = File(desc="Correlation matrix and shifts stored in Python pickle file. Negative shift means row signal was shifted to occur earlier.", exists=True)
    output_file_mat = File(desc="Correlation matrix and shifts stored in Matlab .mat file", exists=True)
    output_file_png = File(desc="Visualization of correlation matrix", exists=True)
    output_file_shift_png = File(desc="Visualization of correlation matrix shifts in seconds", exists=True)


class ComputeCorrelationMatrix(BaseInterface):
    input_spec = ComputeCorrelationMatrixInputSpec
    output_spec = ComputeCorrelationMatrixOutputSpec

    def _run_interface(self, runtime):
        label_signals_pkl = self.inputs.label_signals_pkl
        shift_interval_s = self.inputs.shift_interval_s
        if not isdefined(shift_interval_s):
            shift_interval_s = None
        max_shift_s = self.inputs.max_shift_s
        if not isdefined(max_shift_s):
            max_shift_s = None
        else:
            max_shift_s = np.abs(max_shift_s)
        tr = self.inputs.tr
        search_for_neg_corr = self.inputs.search_for_neg_corr

        output_file_pkl = self._list_outputs()['output_file_pkl']
        output_file_mat = self._list_outputs()['output_file_mat']
        output_file_png = self._list_outputs()['output_file_png']
        output_file_shift_png = self._list_outputs()['output_file_shift_png']

        with open(label_signals_pkl, 'rb') as f:
            label_signals = pickle.load(f)[0]

        correlation_mtx, shift_mtx = comput_correlation_matrix(label_signals, shift_interval_s=shift_interval_s,
                                                               max_shift_s=max_shift_s, tr=tr,
                                                               search_for_neg_corr=search_for_neg_corr)
        num_labels = correlation_mtx.shape[0]

        ax = sns.heatmap(
            correlation_mtx,
            xticklabels=label_signals['label'],
            yticklabels=label_signals['label'],
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        )
        fig = plt.gcf()
        fig.set_dpi(100)
        legend_size = .5 + num_labels * 0.1
        label_text_size = 3
        inch_per_label = 0.4
        fig.set_size_inches(num_labels * inch_per_label + label_text_size + legend_size,
                            num_labels * inch_per_label + label_text_size)
        plt.tight_layout()
        plt.savefig(output_file_png)

        plt.close()
        if max_shift_s:
            vmax = max_shift_s
        else:
            vmax = 2*tr
        ax = sns.heatmap(
            shift_mtx,
            xticklabels=label_signals['label'],
            yticklabels=label_signals['label'],
            vmin=-vmax, vmax=vmax, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        )
        fig = plt.gcf()
        fig.set_dpi(100)
        legend_size = .5 + num_labels * 0.1
        label_text_size = 3
        inch_per_label = 0.4
        fig.set_size_inches(num_labels * inch_per_label + label_text_size + legend_size,
                            num_labels * inch_per_label + label_text_size)
        plt.tight_layout()
        plt.savefig(output_file_shift_png)
        plt.close()

        sio.savemat(output_file_mat,
                    {
                        'correlation_mtx': correlation_mtx,
                        'shift_mtx': shift_mtx,
                    })

        with open(output_file_pkl, 'wb') as f:
            pickle.dump([correlation_mtx, shift_mtx], f)

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()

        output_name_without_extension = self.inputs.output_name_without_extension
        outputs['output_file_pkl'] = os.path.abspath(output_name_without_extension + '.pkl')
        outputs['output_file_mat'] = os.path.abspath(output_name_without_extension + '.mat')
        outputs['output_file_png'] = os.path.abspath(output_name_without_extension + '.png')
        outputs['output_file_shift_png'] = os.path.abspath(output_name_without_extension + '_shift.png')

        return outputs

class CFMMComputeCorrelationMatrix(CFMMInterface):
    group_name = 'Compute Correlation Matrix'
    flag_prefix = 'corrmtx_'
    def __init__(self, *args, **kwargs):
        super().__init__(ComputeCorrelationMatrix, *args, **kwargs)

if __name__ == '__main__':
    from nipype_interfaces.ExtractLabels import read_label_mapping_file, extract_label_mean_signal, ExractLabelMeans

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
    # corr,shift = comput_correlation_matrix(label_signals)



    # tmp = ExractLabelMeans()
    # tmp.inputs.fmri_volume = fmri_volume
    # tmp.inputs.label_mapping = label_mapping
    # result = tmp.run()

    tmp2 = ComputeCorrelationMatrix()
    tmp2.inputs.label_signals_pkl = '/softdev/akuurstr/python/modules/mousefMRIPrep/nipype_interfaces/label_signals.pkl'
    result2 = tmp2.run()