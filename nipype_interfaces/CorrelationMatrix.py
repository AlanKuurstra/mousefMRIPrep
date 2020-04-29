import numpy as np
import nibabel as nib
import xml.etree.ElementTree as ET
import csv
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import scipy.io as sio
from tools.split_exts import split_exts
import os
import json

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
from nipype.pipeline import engine as pe
from nipype.interfaces.utility import Function
from nipype.interfaces import utility as niu



def read_label_mapping_file(label_mapping_file):
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
            label_atlas = line[1]
            label_int = int(line[2])

            labels_dict.setdefault(label_name, []).append((label_atlas, label_int))

    label_names = list(labels_dict.keys())
    label_atlases = [[atlas_int_pair[0] for atlas_int_pair in value] for value in labels_dict.values()]
    label_ints = [[atlas_int_pair[1] for atlas_int_pair in value] for value in labels_dict.values()]
    return label_names,label_atlases,label_ints

def get_read_label_mapping_file_node(name='read_label_mapping_file'):
    node = pe.Node(
        interface=Function(input_names=["label_mapping_file"], output_names=['label_names','label_atlases','label_ints'],
                           function=read_label_mapping_file), name=name)
    return node





class ExractLabelMeansInputSpec(CommandLineInputSpec):
    volume = File(desc="List of volumes, typically an fmri split across time.", exists=True,mandatory=True)
    label_name_list = traits.List(traits.String, desc='xxx', exists=True,mandatory=True)
    label_atlases_list =traits.List(InputMultiPath(desc="xxx"), exists=True, mandatory=True)
    label_ints_list = traits.List(traits.List(traits.Int,desc='xxx'), exists=True,mandatory=True)
    output_name = File('label_averages.json', desc="File", mandatory=False, usedefault=True)

class ExractLabelMeansOutputSpec(TraitedSpec):
    output_file_json = File(desc="Extracted mean label signals stored in json file", exists=True)

class ExractLabelMeans(BaseInterface):
    input_spec = ExractLabelMeansInputSpec
    output_spec = ExractLabelMeansOutputSpec

    def _run_interface(self, runtime):
        volume_loc = self.inputs.volume
        volume = nib.load(volume_loc).get_data()
        label_name_list = self.inputs.label_name_list
        label_atlases_list = self.inputs.label_atlases_list
        label_ints_list = self.inputs.label_ints_list
        output_file_json = self._list_outputs()['output_file_json']


        print(f'processing {volume_loc}')
        assert len(label_name_list)==len(label_atlases_list)==len(label_ints_list), "label_name_list, label_img_list and label_int_list must be the same length"

        label_atlas_set = set()
        for labels in label_atlases_list:
            for label in labels:
                label_atlas_set.add(label)

        atlas_cache_dict = {}
        for atlas_loc in label_atlas_set:
            atlas_cache_dict[atlas_loc] = nib.load(atlas_loc).get_data()

        label_average_dict = {}
        #label_average_dict = {'volume':volume_loc}
        for label_name,label_atlases,label_ints in zip(label_name_list,label_atlases_list,label_ints_list):
            # in numpy masked arrays, True means invalid
            mask = True
            for label_atlas, label_int in zip(label_atlases, label_ints):
                mask &= (atlas_cache_dict[label_atlas] != int(label_int))
            volume_masked = np.ma.MaskedArray(data=volume, mask=mask)
            # specify dtype because volumes are often float32, but to avoid rounding errors we do calculations as float64
            label_average_dict[label_name] = volume_masked.mean(dtype='float')
            # if the mask has 0 voxels for some reason, then the mean is a masked array with data=0 mask=True
            if label_average_dict[label_name] is np.ma.masked:
                #label_average_dict[label_name] = float(label_average_dict[label_name].data)
                label_average_dict[label_name] = np.NaN

        with open(output_file_json, 'w') as f:
            json.dump(label_average_dict, f, indent=4)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        output_name = self.inputs.output_name
        outputs['output_file_json'] = os.path.abspath(output_name)
        return outputs

class AssembleLabelSignalsInputSpec(CommandLineInputSpec):
    label_mean_json_files =InputMultiPath(desc="xxx", exists=True, mandatory=True)
    output_name_without_extension = File('label_signals', desc="File", mandatory=False, usedefault=True)

class AssembleLabelSignalsOutputSpec(TraitedSpec):
    output_file_pkl = File(desc="Assembled mean label signals stored in Python pickle file", exists=True)
    output_file_mat = File(desc="Assembled mean label signals stored in Matlab .mat file", exists=True)

class AssembleLabelSignals(BaseInterface):
    input_spec = AssembleLabelSignalsInputSpec
    output_spec = AssembleLabelSignalsOutputSpec

    def _run_interface(self, runtime):
        label_mean_json_files = self.inputs.label_mean_json_files
        output_file_pkl =  self._list_outputs()['output_file_pkl']
        output_file_mat =  self._list_outputs()['output_file_mat']


        label_signal_dict = {}
        for label_mean_json_file in label_mean_json_files:
            with open(label_mean_json_file,'r') as f:
                label_mean_dict = json.load(f)
            for name,value in label_mean_dict.items():
                label_signal_dict.setdefault(name, []).append(value)

        #convert dict to structured array so that we can label the columns

        for key in label_signal_dict.keys():
            label_signal_dict[key] = np.array(label_signal_dict[key])

        names = ['label', 'avg_signal']
        formats = ['U50', np.ndarray]
        dtype = dict(names=names, formats=formats)
        label_signals = np.array(list(label_signal_dict.items()), dtype=dtype)

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

def init_extract_label_means(name='extract_label_means',mem_gb_mapnode=3,nthreads_mapnode=1):
    wf = pe.Workflow(name)

    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'split_volumes_list',
        'label_file',
    ]), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['output_file_pkl',
                'output_file_mat',
                ]), name='outputnode')

    read_labels = get_read_label_mapping_file_node()
    extract_label_means = pe.MapNode(interface=ExractLabelMeans(), iterfield=['volume'], mem_gb=mem_gb_mapnode,n_procs=nthreads_mapnode,name='extract_label_means')
    assemble_signals = pe.Node(interface=AssembleLabelSignals(),name='assemble_signals')

    wf.connect([
        (inputnode, read_labels, [('label_file', 'label_mapping_file')]),
        (inputnode, extract_label_means, [('split_volumes_list', 'volume')]),
        (read_labels, extract_label_means, [('label_names', 'label_name_list')]),
        (read_labels, extract_label_means, [('label_atlases', 'label_atlases_list')]),
        (read_labels, extract_label_means, [('label_ints', 'label_ints_list')]),
        (extract_label_means, assemble_signals, [('output_file_json', 'label_mean_json_files')]),
        (assemble_signals, outputnode, [('output_file_pkl', 'output_file_pkl')]),
        (assemble_signals, outputnode, [('output_file_mat', 'output_file_mat')]),
    ])
    return wf

# class ExractLabelMeansInputSpec(CommandLineInputSpec):
#     split_volumes_list = InputMultiPath(desc="List of volumes, typically an fmri split across time.", exists=True, mandatory=True)
#     label_file = File(desc='File with list of labels to extract.', exists=True,mandatory=True)
#     output_name_without_extension = File('label_signals', desc="File", mandatory=False, usedefault=True)
#
# class ExractLabelMeansOutputSpec(TraitedSpec):
#     output_file_pkl = File(desc="Extracted mean label signals stored in Python pickle file", exists=True)
#     output_file_mat = File(desc="Extracted mean label signals stored in Matlab .mat file", exists=True)
#
#
# class ExractLabelMeans(BaseInterface):
#     input_spec = ExractLabelMeansInputSpec
#     output_spec = ExractLabelMeansOutputSpec
#
#     def _run_interface(self, runtime):
#         split_volumes_list = self.inputs.split_volumes_list
#         label_file = self.inputs.label_file
#         output_file_pkl =  self._list_outputs()['output_file_pkl']
#         output_file_mat =  self._list_outputs()['output_file_mat']
#
#         atlas_and_labels_dict = {}
#         with open(label_file, 'r', encoding='utf-8-sig') as f:
#             #csv_file = csv.reader(f, delimiter='\t')
#             #for line in csv_file:
#             for line in f.readlines():
#                 line=re.split('[ \t,;]+',line.rstrip())
#                 if (len(line) == 0):
#                     continue
#                 elif (len(line) % 3) != 0:
#                     assert "incorrect line"
#                 label_name = line[0]
#                 label_atlas = line[1]
#                 label_int = line[2]
#                 #instead of making the name unique, maybe we should take the union of labels with the same name
#                 atlas_string, _ = split_exts(os.path.basename(label_atlas))
#                 full_label_name = atlas_string + '_' + label_name
#                 #this would need to change to be { label_name: [list of (label_img,label_int) tuples] }
#                 atlas_and_labels_dict.setdefault(label_atlas, []).append((full_label_name, label_int))
#
#         atlas_cache_dict = {}
#         for atlas_loc in atlas_and_labels_dict.keys():
#             atlas_cache_dict[atlas_loc] = nib.load(atlas_loc).get_data()
#
#         # dictionary instead of structured array because some atlas labels have 0 pixels and we don't want to include
#         # 0 timeseries in the structured array. convert to structured array at the end for saving.
#         avg_signal_dict = {}
#         # multiprocess this over volumes yourself so that the labels can stay in a c shared array (rather than map node)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#         for volume_loc in split_volumes_list:
#             print(volume_loc)
#             volume = nib.load(volume_loc).get_data()
#             for label_img_loc, labels in atlas_and_labels_dict.items():
#                 # label_img = nib.load(label_img_loc).get_data()
#                 label_img = atlas_cache_dict[label_img_loc]
#                 for label_name, label_int in labels:
#                     print(label_name, label_int)
#                     mask = label_img == int(label_int)
#                     tmp = volume[mask]
#                     if tmp.shape[0] != 0:
#                         avg_signal_dict.setdefault(label_name, []).append(tmp.mean(axis=0))
#
#         for key in avg_signal_dict.keys():
#             avg_signal_dict[key] = np.array(avg_signal_dict[key])
#
#         # convert dict to structured array so that we can label the columns
#         names = ['label', 'avg_signal']
#         formats = ['U50', np.ndarray]
#         dtype = dict(names=names, formats=formats)
#         label_signals = np.array(list(avg_signal_dict.items()), dtype=dtype)
#
#         with open(output_file_pkl, 'wb') as f:
#             pickle.dump([label_signals,], f)
#
#         sio.savemat(output_file_mat,
#                     {
#                         'label_signals': label_signals,
#                     })
#         return runtime
#
#     def _list_outputs(self):
#         outputs = self.output_spec().get()
#
#         output_name_without_extension = self.inputs.output_name_without_extension
#         outputs['output_file_pkl'] = os.path.abspath(output_name_without_extension + '.pkl')
#         outputs['output_file_mat'] = os.path.abspath(output_name_without_extension + '.mat')
#         return outputs



class ComputeCorrelationMatrixInputSpec(CommandLineInputSpec):
    label_signals_pkl = File(desc='File with list of labels to extract.', exists=True,mandatory=True)
    output_name_without_extension = File('correlation_matrix', desc="File", mandatory=False, usedefault=True)
    shift_interval_s = traits.Float(1.0, desc='Resolution of time shifts in seconds.', mandatory=False, usedefault=True)
    max_shift_s = traits.Float(1.0, desc='Max allowed shift (in seconds) when searching for best correlation.', mandatory=False, usedefault=True)
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
        max_shift_s = np.abs(self.inputs.max_shift_s)
        tr = self.inputs.tr
        search_for_neg_corr = self.inputs.search_for_neg_corr

        output_file_pkl =  self._list_outputs()['output_file_pkl']
        output_file_mat =  self._list_outputs()['output_file_mat']
        output_file_png = self._list_outputs()['output_file_png']
        output_file_shift_png = self._list_outputs()['output_file_shift_png']


        with open(label_signals_pkl, 'rb') as f:
            label_signals = pickle.load(f)[0]

        num_labels = len(label_signals['label'])
        num_timepoints = label_signals['avg_signal'][0].size


        original_index = np.arange(num_timepoints) * tr
        if max_shift_s>original_index[-2]:
            raise Exception(f'max shift of {max_shift_s } is too large. Mus be <= {original_index[-2]}')
        if max_shift_s<shift_interval_s:
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
                    current_corrcoef = np.corrcoef(row_signal_resampled, col_signal[begin_valid_boundary:end_valid_boundary])[0,1]
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
        ax = sns.heatmap(
            shift_mtx,
            xticklabels=label_signals['label'],
            yticklabels=label_signals['label'],
            vmin=-max_shift_s, vmax=max_shift_s, center=0,
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



if __name__ == "__main__":
    if 0:
        host_atlas_location = '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/atlases'
        guest_atlas_location = '/atlases'
        label_list = {
            host_atlas_location+'/labels/AMBMC-c57bl6-basalganglia-labels-15um.nii.gz': host_atlas_location+'/labels/AMBMC_basalganglia_labels.xml',
            host_atlas_location+'/labels/AMBMC-c57bl6-cerebellum-labels-15um.nii.gz': host_atlas_location+'/labels/AMBMC_cerebellum_labels.xml',
            host_atlas_location+'/labels/AMBMC-c57bl6-cortex-labels-15um.nii.gz': host_atlas_location+'/labels/AMBMC_cortex_labels.xml',
            host_atlas_location+'/labels/AMBMC-c57bl6-hippocampus-labels-15um.nii.gz': host_atlas_location+'/labels/AMBMC_hippocampus_labels.xml'
        }

        # create label text file
        # label_img_loc / name / label_int

        with open('label_mapping_host.txt','w') as f:
            for label_img,label_int_to_name_mapping in list(label_list.items()):

                map_root = ET.parse(label_int_to_name_mapping).getroot()
                #print(map_root.getchildren())
                # print(map_root.findall('data/'))

                for label in map_root.findall('data/'):
                    label_text = label.text
                    label_int = label.get('index')
                    #write_line = '\t'.join((label_text, label_img, label_int)) + '\n'
                    write_line = "{:<20} {:<120} {:<5}\n".format(label_text, label_img,label_int)
                    print(write_line)
                    f.write(write_line)
        with open('label_mapping_guest.txt','w') as f:
            for label_img,label_int_to_name_mapping in list(label_list.items()):

                map_root = ET.parse(label_int_to_name_mapping).getroot()
                #print(map_root.getchildren())
                # print(map_root.findall('data/'))

                for label in map_root.findall('data/'):
                    label_text = label.text
                    label_int = label.get('index')
                    label_img_guest = label_img.replace(host_atlas_location,guest_atlas_location)
                    #write_line = '\t'.join((label_text, label_img_guest, label_int)) + '\n'
                    write_line = "{:<20} {:<65} {:<5}\n".format(label_text, label_img_guest,label_int)
                    print(write_line)
                    f.write(write_line)

    if 0:
        tmp = ExractLabelMeans()
        tmp.inputs.split_volumes_list = ['/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas0/vol0000_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas1/vol0001_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas2/vol0002_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas3/vol0003_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas4/vol0004_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas5/vol0005_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas6/vol0006_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas7/vol0007_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas8/vol0008_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas9/vol0009_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas10/vol0010_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas11/vol0011_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas12/vol0012_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas13/vol0013_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas14/vol0014_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas15/vol0015_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas16/vol0016_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas17/vol0017_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas18/vol0018_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas19/vol0019_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas20/vol0020_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas21/vol0021_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas22/vol0022_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas23/vol0023_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas24/vol0024_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas25/vol0025_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas26/vol0026_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas27/vol0027_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas28/vol0028_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas29/vol0029_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas30/vol0030_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas31/vol0031_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas32/vol0032_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas33/vol0033_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas34/vol0034_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas35/vol0035_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas36/vol0036_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas37/vol0037_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas38/vol0038_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas39/vol0039_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas40/vol0040_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas41/vol0041_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas42/vol0042_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas43/vol0043_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas44/vol0044_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas45/vol0045_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas46/vol0046_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas47/vol0047_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas48/vol0048_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas49/vol0049_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas50/vol0050_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas51/vol0051_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas52/vol0052_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas53/vol0053_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas54/vol0054_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas55/vol0055_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas56/vol0056_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas57/vol0057_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas58/vol0058_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas59/vol0059_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas60/vol0060_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas61/vol0061_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas62/vol0062_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas63/vol0063_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas64/vol0064_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas65/vol0065_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas66/vol0066_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas67/vol0067_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas68/vol0068_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas69/vol0069_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas70/vol0070_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas71/vol0071_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas72/vol0072_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas73/vol0073_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas74/vol0074_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas75/vol0075_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas76/vol0076_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas77/vol0077_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas78/vol0078_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas79/vol0079_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas80/vol0080_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas81/vol0081_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas82/vol0082_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas83/vol0083_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas84/vol0084_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas85/vol0085_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas86/vol0086_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas87/vol0087_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas88/vol0088_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas89/vol0089_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas90/vol0090_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas91/vol0091_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas92/vol0092_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas93/vol0093_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas94/vol0094_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas95/vol0095_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas96/vol0096_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas97/vol0097_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas98/vol0098_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas99/vol0099_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas100/vol0100_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas101/vol0101_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas102/vol0102_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas103/vol0103_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas104/vol0104_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas105/vol0105_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas106/vol0106_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas107/vol0107_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas108/vol0108_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas109/vol0109_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas110/vol0110_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas111/vol0111_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas112/vol0112_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas113/vol0113_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas114/vol0114_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas115/vol0115_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas116/vol0116_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas117/vol0117_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas118/vol0118_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas119/vol0119_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas120/vol0120_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas121/vol0121_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas122/vol0122_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas123/vol0123_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas124/vol0124_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas125/vol0125_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas126/vol0126_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas127/vol0127_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas128/vol0128_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas129/vol0129_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas130/vol0130_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas131/vol0131_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas132/vol0132_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas133/vol0133_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas134/vol0134_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas135/vol0135_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas136/vol0136_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas137/vol0137_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas138/vol0138_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas139/vol0139_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas140/vol0140_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas141/vol0141_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas142/vol0142_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas143/vol0143_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas144/vol0144_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas145/vol0145_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas146/vol0146_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas147/vol0147_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas148/vol0148_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas149/vol0149_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas150/vol0150_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas151/vol0151_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas152/vol0152_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas153/vol0153_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas154/vol0154_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas155/vol0155_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas156/vol0156_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas157/vol0157_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas158/vol0158_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas159/vol0159_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas160/vol0160_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas161/vol0161_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas162/vol0162_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas163/vol0163_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas164/vol0164_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas165/vol0165_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas166/vol0166_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas167/vol0167_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas168/vol0168_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas169/vol0169_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas170/vol0170_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas171/vol0171_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas172/vol0172_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas173/vol0173_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas174/vol0174_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas175/vol0175_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas176/vol0176_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas177/vol0177_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas178/vol0178_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas179/vol0179_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas180/vol0180_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas181/vol0181_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas182/vol0182_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas183/vol0183_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas184/vol0184_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas185/vol0185_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas186/vol0186_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas187/vol0187_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas188/vol0188_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas189/vol0189_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas190/vol0190_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas191/vol0191_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas192/vol0192_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas193/vol0193_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas194/vol0194_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas195/vol0195_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas196/vol0196_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas197/vol0197_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas198/vol0198_trans.nii.gz', '/storage/akuurstr/mouse_pipepline_output/BoldToAtlasRegistration/register_bold_to_atlas/mapflow/_register_bold_to_atlas199/vol0199_trans.nii.gz']
        tmp.inputs.label_file='label_mapping.txt'
        results = tmp.run()

        tmp = ComputeCorrelationMatrix()
        tmp.inputs.label_signals_pkl = '/softdev/akuurstr/python/modules/mousefMRIPrep/label_signals.pkl'
        tmp.inputs.shift_interval_s = .1
        tmp.inputs.max_shift_s = 3
        tmp.inputs.tr = 1.5
        tmp.inputs.search_for_neg_corr = True

        results = tmp.run()

        with open(results.outputs.output_file_pkl,'rb') as f:
            corr_mtx,shift_mtx = pickle.load(f)

        np.set_printoptions(linewidth=300)
        from pprint import pprint
        print(np.array2string(corr_mtx))
        print(np.array2string(shift_mtx))

    if 1:
        tmp = init_extract_label_means()
        tmp.inputs.inputnode.split_volumes_list = [
            '/storage/akuurstr/mouse_pipepline_output/mousefMRIPrep_scratch/func_processing/register_func_to_atlas/mapflow/_register_func_to_atlas5/warped.nii',
            '/storage/akuurstr/mouse_pipepline_output/mousefMRIPrep_scratch/func_processing/register_func_to_atlas/mapflow/_register_func_to_atlas6/warped.nii']
        tmp.inputs.inputnode.label_file = '/softdev/akuurstr/python/modules/mousefMRIPrep/examples/label_mapping_host.txt'
        tmp.inputs.inputnode.label_file = '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/atlases/label_mapping_host.txt'
        result = tmp.run()
        with open(list(result.nodes)[-1].get_output('output_file_pkl'), 'rb') as f:
            label_signals = pickle.load(f)

        tmp2 = ComputeCorrelationMatrix()
        tmp2.inputs.label_signals_pkl = list(result.nodes)[-1].get_output('output_file_pkl')
        tmp2.inputs.shift_interval_s = 1
        tmp2.inputs.max_shift_s = 0
        tmp2.inputs.tr = 1.5
        tmp2.inputs.search_for_neg_corr = True
        result2 = tmp2.run()
        
        
            
        







