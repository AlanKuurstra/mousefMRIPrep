# apps for container
# tar2bids
# fix orientation and slicetiming

# create initial masks (either brainsuite or with model and probability mask)
# create model and probability mask (either with create brainsuite initial masks or with manual masks).
# downsample atlas to the same resolution as functional

# we should do something different with atlas downsampling if hgihres atlas not same res as highres labels - need to make
# sure the downsampled atlas is shifted the correct way.


from workflows.CFMMWorkflow import CFMMWorkflow
from workflows.MouseFuncToAtlas import MouseFuncToAtlas, MouseFuncToAtlasBIDS
from nipype_interfaces.ComputeCorrelationMatrix import CFMMComputeCorrelationMatrix
from nipype_interfaces.ExtractLabels import ExractLabelMeans, get_node_read_label_mapping_file
import nipype.pipeline.engine as pe
from workflows.CFMMLogging import NipypeLogger as logger
from workflows.CFMMBIDS import CFMMBIDSWorkflowMixin, BIDSInputExternalSearch, CMDLINE_VALUE
from workflows.MouseAnatToAtlas import MouseAnatToAtlasBIDS
from workflows.CFMMCommon import delistify
from bids.layout.layout import parse_file_entities


class MouseCorrelationMatrixBIDS(CFMMWorkflow, CFMMBIDSWorkflowMixin):
    group_name = 'Correlation Matrix'
    flag_prefix = 'corr_'

    def _add_parameters(self):
        # how can we get the same help as the children?
        self._add_parameter('func', help='', iterable=True)
        self._add_parameter('func_mask', help='', iterable=True)
        self._add_parameter('anat', iterable=True)
        self._add_parameter('anat_mask', iterable=True)
        self._add_parameter('label_mapping', help='', required=True, )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func2atlas = MouseFuncToAtlasBIDS(owner=self, exclude_list=['func', 'func_mask', 'anat', 'anat_mask'])
        self.corr_mtx = CFMMComputeCorrelationMatrix(owner=self)
        self.outputs = ['label_signals_mat',
                        'label_signals_pkl',
                        'corr_mtx_pkl',
                        'corr_mtx_mat',
                        'corr_mtx_png',
                        'corr_mtx_shift_png',
                        ]

        self.add_bids_parameter_group()
        self.bids._modify_parameter('analysis_level', 'choices', ['participant'])

        self.func_bids = BIDSInputExternalSearch(self,
                                                 'func',
                                                 entities_to_overwrite={'subject': CMDLINE_VALUE,
                                                                        'session': CMDLINE_VALUE,
                                                                        'run': CMDLINE_VALUE,
                                                                        'extension': ['.nii', '.nii.gz'],
                                                                        },
                                                 output_derivatives={
                                                     'label_signals_mat': 'LabelSignals',
                                                     'label_signals_pkl': 'LabelSignals',
                                                     'corr_mtx_pkl': 'CorrelationMatrix',
                                                     'corr_mtx_mat': 'CorrelationMatrix',
                                                     'corr_mtx_png': 'CorrelationMatrix',
                                                     'corr_mtx_shift_png': 'CorrelationShiftMatrix',
                                                 }
                                                 )

        self.func_mask_bids = BIDSInputExternalSearch(self,
                                                      'func_mask',
                                                      dependent_search=self.func_bids,
                                                      dependent_entities=['subject', 'session', 'run'],
                                                      create_base_bids_string=False,
                                                      entities_to_overwrite={
                                                          'desc': CMDLINE_VALUE,
                                                          'extension': ['.nii', '.nii.gz'],
                                                      },
                                                      )
        self._modify_parameter('func_mask_desc', 'default', "'ManualBrainMask'")

        self.anat_bids = BIDSInputExternalSearch(self,
                                                 'anat',
                                                 dependent_search=self.func_bids,
                                                 dependent_entities=['subject','session'],
                                                 entities_to_overwrite={
                                                     'run': CMDLINE_VALUE,
                                                     'extension': ['.nii', '.nii.gz'],
                                                     'scope': 'self',
                                                 },
                                                 )

        self.anat_mask_bids = BIDSInputExternalSearch(self,
                                                      'anat_mask',
                                                      dependent_search=self.anat_bids,
                                                      dependent_entities=['subject', 'session', 'run'],
                                                      create_base_bids_string=False,
                                                      entities_to_overwrite={
                                                          'desc': CMDLINE_VALUE,
                                                          'extension': ['.nii', '.nii.gz'],
                                                      },
                                                      )
        self._modify_parameter('anat_mask_desc', 'default', "'ManualBrainMask'")

    def create_workflow(self):
        # correlation wf
        func2atlas_wf = self.func2atlas.create_workflow()
        read_label_mapping = get_node_read_label_mapping_file(name='read_label_mapping')
        extract_label_means = pe.Node(interface=ExractLabelMeans(), name='extract_label_means')
        compute_corr_mtx = self.corr_mtx.get_node(name='compute_corr_mtx')
        inputnode, outputnode, wf = self.get_io_and_workflow()
        wf.connect([
            # this bids wf does not have a non-bids super workflow.
            # we must connect both the func, and func_original_file
            (inputnode, func2atlas_wf, [('func', 'inputnode.func')]),
            (inputnode, func2atlas_wf, [('func_original_file', 'inputnode.func_original_file')]),
            (inputnode, func2atlas_wf, [('func_mask', 'inputnode.func_mask')]),
            (inputnode, func2atlas_wf, [('func_mask_original_file', 'inputnode.func_mask_original_file')]),
            (inputnode, func2atlas_wf, [('anat', 'inputnode.anat')]),
            (inputnode, func2atlas_wf, [('anat_original_file', 'inputnode.anat_original_file')]),
            (inputnode, func2atlas_wf, [('anat_mask', 'inputnode.anat_mask')]),
            (inputnode, func2atlas_wf, [('anat_mask_original_file', 'inputnode.anat_mask_original_file')]),

            (inputnode, read_label_mapping, [('label_mapping', 'label_mapping_file')]),
            (func2atlas_wf, extract_label_means, [('outputnode.func_to_atlas', 'fmri_volume')]),
            (read_label_mapping, extract_label_means, [('label_mapping', 'label_mapping')]),
            (extract_label_means, outputnode, [('output_file_pkl', 'label_signals_pkl')]),
            (extract_label_means, outputnode, [('output_file_mat', 'label_signals_mat')]),
            (extract_label_means, compute_corr_mtx, [('output_file_pkl', 'label_signals_pkl')]),
            (compute_corr_mtx, outputnode, [('output_file_pkl', 'corr_mtx_pkl')]),
            (compute_corr_mtx, outputnode, [('output_file_mat', 'corr_mtx_mat')]),
            (compute_corr_mtx, outputnode, [('output_file_png', 'corr_mtx_png')]),
            (compute_corr_mtx, outputnode, [('output_file_shift_png', 'corr_mtx_shift_png')]),
        ])
        self.add_bids_to_workflow(wf)

        return wf
