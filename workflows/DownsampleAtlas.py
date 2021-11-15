from cfmm.workflow import Workflow
from nipype_interfaces.DownsampleAtlas import get_node_downsample_atlas
from cfmm.bids_parameters import BIDSWorkflowMixin, BIDSInputExternalSearch, CMDLINE_VALUE
from cfmm.CFMMCommon import get_fn_node


class DownsampleAtlas(Workflow):
    group_name = 'Downsample Atlas'
    flag_prefix = 'downsample_'

    def _add_parameters(self):
        self._add_parameter('atlas',
                            help='Atlas to downsample')

        self._add_parameter('label_images',
                            help=f'A list of label images to downsample.',
                            )

        self._add_parameter('target_voxel_sz',
                            help='A three tuple containing the desired voxel sizes for the downsampled atlas.')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs = ['downsampled_atlas', 'downsampled_labels', 'downsample_shift_transform_file']

    def create_workflow(self, arg_dict=None):
        downsample_node = get_node_downsample_atlas()
        inputnode, outputnode, wf = self.get_io_and_workflow()
        wf.connect([
            (inputnode, downsample_node, [('label_images', 'highres_label_list')]),
            (inputnode, downsample_node, [('atlas', 'highres_atlas')]),
            (inputnode, downsample_node, [('target_voxel_sz', 'target_voxel_sz')]),
            (downsample_node, outputnode, [('output_lowres_atlas_path', 'downsampled_atlas')]),
            (downsample_node, outputnode, [('shift_transform_file', 'downsample_shift_transform_file')]),
            (downsample_node, outputnode, [('lowres_labels', 'downsampled_labels')]),
        ])
        return wf


def resolution_desc(reference):
    import nibabel as nib
    ref = nib.load(reference)
    ref_header = ref.header
    res = 'x'.join(ref_header['pixdim'][1:4].astype(str)).replace('.', 'p') + ref_header.get_xyzt_units()[0]
    return f'Downsampled{res}'


def resolution_desc_set(references):
    descs = [resolution_desc(ref) for ref in references]
    descs_set = list(set(descs))
    descs_first_indices = [descs.index(desc) for desc in descs_set]
    references_set = [references[index] for index in descs_first_indices]
    return references_set, descs_set


def shift_desc(reference):
    from workflows.DownsampleAtlas import resolution_desc
    return resolution_desc(reference) + 'ShiftTransformation'


# output_names must be 'desc' for the bids helper function to connect it correctly
def get_node_dynamic_res_desc(name='dynamic_desc'):
    node = get_fn_node(resolution_desc, ['desc'], name=name)
    return node


def get_node_dynamic_shift_desc(name='dynamic_desc'):
    node = get_fn_node(shift_desc, ['desc'], name=name)
    return node


class DownsampleAtlasBIDS(DownsampleAtlas, BIDSWorkflowMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_bids_parameter_group()
        self.bids._modify_parameter('analysis_level', 'choices', ['participant'])

        res_desc_node = get_node_dynamic_res_desc('res_desc')
        shift_desc_node = get_node_dynamic_shift_desc('shift_desc')

        self.atlas_bids = BIDSInputExternalSearch(self,
                                                  'atlas',
                                                  create_base_bids_string=False,
                                                  entities_to_overwrite={'subject': CMDLINE_VALUE,
                                                                         'desc': CMDLINE_VALUE,
                                                                         'extension': ['.nii', '.nii.gz'],
                                                                         'regex_search': True,
                                                                         'scope': 'Atlases',
                                                                         },
                                                  # output_derivatives={'downsampled_atlas': res_desc_node,
                                                  #                     'downsample_shift_transform_file': shift_desc_node, }
                                                  output_derivatives={'downsampled_atlas': 'BinDownsampled',
                                                                      'downsample_shift_transform_file': 'BinDownsampleShift', }
                                                  )

        self.label_images_bids = BIDSInputExternalSearch(self,
                                                         'label_images',
                                                         create_base_bids_string=False,
                                                         entities_to_overwrite={'subject': CMDLINE_VALUE,
                                                                                'desc': CMDLINE_VALUE,
                                                                                'extension': ['.nii', '.nii.gz'],
                                                                                'regex_search': True,
                                                                                'scope': 'Atlases',
                                                                                },
                                                         # output_derivatives={'downsampled_labels': res_desc_node},
                                                         output_derivatives={'downsampled_labels': 'BinDownsampled'},
                                                         derivatives_mapnode=True,
                                                         )

    def create_workflow(self):
        wf = super().create_workflow()
        self.add_bids_to_workflow(wf)

        # these are all the nodes that need a pipeline connection to their input 'reference'
        # inputs_with_dynamic_description = ['res_desc', 'shift_desc']
        # inputnode = wf.get_node('inputnode')
        # self.connect_dynamic_derivatives_desc(wf, inputnode, 'reference', inputs_with_dynamic_description, 'reference')

        return wf


if __name__ == "__main__":
    cmd_args = [
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids'",
        "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives'",
        "'participant'",
        '--input_derivatives_dirs',
        "['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives']",
        '--bids_layout_db', "'./test_downsample/bids_database'",
        '--reset_db',

        # '--atlas',"'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/atlases/BIDS/sub-AMBMCc57bl6_desc-ModelDownsampled.nii.gz'",
        '--atlas_subject', "'^AMBMCc57bl6$'",
        '--atlas_desc', "'^Model$'",

        # '--label_images',
        # "['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/atlases/BIDS/sub-AMBMCc57bl6_desc-CortexLabels15um.nii.gz',"
        # "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/atlases/BIDS/sub-AMBMCc57bl6_desc-CerebellumLabels15um.nii.gz']",
        '--label_images_subject', "'^AMBMCc57bl6$'",
        '--label_images_desc', "'.*Label.*'",

        '--target_voxel_sz', "[0.3,0.3,0.5]",

        '--nipype_processing_dir', "'./test_downsample'",
    ]

    tmp = DownsampleAtlasBIDS()
    tmp.run_bids(cmd_args)
