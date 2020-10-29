from workflows.CFMMWorkflow import CFMMWorkflow
from nipype_interfaces.DownsampleAtlas import get_node_downsample_atlas
from workflows.CFMMBIDS import CFMMBIDSWorkflowMixin, BIDSAppArguments, CMDLINE_VALUE
from nipype_interfaces.DerivativesDatasink import get_derivatives_entities
from workflows.CFMMCommon import get_fn_node


class DownsampleAtlas(CFMMWorkflow):
    group_name = 'Downsample Atlas'
    flag_prefix = 'downsample_'

    def _add_parameters(self):
        self._add_parameter('atlas',
                            help='')

        self._add_parameter('label_images',
                            type=eval,
                            help=f'A list of label images to downsample.',
                            )

        self._add_parameter('reference',
                            type=eval,
                            help='')

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.outputs = ['downsampled_atlas', 'downsampled_labels','downsample_shift_transform_file']

    def create_workflow(self, arg_dict=None):
        downsample_node = get_node_downsample_atlas()
        inputnode, outputnode, wf = self.get_io_and_workflow()
        wf.connect([
            (inputnode, downsample_node, [('label_images', 'highres_label_list')]),
            (inputnode, downsample_node, [('atlas', 'highres_atlas')]),
            (inputnode, downsample_node, [('reference', 'lowres_functional')]),
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
    descs=[resolution_desc(ref) for ref in references]
    descs_set = list(set(descs))
    descs_first_indices=[descs.index(desc) for desc in descs_set]
    references_set = [references[index] for index in descs_first_indices]
    return references_set, descs_set

def shift_desc(reference):
    from workflows.DownsampleAtlas import resolution_desc
    return resolution_desc(reference)+'ShiftTransformation'

#output_names must be 'desc' for the bids helper function to connect it correctly
def get_node_dynamic_res_desc(name='dynamic_desc'):
    node = get_fn_node(resolution_desc,['desc'],name=name)
    return node

def get_node_dynamic_shift_desc(name='dynamic_desc'):
    node = get_fn_node(shift_desc,['desc'],name=name)
    return node

class DownsampleAtlasBIDS(DownsampleAtlas,CFMMBIDSWorkflowMixin):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.bids = BIDSAppArguments(owner=self)
        self.bids._modify_parameter('analysis_level','choices', ['participant'])

        res_desc_node = get_node_dynamic_res_desc('res_desc')
        shift_desc_node = get_node_dynamic_shift_desc('shift_desc')


        self.create_bids_input('reference',
                               # if iterable is true, need the type conversion from argparse to be eval!!!!!
                               entities_to_overwrite={'subject': CMDLINE_VALUE,
                                                       'session': CMDLINE_VALUE,
                                                       'run': CMDLINE_VALUE,
                                                       'extension': ['.nii', '.nii.gz'],
                                                       'scope': 'self'},
                               iterable=True)
        self.create_bids_input('atlas',
                               entities_to_overwrite={'subject':CMDLINE_VALUE,
                                                       'extension': ['.nii', '.nii.gz'],
                                                       'desc': CMDLINE_VALUE,
                                                       'regex_search': True,
                                                       'scope':'Atlases'},
                               output_derivatives={'downsampled_atlas':res_desc_node,
                                                   'downsample_shift_transform_file':shift_desc_node,}
                               )
        self.create_bids_input('label_images',
                               entities_to_overwrite={'subject':CMDLINE_VALUE,
                                                       'extension': ['.nii', '.nii.gz'],
                                                       'desc': CMDLINE_VALUE,
                                                       'regex_search': True,
                                                       'scope':'Atlases'},
                               output_derivatives={'downsampled_labels':res_desc_node},
                               mapnode=True
                               )

    def create_workflow(self, arg_dict=None):
        if arg_dict:
            self.populate_parameters(arg_dict)
        wf = super().create_workflow()
        inputnode = wf.get_node('inputnode') #or self.inputnode or self.get_inputnode()
        outputnode = wf.get_node('outputnode')
        bids_group = self.bids
        self.add_bids_to_workflow(wf,inputnode,outputnode,bids_group)

        # these are all the nodes that need a pipeline connection to their input 'reference'
        inputs_with_dynamic_description = ['res_desc','shift_desc']
        self.connect_dynamic_derivatives_desc(wf, inputnode, 'reference', inputs_with_dynamic_description,'reference')



        def listify(possible_list):
            return [possible_list] if type(possible_list) != list else possible_list
        atlas = self.input_parameter_search('atlas')
        labels = listify(self.input_parameter_search('label_images'))

        # let's change the iterable to reduce unnecessary computations
        # 1) search for images with unique resolutions (only create an atlas if the reference has a different resolution
        # from those prior)
        # 2) search for the atlas in the derivatives and skip the computation if it already exists
        layout = self.bids.bids_layout_db.get_layout()
        for indx in range(len(inputnode.iterables)):
            iterable_name,iterable_value=inputnode.iterables[indx]
            if iterable_name == 'reference':
                references_set, descs_set = resolution_desc_set(iterable_value)
                refs_with_existing_derivatives=[]
                for desc, ref in zip(descs_set,references_set):
                    all_derivatives_exist=True
                    search_entities = get_derivatives_entities(atlas, desc)
                    existing_derivatives = layout.get(scope=self.__class__.__name__, **search_entities)
                    all_derivatives_exist = False if existing_derivatives == [] else all_derivatives_exist
                    search_entities = get_derivatives_entities(atlas, desc+'ShiftTransformation')
                    existing_derivatives = layout.get(scope=self.__class__.__name__, **search_entities)
                    all_derivatives_exist = False if existing_derivatives == [] else all_derivatives_exist
                    for original_file in labels:
                        search_entities = get_derivatives_entities(original_file, desc)
                        existing_derivatives = layout.get(scope=self.__class__.__name__, **search_entities)
                        all_derivatives_exist = False if existing_derivatives == [] else all_derivatives_exist
                    if all_derivatives_exist:
                        refs_with_existing_derivatives.append(ref)
                references_set = [x for x in references_set if x not in refs_with_existing_derivatives]
                inputnode.iterables[indx] = (iterable_name,references_set)
        for indx in range(len(inputnode.iterables)):
            iterable_name,iterable_value=inputnode.iterables[indx]
            if iterable_name == 'reference_original_file':
                inputnode.iterables[indx] = (iterable_name, references_set)
        return wf


if __name__ == "__main__":

    cmd_args2 = [
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids',
        '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives',
        'participant',
        '--input_derivatives_dirs',
        "['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives']",
        '--bids_layout_db', './test_downsample/bids_database',


        #'--atlas','/storage/akuurstr/Esmin_mouse_registration/mouse_scans/atlases/BIDS/sub-AMBMCc57bl6_desc-ModelDownsampled.nii.gz',
        '--atlas_subject',"'^AMBMCc57bl6$'",
        '--atlas_desc', "'^ModelDownsampled$'",

        # '--label_images',
        # "['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/atlases/BIDS/sub-AMBMCc57bl6_desc-CortexLabels15um.nii.gz',"
        # "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/atlases/BIDS/sub-AMBMCc57bl6_desc-CerebellumLabels15um.nii.gz']",
        '--label_images_subject',"'^AMBMCc57bl6$'",
        '--label_images_desc',"'.*Label.*'",

        # '--reference', "['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/sub-Nl311f9/ses-2020021001/func/sub-Nl311f9_ses-2020021001_task-rs_run-02_bold.nii.gz',"
        #                "'/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/sub-Nl311f9/ses-2020021001/func/sub-Nl311f9_ses-2020021001_task-rs_run-02_bold.nii.gz']",
        #'--reference_subject',"'Nl311f9'",
        #'--reference_session',"'2020021001'",
        #'--reference_run', "['1','2']",
        '--reference_base_bids_string','task-rs_bold',
        '--nipype_processing_dir', './test_downsample',
        '--keep_unnecessary_outputs',
    ]

    tmp = DownsampleAtlasBIDS()
    tmp.run(cmd_args2)
