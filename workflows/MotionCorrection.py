from workflows.CFMMAnts import AntsDefaultArguments, CFMMAntsRegistration
from cfmm.workflow import Workflow
from workflows.BrainExtraction import CFMMVolumesToAvg
from nipype.interfaces.fsl.maths import MeanImage
from nipype.pipeline import engine as pe
from cfmm.CFMMCommon import NipypeWorkflowArguments
from nipype.interfaces.fsl import Split
from workflows.CFMMFSL import MergeLarge
from nipype.interfaces.utility import Function
from nipype.interfaces.ants import ApplyTransforms
from nipype_interfaces.AntsDisplacementManip import MergeDisplacement


# cutom nipype interface to antsMotionCorr
# https://github.com/rwblair/nipype/blob/add_antsMotionCorr/nipype/interfaces/ants/preprocess.py
# doesn't expose ANTs nonlinear registration capabilities
# let's just do it ourselves using antsRegistration

def get_tr(in_file, tr=None):
    # is there a cheaper way to get the tr?
    if tr:
        return tr
    import nibabel as nib
    tmp = nib.load(in_file)
    #tmp.header.get_xyzt_units()[1]
    return tmp.header['pixdim'][4]

def reverse_list(forward_list):
    reverselist = forward_list.copy()
    reverselist.reverse()
    return reverselist

# not possible to make the motion correct with mapnodes
# can't encapsulate an entire workflow in a mapnode
# can't encapsulate the individual nodes in mapnodes because one of the nodes already is a mapnode and we can't
# make a mapnode of mapnodes
class MotionCorrection(Workflow):
    group_name = 'Motion Correction'
    flag_prefix = 'mc_'

    def _add_parameters(self):
        self._add_parameter('in_file',
                            help='Specify location of the input file for motion correction.')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nipype = NipypeWorkflowArguments(owner=self)
        self.roi = CFMMVolumesToAvg(owner=self)
        self.ants_args = AntsDefaultArguments(owner=self)
        self.mc_ants_reg = CFMMAntsRegistration(owner=self)

        self.mc_ants_reg.get_parameter('float').default_provider = self.ants_args.get_parameter('float')
        self.mc_ants_reg.get_parameter('interpolation').default_provider = self.ants_args.get_parameter('interpolation')
        self.mc_ants_reg.get_parameter('num_threads').default_provider = self.nipype.get_parameter('nthreads_node')

        self.outputs = ['motion_corrected_output','motion_correction_transform']

    def create_workflow(self, arg_dict=None):
        # shortcut so populate_parameters() doesn't need to explicitly be called before get_workflow()
        if arg_dict is not None:
            self.populate_user_value(arg_dict)
            self.validate_parameters()

        nthreads_node = self.nipype.get_parameter('nthreads_node').user_value
        nthreads_mapnode = self.nipype.get_parameter('nthreads_mapnode').user_value
        mem_gb_mapnode = self.nipype.get_parameter('mem_gb_mapnode').user_value


        imgs_to_avg = self.roi.get_node(name='imgs_to_avg')
        avg_img = pe.Node(MeanImage(), name='avg_img')

        get_tr_node = pe.Node(
            Function(input_names=["in_file", "tr"], output_names=["tr"],
                     function=get_tr), name='get_tr_node')


        split_func = pe.Node(interface=Split(), name='split_func', n_procs=nthreads_node)
        split_func.inputs.dimension = 't'

        mc_ants_reg_interface = self.mc_ants_reg.get_interface()
        mc_ants_reg = pe.MapNode(interface=mc_ants_reg_interface, name='mc_ants_reg', iterfield=['moving_image', ],
                                 n_procs=nthreads_mapnode, mem_gb=mem_gb_mapnode)

        reverse_transform_list = pe.MapNode(
            Function(input_names=["forward_list"], output_names=["transforms_reversed"],
                     function=reverse_list), name='reverse_transform_list', iterfield=['forward_list'])

        combine_mc_displacements = pe.MapNode(interface=ApplyTransforms(), name='combine_mc_displacements',
                                              iterfield=['transforms', 'reference_image', 'input_image'],
                                              n_procs=nthreads_mapnode, mem_gb=mem_gb_mapnode)
        combine_mc_displacements.inputs.dimension = 3
        combine_mc_displacements.inputs.output_image = 'motion_corr_transform.nii.gz'
        combine_mc_displacements.inputs.print_out_composite_warp_file = True

        # shell command won't work with list of 600
        #create_4d_mc_img = pe.Node(interface=Merge(), name='create_4d_mc_img', n_procs=nthreads_node,)
        #create_4d_mc_img.inputs.dimension = 't'
        create_4d_mc_img = pe.Node(interface=MergeLarge(), name='create_4d_mc_img', n_procs=nthreads_node, )

        create_4d_mc_displacement = pe.Node(interface=MergeDisplacement(), name='create_4d_mc_displacement',
                                            n_procs=nthreads_node)

        inputnode, outputnode, wf = self.get_io_and_workflow()

        wf.connect([
            (inputnode, get_tr_node, [('in_file', 'in_file')]),
            (inputnode, imgs_to_avg, [('in_file', 'in_file')]),
            (imgs_to_avg, avg_img, [('roi_file', 'in_file')]),
            (inputnode, split_func, [('in_file', 'in_file')]),
            (split_func, mc_ants_reg, [('out_files', 'moving_image')]),
            (avg_img, mc_ants_reg, [('out_file', 'fixed_image')]),
            (mc_ants_reg, reverse_transform_list, [('forward_transforms', 'forward_list')]),
            (reverse_transform_list, combine_mc_displacements, [('transforms_reversed', 'transforms')]),
            (mc_ants_reg, combine_mc_displacements, [('warped_image', 'reference_image')]),
            (mc_ants_reg, combine_mc_displacements, [('warped_image', 'input_image')]),
            (mc_ants_reg, create_4d_mc_img, [('warped_image', 'in_files')]),
            (get_tr_node, create_4d_mc_img, [('tr', 'tr')]),
            (create_4d_mc_img, outputnode, [('merged_file', 'motion_corrected_output')]),
            (combine_mc_displacements, create_4d_mc_displacement, [('output_image', 'displacement_imgs')]),
            (get_tr_node, create_4d_mc_displacement, [('tr', 'tr')]),
            (create_4d_mc_displacement, outputnode, [('output_file', 'motion_correction_transform')]),
            ])
        return wf
