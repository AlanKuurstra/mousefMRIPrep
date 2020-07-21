from nipype.pipeline import engine as pe
from nipype.interfaces.utility import Function

def downsample_atlas(highres_atlas, highres_label_list, lowres_func,
                     output_lowres_atlas_path = None, output_lowres_label_path_list=None):
    import numpy as np
    import nibabel as nib
    import subprocess
    import os
    from tools.split_exts import split_exts
    import tempfile

    if type(highres_label_list) == str:
        highres_label_list = [highres_label_list]
    if type(output_lowres_label_path_list) == str:
        output_lowres_label_path_list = [output_lowres_label_path_list]

    # setup nifti outputs
    if output_lowres_label_path_list is None:
        output_lowres_label_path_list=[]
        for highres_label in highres_label_list:
            name, ext = split_exts(highres_label)
            output_lowres_label_path_list.append(os.path.abspath(f'{name}_bin_downsampled{ext}'))
    assert (len(highres_label_list) == len(output_lowres_label_path_list))
    if output_lowres_atlas_path is None:
        output_lowres_atlas_path = os.path.abspath('atlas_downsampled.nii.gz')
    tmpdir = tempfile.TemporaryDirectory()
    highres_atlas_copy = os.path.join(tmpdir.name,'atlas_highres_copy.nii.gz')

    # resolution to downsample to
    func_low_obj = nib.load(lowres_func)
    # sform and qfrom defined. qform rotations ignored and only use sform which only has scaling.
    target_voxel_sz = func_low_obj.header['pixdim'][1:4]

    atlas_already_downsampled = False
    lowres_label_map = {}
    for highres_label,output_lowres_label_path in zip(highres_label_list,output_lowres_label_path_list):
        label_high_obj = nib.load(highres_label)

        # store original atlas/label affine settings

        # overall affine (should match one of fallback,qform,sform)
        # when qform_code=sfrom_code=0, nibabel is assuming buffer in LAS against nifti standard
        # this is a problem for img_obj.affine, manipulate default_x_flip to avoid this behaviour
        label_high_obj.header.default_x_flip=False
        label_high_obj._affine = label_high_obj.header.get_best_affine()
        orig_affine = label_high_obj.affine
        #fallback affine (offset comes from nibabel putting [0,0,0] voxel in center of image...also against nifti standard)
        orig_header_base_affine = label_high_obj.header.get_base_affine()
        #qform affine
        orig_qform_affine,orig_qform_code = label_high_obj.get_qform(coded=True)
        #sform affine
        orig_sform_affine, orig_sform_code = label_high_obj.get_sform(coded=True)

        # data to downsample
        label_high_data = label_high_obj.get_data()

        voxel_sz_high = label_high_obj.header['pixdim'][1:4]

        # floor to retain detail
        downsample_stride = np.floor(target_voxel_sz / voxel_sz_high).astype(int)
        # kernel will be symmetric with respect to center voxel - use more voxels in the kernel's label histogram if necessary
        kernel_rad = np.ceil((downsample_stride - 1) / 2.0).astype(int)

        # high res voxel dims
        dim_h = label_high_obj.header['dim'][1:4]

        # just start as close to image edge as possible
        first_downsampled_voxel=kernel_rad

        # how many low res voxels in each dim (taking into account padding due to strides)
        dim_l = np.floor((dim_h - first_downsampled_voxel - kernel_rad) / downsample_stride).astype(int) + 1

        last_possible_downsampled_voxel= dim_h - (first_downsampled_voxel + (dim_l - 1) * downsample_stride + 1)

        label_downsample=np.empty(dim_l, 'uint8')
        x_l=0
        y_l=0
        z_l=0
        for x in range(first_downsampled_voxel[0], dim_h[0] - last_possible_downsampled_voxel[0], downsample_stride[0]):
            #print(x-kernel_rad,x, x+kernel_rad)#inclusive
            xbegin=x - kernel_rad[0]
            xend = x + kernel_rad[0] + 1
            #print("x:",xbegin, x, xend)
            y_l=0
            for y in range(first_downsampled_voxel[1], dim_h[1] - last_possible_downsampled_voxel[1], downsample_stride[1]):
                ybegin = y - kernel_rad[1]
                yend = y + kernel_rad[1] + 1
                #print("y:",ybegin, y, yend)
                z_l=0
                for z in range(first_downsampled_voxel[2], dim_h[2] - last_possible_downsampled_voxel[2], downsample_stride[2]):
                    zbegin = z - kernel_rad[2]
                    zend = z + kernel_rad[2] + 1
                    #print("z:",zbegin, z, zend)
                    most_frequent = np.bincount(label_high_data[xbegin:xend, ybegin:yend, zbegin:zend].ravel()).argmax()
                    #print(x_l,y_l,z_l,most_frequent)
                    label_downsample[x_l, y_l, z_l] = most_frequent
                    z_l+=1
                y_l+=1
            x_l+=1

        label_downsample_header = label_high_obj.header.copy()
        # remove atlas/label orientation information
        # using np.eye for affine mtx - this also causes nibabel to automatically change the pixel dimension information
        label_downsample_header.set_qform(np.eye(4),2)
        label_downsample_header.set_sform(np.eye(4),0)
        # change pixel dimensions to be the stride (this will help when using ants to resample the atlas (atlas not label) image)
        label_downsample_header.set_zooms(downsample_stride)
        nifti_image = nib.Nifti1Image(label_downsample, None, label_downsample_header)
        nifti_image.to_filename(output_lowres_label_path)

        if not atlas_already_downsampled:
            # we now want to downsample the high res atlas to correspond to the label image
            # however by performing stride downsampling on the label image, we shifted the center voxel of the image
            # here we will shift the qform of the high res atlas so the downsampled atlas is shifted during ANTs resampling and
            # is aligned with our downsampled label. This atlas will be used as the reference when applying the func->anat->atlas
            # transforms to the functional images.
            atlas_obj = nib.load(highres_atlas)
            qform_shifted = np.eye(4)
            qform_shifted[:-1,-1] = -1*first_downsampled_voxel #ants respects the nifti standard with (0,0,0) voxel

            # DO NOT UPDATE QFORM USING THE HEADER OBJECT, MUST USE THE NIFTI IMAGE OBJECT TO UPDATE THE IMAGE AFFINE
            #atlas_obj.header.set_qform(qform_shifted,2) #BAD!! does not update atlas_obj._affine and also sometimes messes with sform
            atlas_obj.set_qform(qform_shifted,2)
            atlas_obj.set_sform(np.eye(4),0)
            atlas_obj.to_filename(highres_atlas_copy)

            # note: ants puts the qform_code as 1 (scanner anat) instead of 2 (aligned anat) which we are using
            cmd_list = ['antsApplyTransforms',
                        '-d', '3',
                        '-i', highres_atlas_copy,
                        '-r', output_lowres_label_path,
                        '-o', output_lowres_atlas_path]
            print(" ".join(cmd_list))
            subprocess.run(cmd_list)

        # return to original orientations, but change the pixel dimension to downsampled size
        # DO NOT UPDATE QFORM USING THE HEADER OBJECT, MUST USE THE NIFTI IMAGE OBJECT TO UPDATE THE IMAGE AFFINE
        # nifti_obj.header.set_qform() does not update nifti_obj._affine and also sometimes messes with the sform
        tmp = nib.load(output_lowres_label_path)
        tmp.set_qform(orig_qform_affine, orig_qform_code)
        tmp.set_sform(orig_sform_affine, orig_sform_code)
        tmp.header['pixdim'][1:4] = voxel_sz_high * downsample_stride
        # changing pixdim changes header base_affine, but doesn't change the nifti object's _affine or get_affine()
        # for this reason we can't just save this nifti object to disk, could try:
        # tmp._affine = tmp.header.get_best_affine() .... but maybe it's safer to just create a new object.
        nifti_image = nib.Nifti1Image(tmp.get_data(), None, tmp.header)
        nifti_image.to_filename(output_lowres_label_path)
        lowres_label_map[highres_label] = output_lowres_label_path

        if not atlas_already_downsampled:
            tmp = nib.load(output_lowres_atlas_path)
            tmp.set_qform(orig_qform_affine, orig_qform_code)
            tmp.set_sform(orig_sform_affine, orig_sform_code)
            tmp.header['pixdim'][1:4] = voxel_sz_high * downsample_stride
            nifti_image = nib.Nifti1Image(tmp.get_data(), None, tmp.header)
            nifti_image.to_filename(output_lowres_atlas_path)
            atlas_already_downsampled = True

    # to align an image that's being registered to the high resolution atlas,
    # we can't just change the offset in its qform. (like we did to align the atlas with label).
    # we need to first apply registartion transformations, and then the pixel shift transformation
    # AFTER, not before
    # so we need to create a shift transform that can be applied after registration transforms
    import scipy.io as sio
    fixed_mtx_name = 'fixed'
    fixed_mtx = np.asarray([0, 0, 0])[:, np.newaxis]
    affine_mtx_name = 'AffineTransform_double_3_3'
    # the shift in atlas space will be the voxel offset rotated into atlas space
    # rotation from RAS to atlas space
    orig_rot = orig_affine[:3, :3]
    # find voxel offsets in atlas space
    shift = np.dot(orig_rot, first_downsampled_voxel)
    shift_mtx = np.concatenate([np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1]), shift])[:, np.newaxis]

    shift_transform_file = os.path.abspath("shift.mat")
    sio.savemat(shift_transform_file, {affine_mtx_name: shift_mtx, fixed_mtx_name: fixed_mtx}, format='4')

    return output_lowres_atlas_path, lowres_label_map, shift_transform_file

def get_node_downsample_atlas(name='downsample_atlas'):
    node = pe.Node(
        Function(input_names=[
            "highres_atlas",
            "highres_label_list",
            "lowres_func",
            "output_lowres_atlas_path",
            "output_lowres_label_path_list",
        ],
             output_names=["output_lowres_atlas_path",
                           "lowres_label_map",
                           "shift_transform_file"],
             function=downsample_atlas), name = name)
    return node


if __name__ == '__main__':
    # lowres_atlas, lowres_label, shift_transform = downsample_atlas(highres_atlas, highres_label, lowres_func)
    import os
    from nipype_interfaces.CorrelationMatrix import get_node_label_list

    label_list_node = get_node_label_list()
    label_list_node.inputs.label_mapping_file = '/softdev/akuurstr/python/modules/mousefMRIPrep/examples/label_mapping_host.txt'

    lowres_func = "/storage/akuurstr/mouse_pipepline_output/mousefMRIPrep_scratch/func_processing/preprocess_func_stc_split/vol0000.nii.gz"
    downsample_node = get_node_downsample_atlas()
    #tmp.inputs.highres_label_list = '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/atlases/labels/AMBMC-c57bl6-cortex-labels-15um.nii.gz'
    downsample_node.inputs.highres_atlas = '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/atlases/AMBMC_model.nii.gz'
    downsample_node.inputs.lowres_func = lowres_func
    #results = downsample_node.run()

    wf = pe.Workflow(name='test_downsample')
    wf.base_dir = os.path.abspath('.')
    wf.connect([
        (label_list_node, downsample_node, [('label_list', 'highres_label_list')]),
    ])
    exec_graph = wf.run()
    results = list(exec_graph.nodes)[1].result

    import subprocess
    lowres_label = results.outputs.lowres_label_map['/storage/akuurstr/Esmin_mouse_registration/mouse_scans/atlases/labels/AMBMC-c57bl6-cortex-labels-15um.nii.gz']
    shift_transform = results.outputs.shift_transform_file
    output_func_reg = os.path.abspath('func_registered.nii.gz')

    cmd_list = [
        "antsApplyTransforms",
        "--default-value", "0",
        "--dimensionality", "3",
        "--float", "1",
        "--input", lowres_func,
        "--interpolation", "Linear",
        "--output", output_func_reg,
        "--reference-image", lowres_label,
        "--transform", shift_transform, "/storage/akuurstr/mouse_pipepline_output/mousefMRIPrep_scratch/func_processing/concat_transforms_func_to_atlas/func_to_atlas_transform.nii"
    ]
    print(" ".join(cmd_list))
    subprocess.run(cmd_list)








