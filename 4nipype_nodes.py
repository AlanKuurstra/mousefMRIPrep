from bids import BIDSLayout
import os
import numpy as np
import nipype.pipeline.engine as pe
import nipype.interfaces.brainsuite as bs
import nipype.interfaces.io as io
import os
from nipype.interfaces.io import BIDSDataGrabber


bids_dir = '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse scans/bids'
ref_file = os.path.join(bids_dir,'sub-1F4','func','sub-1F4_task-rs_run-01_bold.nii.gz')
layout = BIDSLayout(bids_dir)
metadata = layout.get_metadata(ref_file)

# slice-time corrected using `3dTshift` from AFNI

# requires:
#print(metadata["RepetitionTime"])
#print(metadata.get('SliceEncodingDirection', 'k')) # defaults to k
#print(metadata['SliceTiming'])

import nipype

# bidssrc = pe.Node(BIDSDataGrabber(subject_data=subject_data,
#                                   anat_only=anat_only,
#                                   subject_id=subject_id),
#                   name='bidssrc')

bg = pe.Node(BIDSDataGrabber(outfields = ['T2w']),name='bids-grabber')
bg.inputs.base_dir = bids_dir
bg.inputs.raise_on_empty = False
subjects = layout.get_subjects()
#print(layout.get_sessions())
#print(layout.get_modalities())
print(layout.get_datatypes()) #datatype instead of modality
#how to find out types (eg t1w,t1w,bold)
#print(layout.get(subject=subjects[0],datatype='func'))

for entity_tag_pair in layout.get_entities().items():
    print(entity_tag_pair)
    if entity_tag_pair[0] == 'global':
        print("skipping...")
        continue
    entity_tags = []
    for tag_object in entity_tag_pair[1].tags.values():
        #print(type(tag_object.value),tag_object)
        #print(tag_object.value)
        if type(tag_object.value) == list:
            nested_tuples = (tuple(l) for l in tag_object.value)
            entity_tags.append(nested_tuples)
        else:
            entity_tags.append(tag_object.value)
    print(set(entity_tags))
    print()
# how do we get the acceptable values for the entity?
# how does pybids search the entities?


bg.inputs.subject = subjects[0]
# perhaps type has be depreceated. maybe try suffix
bg.inputs.output_query = {'T2ws':dict(datatype='anat',suffix='T2w',extension=['.nii','.nii.gz']),'bolds': dict(datatype='func',suffix='bold',extension=['.nii','.nii.gz'])}
#bg.inputs.output_query = {'T2w':dict(datatype='anat',ProtocolName='T2_TurboRARE_AX150150500_192x192_AX31_4A'),'bolds': dict(datatype='func',ProtocolName='T2star_rsFMRI_3x3x500_AX192x96_SAT')} #this also works
res = bg.run()
print(res.outputs)

#sessions = layout.get_sessions(subject=subjects[0])
#layout.get_modalities()
#tasks = layout.get_tasks()
#scans = layout.get(subject=subjects[0])
