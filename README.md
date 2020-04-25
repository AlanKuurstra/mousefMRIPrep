# mousefMRIPrep


 


# for the following commands to work there's only a few things you'll need to do manually
# 1) you'll need to change the environment variable LARGE_SCRATCH_FOLDER to point to a folder on your BMI system that has a lot (probably one or two TB) of disk space.
# 2) you'll need to change the environment variable DATA_FOLDER to point to where you want to save the final results, it will probably take less than 100GB
# 3) you'll need to move your atlas, atlas mask, atlas labels, and a label mapping file to the $ATLASES folder. for me this was (/scratch/akuurstr/mousefMRIPrep_test/atlases) but for you it will depend on what you decide for LARGE_SCRATCH_FOLDER. I'm attaching a .zip file that shows you the directory structure I used for the atlases. Note that the .zip file doesn't actually have the atlas, just a 0B placeholder.
# 4) you'll need to manually create the mask for individual mice and store them in the indicated location. You can use the masks you've already segmented, but you'll need to make sure that the orientation correctly lines up with the bids data.  For instance, you'll need to compare your segmented image saved:
#    $BIDS_DERIVATIVES/CreateInitialMasks/sub-Nl311f9/ses-2020021001/anat/sub-Nl311f9_ses-2020021001_acq-TurboRARE_run-1_desc-ManualBrainMask_T2w.nii.gz
#    and
#    $BIDS/sub-Nl311f9/ses-2020021001/anat/sub-Nl311f9_ses-2020021001_acq-TurboRARE_run-1_T2w.nii.gz



# set environment variables related to data storage folders

# these two need to change for your system
LARGE_SCRATCH_FOLDER=/scratch/akuurstr
DATA_FOLDER=~/menon_mouse

# these can probably stay the same
TMP_FOLDER=$LARGE_SCRATCH_FOLDER/tmp
TEST_FOLDER=$LARGE_SCRATCH_FOLDER/mousefMRIPrep_test
SINGULARITY_IMAGES=$TEST_FOLDER/singularity_images
TAR=$TEST_FOLDER/tar
DICOM=$TEST_FOLDER/intermediate_dicoms
ATLASES=$TEST_FOLDER/atlases
NIPYPE_SCRATCH=$TEST_FOLDER/mousefMRIPrep_scratch
BIDS=$DATA_FOLDER/bids
BIDS_DERIVATIVES=$BIDS/derivatives


# create folders that might not exist
mkdir $TMP_FOLDER
mkdir $DATA_FOLDER
mkdir $TEST_FOLDER 
mkdir $SINGULARITY_IMAGES
mkdir $TAR
mkdir $DICOM
mkdir $ATLASES
mkdir $NIPYPE_SCRATCH
mkdir $BIDS
mkdir $BIDS_DERIVATIVES

# download containers for processing data
cd $SINGULARITY_IMAGES
singularity pull shub://khanlab/cfmm2tar
# this one takes a while, so we will download it in the backgroundand download dicoms while we wait
singularity pull docker://akuurstr/mousefmriprep:latest &

# download mouse dicom images from cfmm dicom server
singularity run \
-B $TAR:/output \
-B $TMP_FOLDER:/scratch \
$SINGULARITY_IMAGES/cfmm2tar_latest.sif \
-d '20200101-' \
-p Menon^Mouse_APPNL-G-F \
/output

# create bids
singularity exec \
-B $TAR:/tar \
-B $BIDS:/bids \
-B $DICOM:/dicom \
-B $TMP_FOLDER:/tmp \
$SINGULARITY_IMAGES/mousefmriprep_latest.sif \
tar2bids.py /tar /bids --intermediate_dicom_dir /dicom

# fix quadraped image orientation and introduce slice timing info
singularity exec \
-B $TAR:/tar \
-B $BIDS:/bids \
-B $DICOM:/dicom \
-B $TMP_FOLDER:/tmp \
$SINGULARITY_IMAGES/mousefmriprep_latest.sif \
fix_mouse_bids.py /dicom /bids


# create initial inaccurate mouse brain masks
singularity exec \
-B $BIDS:/bids \
-B $BIDS_DERIVATIVES:/derivatives \
-B $TMP_FOLDER:/tmp \
$SINGULARITY_IMAGES/mousefmriprep_latest.sif \
create_initial_masks.py /bids --derivatives_dir /derivatives


# MANUALLY MODIFY THE MOUSE BRAIN MASKS
# eg. open and manually modify
# $BIDS_DERIVATIVES/CreateInitialMasks/sub-Nl311f9/ses-2020021001/anat/sub-Nl311f9_ses-2020021001_acq-TurboRARE_run-1_desc-CreateInitialMasksBrainsuiteBrainMask_T2w.nii.gz
# and save as
# $BIDS_DERIVATIVES/CreateInitialMasks/sub-Nl311f9/ses-2020021001/anat/sub-Nl311f9_ses-2020021001_acq-TurboRARE_run-1_desc-ManualBrainMask_T2w.nii.gz
# where desc-CreateInitialMasksBrainsuiteBrainMask has been changed to desc-ManualBrainMask


#	MANUALLY COPY ATLASES TO $ATLASES FOLDER
# you will need an atlas, an atlas mask, and all the label images.
# in addition, you will need a label_mapping.txt file that specifies which labels you want computed in the correlation matrix.
# the format for a line in label_mapping.txt is:
# label_name img_location label_integer
# and the img_location has to be the container's path to the label image, not the host's path to the label image.

# example command to process one mouse
singularity run \
-B $BIDS:/bids \
-B $BIDS_DERIVATIVES:/derivatives \
-B $NIPYPE_SCRATCH:/nipype \
-B $ATLASES:/atlases/ \
-B $TMP_FOLDER:/tmp \
$SINGULARITY_IMAGES/mousefmriprep_latest.sif \
/bids /derivatives participant \
--participant_label Nl311f9 \
--func_session_labels 2020021001 \
--func_run_label 01 \
--anat_brain_extract_method USER_PROVIDED_MASK \
--anat_mask /derivatives/CreateInitialMasks/sub-Nl311f9/ses-2020021001/anat/sub-Nl311f9_ses-2020021001_acq-TurboRARE_run-1_desc-ManualBrainMask_T2w.nii.gz \
--atlas /atlases/AMBMC_model.nii.gz \
--atlas_mask /atlases/AMBMC_model_mask.nii.gz \
--label_mapping /atlases/label_mapping.txt \
--nipype_processing_dir /nipype \
--keep_unnecessary_outputs 



# the output directory of interest is:
# $BIDS_DERIVATIVES/MousefMRIPrep/sub-Nl311f9/ses-2020021001 

# to asses the quality of transform look at the anat image in the atlas space:
# $BIDS_DERIVATIVES/MousefMRIPrep/sub-Nl311f9/ses-2020021001/anat/sub-Nl311f9_ses-2020021001_acq-TurboRARE_run-1_desc-AnatToAtlas_T2w.nii.gz

# to asses the quality of transform also look at the func image in the atlas space:
# $BIDS_DERIVATIVES/MousefMRIPrep/sub-Nl311f9/ses-2020021001/func/sub-Nl311f9_ses-2020021001_task-rs_run-1_desc-FuncAvgToAtlas_bold.nii.gz

# the .mat file storing the average label signals is: 
# $BIDS_DERIVATIVES/MousefMRIPrep/sub-Nl311f9/ses-2020021001/func/sub-Nl311f9_ses-2020021001_task-rs_run-1_desc-LabelSignals_bold.mat

# the .mat file storing the correlation matrix is: 
# $BIDS_DERIVATIVES/MousefMRIPrep/sub-Nl311f9/ses-2020021001/func/sub-Nl311f9_ses-2020021001_task-rs_run-1_desc-CorrelationMatrix_bold.mat











# for submitting jobs on compute canada

 #!/bin/bash
 #
 #SBATCH --account=def-akhanf-ab
 #SBATCH --ntasks 1
 #SBATCH --cpus-per-task=20
 #SBATCH --mem 64000
 #SBATCH -t 2-00:00:00
 
LARGE_SCRATCH_FOLDER=/scratch/akuurstr
DATA_FOLDER=~/menon_mouse
TMP_FOLDER=$LARGE_SCRATCH_FOLDER/tmp
TEST_FOLDER=$LARGE_SCRATCH_FOLDER/mousefMRIPrep_test
SINGULARITY_IMAGES=$TEST_FOLDER/singularity_images
ATLASES=$TEST_FOLDER/atlases
NIPYPE_SCRATCH=$TEST_FOLDER/mousefMRIPrep_scratch
BIDS=$DATA_FOLDER/bids
BIDS_DERIVATIVES=$BIDS/derivatives
 
singularity run \
-B $BIDS:/bids \
-B $BIDS_DERIVATIVES:/derivatives \
-B $NIPYPE_SCRATCH:/nipype \
-B $ATLASES:/atlases/ \
-B $TMP_FOLDER:/tmp \
$SINGULARITY_IMAGES/mousefmriprep_latest.sif \
/bids /derivatives participant \
--participant_label Nl311f9 \
--func_session_labels 2020021001 \
--func_run_label 01 \
--anat_brain_extract_method USER_PROVIDED_MASK \
--anat_mask /derivatives/CreateInitialMasks/sub-Nl311f9/ses-2020021001/anat/sub-Nl311f9_ses-2020021001_acq-TurboRARE_run-1_desc-ManualBrainMask_T2w.nii.gz \
--atlas /atlases/AMBMC_model.nii.gz \
--atlas_mask /atlases/AMBMC_model_mask.nii.gz \
--label_mapping /atlases/label_mapping.txt \
--nipype_processing_dir /nipype \
--keep_unnecessary_outputs 
