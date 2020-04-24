#!/usr/bin/env python3
import os
import argparse
from argparse import ArgumentParser
from nipype import config, logging
from workflows.BrainExtraction import BrainExtractMethod
from workflows.FuncProcessing import MotionCorrectionTransform
from workflows.AnatProcessing import init_anat_processing
from workflows.FuncProcessing import init_func_processing
from tools.RestrictedDict import RestrictedDict
import json
import sys
from bids import BIDSLayout
from bids.layout.layout import parse_file_entities
import tempfile

debugging = False
pipeline_name = 'MousefMRIPrep'

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    # followed format from BIDS-Apps/nipypelines
    defstr = ' (default %(default)s)'
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('bids_dir', help='Data directory formatted according to BIDS standard.')
    parser.add_argument('output_derivatives_dir', help='Directory where processed output files are to be stored in bids derivatives format.')
    parser.add_argument('analysis_level',
                        help='Level of the analysis that will be performed.',
                        choices=['participant'])
    parser.add_argument('--participant_label',
                        help='The label(s) of the participant(s) that should be analyzed. The label '
                             'corresponds to sub-<participant_label> from the BIDS spec '
                             '(do not include prefix "sub-"). If this parameter is not '
                             'provided all subjects will be analyzed. Multiple '
                             'participants can be specified with a space separated list.',
                        )
    parser.add_argument('--func_session_labels', help='The label(s) of the session(s) that should be analyzed. The label '
                                                'corresponds to ses-<session_label> from the BIDS spec '
                                                '(do not include prefix "ses-"). If this parameter is not '
                                                'provided all sessions will be analyzed. Multiple '
                                                'sessions can be specified with a space separated list.',
                        nargs="+")
    parser.add_argument('--func_run_labels', help='The label(s) of the run(s) that should be analyzed. The label '
                                                'corresponds to run-<run_label> from the BIDS spec '
                                                '(do not include prefix "run-"). If this parameter is not '
                                                'provided all runs will be analyzed. Multiple '
                                                'runs can be specified with a space separated list.',
                        nargs="+")

    parser.add_argument('--anat_session_label', help='The label(s) of the session(s) that should be analyzed. The label '
                                                'corresponds to ses-<session_label> from the BIDS spec '
                                                '(do not include prefix "ses-"). If this parameter is not '
                                                'provided all sessions will be analyzed. Multiple '
                                                'sessions can be specified with a space separated list.',
                        )
    parser.add_argument('--anat_run_label', help='The label(s) of the run(s) that should be analyzed. The label '
                                                'corresponds to run-<run_label> from the BIDS spec '
                                                '(do not include prefix "run-"). If this parameter is not '
                                                'provided all runs will be analyzed. Multiple '
                                                'runs can be specified with a space separated list.',
                        )

    parser.add_argument('--config_file',
                        help='')

    parser.add_argument('--write_config_file',
                        const='config.json',
                        nargs='?',
                        help='')

    parser.add_argument('--func_entities',
                        default='task-rs_bold.nii.gz',
                        help='')
    parser.add_argument('--anat_entities',
                        default='acq-TurboRARE_T2w.nii.gz',
                        help='')


    parser.add_argument('--input_masks_description_label',
                        default='ManualBrainMask',
                        help='mutually exclusive with func_mask and anat_mask')
    parser.add_argument('--func_mask',
                        help='mutually exclusive with input_masks_description_label')
    parser.add_argument('--anat_mask',
                        help='mutually exclusive with input_masks_description_label')



    parser.add_argument('--func_template_desc',
                        help='mutually exclusive with func_template and func_template_probability_mask. Overrides option in config file.')
    parser.add_argument('--func_template',
                        help='mutually exclusive with func_template_desc. Overrides option in config file.')
    parser.add_argument('--func_template_probability_mask',
                        help='mutually exclusive with func_template_desc. Overrides option in config file.')

    parser.add_argument('--anat_template_desc',
                        help='mutually exclusive with anat_template and anat_template_probability_mask. Overrides option in config file.')
    parser.add_argument('--anat_template',
                        help='mutually exclusive with anat_template_desc. Overrides option in config file.')
    parser.add_argument('--anat_template_probability_mask',
                        help='mutually exclusive with anat_template_desc. Overrides option in config file.')
    parser.add_argument("--force_anat_processing",
                        action='store_true',
                        help="")

    parser.add_argument('--atlas',
                        help='Overrides option in config file.')
    parser.add_argument('--atlas_mask',
                        help='Overrides option in config file.')
    parser.add_argument('--label_mapping',
                        help='Overrides option in config file.')


    parser.add_argument("--mc_transform_method",
                        choices=MotionCorrectionTransform.__members__,
                        default=MotionCorrectionTransform.NO_MC.name,
                        help="")

    parser.add_argument("--perform_stc",
                        type=str2bool,
                        help="True or False. If None, will look in .json.")

    parser.add_argument("--perform_func_to_anat_registration",
                        action='store_true',
                        help="")

    parser.add_argument("--no_masks_func_to_anat_registration",
                        action='store_true',
                        help="")

    parser.add_argument("--no_masks_anat_to_atlas_registration",
                        action='store_true',
                        help="")

    parser.add_argument("--func_brain_extract_method",
                        choices=BrainExtractMethod.__members__,
                        default=BrainExtractMethod.NO_BRAIN_EXTRACTION.name,
                        help="")

    parser.add_argument("--anat_brain_extract_method",
                        choices=BrainExtractMethod.__members__,
                        default=BrainExtractMethod.REGISTRATION_NO_INITIAL_MASK.name,
                        help="")

    parser.add_argument("--interpolation",
                        default='Linear',
                        help="antsRegistration and antsApplyTransforms interpolation method")

    parser.add_argument("--high_precision",
                        action='store_true',
                        help="antsRegistration antsApplyTransform float precision by default.")

    parser.add_argument("--gzip_large_images",
                        action='store_true',
                        help="")

    parser.add_argument("--n4_bspline_fitting_distance",
                        default=20,
                        type=float,
                        help="n4 bspline fitting distance, for mouse smaller")

    parser.add_argument("--diffusionConstant",
                        default=30,
                        type=float,
                        help="Diffusion constant from BrainSuite brain extraction.")
    parser.add_argument("--diffusionIterations",
                        default=3,
                        type=int,
                        help="Diffusion iterations from BrainSuite brain extraction.")
    parser.add_argument("--edgeDetectionConstant",
                        default=0.55,
                        type=float,
                        help="Edge detection constant from BrainSuite brain extraction.")
    parser.add_argument("--radius",
                        default=2,
                        type=float,
                        help="Radius from BrainSuite brain extraction.")
    parser.add_argument("--skipDilateFinalMask",
                        action='store_true',
                        help="Don't dilate final mask in BrainSuite brain extraction.")

    parser.add_argument("--correlation_shift_interval_s",
                        default=1.5,
                        type=float,
                        help="")
    parser.add_argument("--correlation_max_shift_s",
                        default=0.375,
                        type=float,
                        help="")
    parser.add_argument("--correlation_search_for_neg_corr",
                        action='store_true',
                        help="")

    parser.add_argument("--omp_nthreads",
                        type=int,
                        help="True or False. If None, will use number of available cpus.")
    parser.add_argument("--mem_gb",
                        default=50,
                        type=float,
                        help="")

    parser.add_argument('--nipype_processing_dir',
                        help='Directory where intermediate images, logs, and crash files should be stored.')
    parser.add_argument("-w", "--work_dir", dest="work_dir",
                        help=f"Work directory. Defaults to <nipype_processing_dir>/{pipeline_name}_scratch'")
    parser.add_argument("-l", "--log_dir", dest="log_dir",
                        help="Nipype output log directory. Defaults to <nipype_processing_dir>/log")
    parser.add_argument("-c", "--crash_dir", dest="crash_dir",
                        help="Nipype crash dump directory. Defaults to <nipype_processing_dir>/crash_dump")
    parser.add_argument("-p", "--plugin", dest="plugin",
                        default='Linear', help="Nipype run plugin")
    parser.add_argument("--plugin_args", dest="plugin_args",
                        help="Nipype run plugin arguments")
    parser.add_argument("--keep_unnecessary_outputs", dest="keep_unnecessary_outputs",
                        action='store_true', default=False,
                        help="Keep all nipype node outputs, even if unused")

    if debugging:
        BidsDir = os.path.abspath('/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids')
        derivatives_dir = os.path.abspath('/storage/akuurstr/Esmin_mouse_registration/mouse_scans/bids/derivatives')
        nipype_dir = os.path.abspath('/storage/akuurstr/mouse_pipepline_output')

        parameters = [BidsDir, derivatives_dir, 'participant',
                      '--participant_label', 'Nl311f9',
                      '--func_session_labels', '2020021001',
                      '--func_run_labels', '01',

                      #'--config_file', 'config.json',
                      #'--write_config_file','config2.json',
                      '--perform_func_to_anat_registration',

                      '--atlas', '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/atlases/AMBMC_model.nii.gz',
                      '--atlas_mask', '/storage/akuurstr/Esmin_mouse_registration/mouse_scans/atlases/AMBMC_model_mask.nii.gz',
                      '--label_mapping', '/softdev/akuurstr/python/modules/mousefMRIPrep/examples/label_mapping_host_short.txt',

                      '--func_brain_extract_method', 'BRAINSUITE',
                      '--anat_brain_extract_method', 'BRAINSUITE',
 \
                      #'--anat_template', '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/derivatives/BrainExtractionTemplatesAndProbabilityMasks/AnatTemplate_acq-TurboRARE_desc-0p15x0p15x0p55mm20200402_T2w.nii.gz',
                      #'--anat_template_probability_mask', '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/derivatives/BrainExtractionTemplatesAndProbabilityMasks/AnatTemplateProbabilityMask_acq-TurboRARE_desc-0p15x0p15x0p55mm20200402_T2w.nii.gz',
                      #'--func_template', '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/derivatives/BrainExtractionTemplatesAndProbabilityMasks/FuncTemplate_task-rs_desc-avg0p3x0p3x0p55mm20200402_bold.nii.gz',
                      #'--func_template_probability_mask', '/home/akuurstr/Desktop/Esmin_mouse_registration/mouse_scans/bids/derivatives/BrainExtractionTemplatesAndProbabilityMasks/FuncTemplateProbabilityMask_task-rs_desc-avg0p3x0p3x0p55mm20200402_bold.nii.gz',

                      '--nipype_processing_dir', nipype_dir,
                      '--keep_unnecessary_outputs',
                      ]
        args = parser.parse_args(parameters)
    else:
        parameters = sys.argv
        args = parser.parse_args()

    # store arguments in a restricted dictionary
    unsaved_keys = ['help','bids_dir','output_derivatives_dir','analysis_level', 'participant_label', 'func_session_labels', 'func_run_labels', 'anat_session_label','anat_run_label','config_file', 'write_config_file', 'func_mask', 'anat_mask']
    allowed_keys = [action.dest for action in parser._actions if action.dest not in unsaved_keys]
    arg_dict = RestrictedDict(allowed_keys)

    # dict storing positional and default arg values
    specified_and_default_args_dict = vars(args)

    # dict storing command-line-specified optional arg values
    # find arguments specified on command line that will override defaults and config file
    specified_args=[]
    for parameter in parameters:
        optional_parameter = parser._parse_optional(parameter)
        if optional_parameter is not None:
            specified_args.append(optional_parameter[0].dest)
    specified_args = list(set(specified_args))
    # store command-line-specified optional arg values in a dict
    specified_optional_args_dict = {k: v for k, v in specified_and_default_args_dict.items() if k in specified_args}

    # dict storing config file arg values
    config_args_dict = {}
    if 'config_file' in specified_args:
        with open(specified_optional_args_dict['config_file'],'r') as f:
            config_args_dict = json.load(f)

    if 'write_config_file' in specified_args:
        # omit default values - only store original config file values and command-line-specified values
        arg_dict.update(config_args_dict)
        arg_dict.update(specified_optional_args_dict,ignore_warnings=True)
        print(f"writing argument options to {specified_optional_args_dict['write_config_file']}")
        with open(specified_optional_args_dict['write_config_file'],'w') as f:
            json.dump(arg_dict,f,indent=2)
        sys.exit()

    # initialize with parser default values
    arg_dict.update(specified_and_default_args_dict,ignore_warnings=True)
    # override default arguments with config file values
    arg_dict.update(config_args_dict)
    # override config file values with arguments explicitly specified on command line
    arg_dict.update(specified_optional_args_dict,ignore_warnings=True)


    # set args not stored in arg_dict
    bids_dir = args.bids_dir
    derivatives_dir = args.output_derivatives_dir

    subject = args.participant_label
    func_sessions = args.func_session_labels
    func_runs = args.func_run_labels
    anat_session = args.anat_session_label
    anat_run = args.anat_run_label

    func_mask = args.func_mask
    anat_mask = args.anat_mask

    if arg_dict['input_masks_description_label'] is not None:
        if func_mask is not None:
            print(f"Warning: overriding input_masks_description_label={arg_dict['input_masks_description_label']} search with func_mask {func_mask}")
        else:
            #func_mask =
            #NUMBER OF MASKS IN FUNC_MASK OR FOUND USING THE DESCRIPTION NEEDS TO MATCH THE NUMBER OF FUNCS FOUND USING SESSION&RUN KEYWORDS
            pass

    if arg_dict['input_masks_description_label'] is not None:
        if anat_mask is not None:
            print(f"Warning: overriding input_masks_description_label={arg_dict['input_masks_description_label']} search with anat_mask {anat_mask}")
        else:
            #anat_mask =
            pass

    if arg_dict['func_template_desc'] is not None:
        if ((arg_dict['func_template'] is not None) and (arg_dict['func_template_probability_mask'] is not None)):
            print(f"Waring: overriding func_template_desc={arg_dict['func_template_desc']} search with func_template {arg_dict['func_template']} and func_template_probability_mask {arg_dict['func_template_probability_mask']}")
        else:
            #func_template =
            #func_template_probability_mask =
            pass

    if arg_dict['anat_template_desc'] is not None:
        if ((arg_dict['anat_template'] is not None) and (arg_dict['anat_template_probability_mask'] is not None)):
            print(f"Waring: overriding anat_template_desc={arg_dict['anat_template_desc']} search with anat_template {arg_dict['anat_template']} and anat_template_probability_mask {arg_dict['anat_template_probability_mask']}")
        else:
            # anat_template =
            # anat_template_probability_mask =
            pass


    if ((arg_dict['func_template'] is not None) or (arg_dict['func_template_probability_mask'] is not None)) \
        and \
        ((arg_dict['func_template'] is None) or (arg_dict['func_template_probability_mask'] is None)):
        parser.error("func_template and func_template_probability_mask must be defined together")

    if arg_dict['perform_func_to_anat_registration']:
        if (not arg_dict['no_masks_func_to_anat_registration']):
            if (arg_dict['anat_brain_extract_method'] == BrainExtractMethod.NO_BRAIN_EXTRACTION.name):
                parser.error(f"Using masks in func to anat registration, but anat_brain_extract_method is set to {arg_dict['anat_brain_extract_method']}")
            if (arg_dict['func_brain_extract_method'] == BrainExtractMethod.NO_BRAIN_EXTRACTION.name):
                parser.error(f"Using masks in func to anat registration, but func_brain_extract_method is set to {arg_dict['func_brain_extract_method']}")

        if arg_dict['func_brain_extract_method'] in (BrainExtractMethod.REGISTRATION_NO_INITIAL_MASK.name,BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK.name, BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK.name):
            if arg_dict['func_template'] is None:
                parser.error(f"{arg_dict['func_brain_extract_method']} used for func_brain_extract_method but func_template is not provided or func_template_desc did not find a template")
            if arg_dict['func_template_probability_mask'] is None:
                parser.error(f"{arg_dict['func_brain_extract_method']} used for func_brain_extract_method but func_template_probability_mask is not provided or func_template_desc did not find a probability mask")

        if (arg_dict['func_brain_extract_method'] in (
        BrainExtractMethod.USER_PROVIDED_MASK.name, BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK.name,
        )) and (func_mask is None):
            parser.error(
                f"{arg_dict['func_brain_extract_method']} used for func_brain_extract_method but func_mask is not provided or input_masks_description_label did not find a func mask")


    if (not arg_dict['no_masks_anat_to_atlas_registration']):
        if arg_dict['atlas_mask'] is None:
            parser.error(f"Using masks in anat to atlas registration, but no value provided for atlas_mask")
        if (arg_dict['anat_brain_extract_method'] == BrainExtractMethod.NO_BRAIN_EXTRACTION.name):
            parser.error(f"Using masks in anat to atlas registration, but anat_brain_extract_method is set to {arg_dict['anat_brain_extract_method']}")

    if arg_dict['anat_brain_extract_method'] in (
    BrainExtractMethod.REGISTRATION_NO_INITIAL_MASK.name, BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK.name,
    BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK.name):
        if arg_dict['anat_template'] is None:
            parser.error(
                f"{arg_dict['anat_brain_extract_method']} used for anat_brain_extract_method but anat_template is not provided or anat_template_desc did not find a template")
        if arg_dict['anat_template_probability_mask'] is None:
            parser.error(
                f"{arg_dict['anat_brain_extract_method']} used for anat_brain_extract_method but anat_template_probability_mask is not provided or anat_template_desc did not find a probability mask")

    if (arg_dict['anat_brain_extract_method'] in (
            BrainExtractMethod.USER_PROVIDED_MASK.name, BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK.name,
    )) and (anat_mask is None):
        parser.error(
            f"{arg_dict['anat_brain_extract_method']} used for anat_brain_extract_method but anat_mask is not provided or input_masks_description_label did not find an anat mask")

    layout = BIDSLayout(bids_dir)
    layout.add_derivatives(derivatives_dir)

    func_search_entities = parse_file_entities(os.path.sep+arg_dict['func_entities'])
    func_search_entities['subject'] = subject
    func_search_entities['desc'] = None
    func_search_entities['datatype'] = 'func'
    if func_sessions is not None:
        func_search_entities['session'] = func_sessions
    if func_runs is not None:
        func_search_entities['run'] = func_runs

    funcs = layout.get(**func_search_entities)

    anat_search_entities = parse_file_entities(os.path.sep+arg_dict['anat_entities'])
    anat_search_entities['subject'] = subject
    anat_search_entities['desc'] = None
    anat_search_entities['datatype'] = 'anat'
    if anat_session is not None:
        anat_search_entities['session'] = anat_session
        if anat_run is not None:
            anat_search_entities['run'] = anat_run
    elif func_sessions is not None:
        anat_search_entities['session'] = func_sessions
        if layout.get(**anat_search_entities) == []:
            if 'session' in anat_search_entities:
                anat_search_entities.pop('session')

    anat = layout.get(**anat_search_entities)

    if len(anat)>1:
        print(f"Warning: multiple anat files found, using {anat[0].path}")
    anat = anat[0].path

    nipype_dir = arg_dict['nipype_processing_dir']
    if nipype_dir == None:
        nipype_dir = tempfile.TemporaryDirectory().name
    if args.work_dir:
        work_dir = os.path.abspath(args.work_dir)
    else:
        work_dir = os.path.join(nipype_dir, f'{pipeline_name}_scratch')
    if args.log_dir:
        log_dir = os.path.abspath(args.log_dir)
    else:
        tmp = "log-" + "_".join(subject) + '-' + "_".join(func_sessions)
        tmp = tmp.replace(".*", "all").replace("*", "star")
        log_dir = os.path.join(nipype_dir, 'logs', tmp)
    if args.crash_dir:
        crash_dir = os.path.abspath(args.crash_dir)
    else:
        crash_dir = os.path.join(nipype_dir, 'crash_dump')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    config.update_config({'logging': {
        'log_directory': log_dir,
        'log_to_file': True,
    },
        'execution': {
            'crashdump_dir': crash_dir,
            'crashfile_format': 'txt',
        }})
    logging.update_logging(config)

    plugin = args.plugin
    plugin_args = args.plugin_args
    keep_unnecessary_outputs = args.keep_unnecessary_outputs

    def get_anat_derivatives(original_anat_file,anat_brain_extract_method):
        anat_entities = parse_file_entities(original_anat_file)
        anat_entities.pop('extension')

        anat_entities['desc'] = 'AnatToAtlasTransform'
        anat_to_atlas_composite_transform = layout.get(drop_invalid_filters=False, **anat_entities)
        anat_entities['desc'] = 'n4Corrected'
        anat_n4_corrected = layout.get(drop_invalid_filters=False, **anat_entities)
        # mask desc depends on the proposed method.
        method = None
        if anat_brain_extract_method == BrainExtractMethod.BRAINSUITE:
            method = 'Brainsuite'
        elif anat_brain_extract_method == BrainExtractMethod.REGISTRATION_NO_INITIAL_MASK:
            method = 'NoInitTemplateExtracted'
        elif anat_brain_extract_method == BrainExtractMethod.REGISTRATION_WITH_INITIAL_BRAINSUITE_MASK:
            method = 'BrainsuiteInitTemplateExtracted'
        elif anat_brain_extract_method == BrainExtractMethod.REGISTRATION_WITH_INITIAL_MASK:
            method = 'UserInitTemplateExtracted'
        anat_entities['desc'] = f'{method}BrainMask'
        anat_mask = layout.get(drop_invalid_filters=False, **anat_entities)
        return anat_to_atlas_composite_transform, anat_n4_corrected, anat_mask




    # find the datasink outputs of the anat_processing pipeline
    perform_func_to_anat_registration = arg_dict['perform_func_to_anat_registration']
    anat_brain_extract_method = BrainExtractMethod[arg_dict['anat_brain_extract_method']]
    use_masks_anat_to_atlas_registration = (not arg_dict['no_masks_anat_to_atlas_registration'])
    use_masks_func_to_anat_registration = (not arg_dict['no_masks_func_to_anat_registration'])
    search_anat_to_atlas_composite_transform, search_anat_n4_corrected, search_anat_mask = get_anat_derivatives(anat,anat_brain_extract_method)
    perform_anat_processing= arg_dict['force_anat_processing']
    if search_anat_to_atlas_composite_transform == []:
        perform_anat_processing = True
    if perform_func_to_anat_registration:
        if search_anat_n4_corrected == []:
            perform_anat_processing = True
        if use_masks_func_to_anat_registration and (search_anat_mask == [] and (anat_brain_extract_method != BrainExtractMethod.USER_PROVIDED_MASK)):
            perform_anat_processing = True



    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    func = funcs[0].path
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if perform_anat_processing:
        wf_anat_processing = init_anat_processing(
            # pipeline input parameters
            name='anat_processing',
            input_anat_file=anat,
            atlas=arg_dict['atlas'],
            atlas_mask=arg_dict['atlas_mask'],

            # anat brain extraction (only necessary if using masks in anat to atlas registration)
            anat_brain_extract_method=BrainExtractMethod[arg_dict['anat_brain_extract_method']],
            # if using REGISTRATION_WITH_INITIAL_MASK or USER_PROVIDED_MASK
            anat_file_mask=anat_mask,
            # if using registration brain extraction
            anat_brain_extract_template=arg_dict['anat_template'],
            anat_brain_extract_template_probability_mask=arg_dict['anat_template_probability_mask'],

            # use extracted masks in anat to atlas registration?
            use_masks_anat_to_atlas_registration=(not arg_dict['no_masks_anat_to_atlas_registration']),

            # node resources
            omp_nthreads=arg_dict['omp_nthreads'],
            mem_gb=arg_dict['mem_gb'],

            # registration processing time
            interpolation=arg_dict['interpolation'],
            reduce_to_float_precision=(not arg_dict['high_precision']),

            # for brainsuite brain extraction
            n4_bspline_fitting_distance=arg_dict['n4_bspline_fitting_distance'],
            diffusionConstant=arg_dict['diffusionConstant'],
            diffusionIterations=arg_dict['diffusionIterations'],
            edgeDetectionConstant=arg_dict['edgeDetectionConstant'],
            radius=arg_dict['radius'],
            dilateFinalMask=(not arg_dict['skipDilateFinalMask']),

            # for datasink
            derivatives_collection_dir=derivatives_dir,
            derivatives_pipeline_name=pipeline_name,
        )
        wf_anat_processing.base_dir = work_dir

        if args.plugin_args:
            execGraph_SS_TV = wf_anat_processing.run(args.plugin, plugin_args=eval(args.plugin_args))
        else:
            execGraph_SS_TV = wf_anat_processing.run(args.plugin)
        #reload layout so that the anat files are present
        layout = BIDSLayout(bids_dir)
        layout.add_derivatives(derivatives_dir)
        search_anat_to_atlas_composite_transform, search_anat_n4_corrected, search_anat_mask = get_anat_derivatives(anat,anat_brain_extract_method)

    anat_to_atlas_composite_transform = search_anat_to_atlas_composite_transform[0].path
    anat_n4_corrected = search_anat_n4_corrected[0].path if search_anat_n4_corrected != [] else None
    #if perform_func_to_anat_registration and use_masks_func_to_anat_registration and (anat_brain_extract_method != BrainExtractMethod.USER_PROVIDED_MASK):
    #    anat_mask = search_anat_mask[0].path
    if anat_brain_extract_method != BrainExtractMethod.USER_PROVIDED_MASK:
        anat_mask = search_anat_mask[0].path if search_anat_mask != [] else None

    wf_func_processing = init_func_processing(
        # pipeline input parameters
        name='func_processing',
        input_func_file=func,
        atlas=arg_dict['atlas'],
        atlas_mask=arg_dict['atlas_mask'],
        label_mapping=arg_dict['label_mapping'],

        # motion correction (usually not necessary because of earbars)
        mc_transform_method=MotionCorrectionTransform[arg_dict['mc_transform_method']],

        # slice timing correction
        perform_stc=arg_dict['perform_stc'],

        # func brain extraction (only necessary if performing func to anat registration and using masks)
        # how to extract a mask for functional image
        func_brain_extract_method=BrainExtractMethod[arg_dict['func_brain_extract_method']],
        # if using REGISTRATION_WITH_INITIAL_MASK or USER_PROVIDED_MASK
        func_file_mask=func_mask,
        # if using registration brain extraction
        func_brain_extract_template=arg_dict['func_template'],
        func_brain_extract_probability_mask=arg_dict['func_template_probability_mask'],

        # func to anat options (usually not necessary because of earbars)
        perform_func_to_anat_registration=arg_dict['perform_func_to_anat_registration'],
        # use extracted masks in func to anat registration?
        use_masks_func_to_anat_registration=(not arg_dict['no_masks_func_to_anat_registration']),

        # node resources
        omp_nthreads=arg_dict['omp_nthreads'],
        mem_gb=arg_dict['mem_gb'],

        # registration processing time
        interpolation=arg_dict['interpolation'],
        reduce_to_float_precision=(not arg_dict['high_precision']),
        # gzip func_to_atlas transform and final registered functional image? reduces image size, increases processing time
        gzip_large_images=arg_dict['gzip_large_images'],

        # for brainsuite brain extraction
        n4_bspline_fitting_distance=arg_dict['n4_bspline_fitting_distance'],
        diffusionConstant=arg_dict['diffusionConstant'],
        diffusionIterations=arg_dict['diffusionIterations'],
        edgeDetectionConstant=arg_dict['edgeDetectionConstant'],
        radius=arg_dict['radius'],
        dilateFinalMask=(not arg_dict['skipDilateFinalMask']),

        # correlation matrix
        correlation_shift_interval_s=arg_dict['correlation_shift_interval_s'],
        correlation_max_shift_s=arg_dict['correlation_max_shift_s'],
        correlation_search_for_neg_corr=arg_dict['correlation_search_for_neg_corr'],

        # for datasink
        derivatives_collection_dir=derivatives_dir,
        derivatives_pipeline_name=pipeline_name,
    )
    wf_func_processing.inputs.inputnode.anat_to_atlas_composite_transform = anat_to_atlas_composite_transform
    wf_func_processing.inputs.inputnode.anat_file = anat_n4_corrected
    wf_func_processing.inputs.inputnode.anat_file_mask = anat_mask

    wf_func_processing.base_dir = work_dir
    if args.plugin_args:
        execGraph_SS_TV = wf_func_processing.run(args.plugin, plugin_args=eval(args.plugin_args))
    else:
        execGraph_SS_TV = wf_func_processing.run(args.plugin)
