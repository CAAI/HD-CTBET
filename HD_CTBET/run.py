import torch
import numpy as np
import SimpleITK as sitk
import os
from pathlib import Path
from HD_CTBET.data_loading import load_dict, save_segmentation_nifti
from HD_CTBET.utils import (
    postprocess_prediction,
    get_params_fname,
    maybe_download_parameters,
    preprocess_multithreaded
)
import HD_CTBET
from nnunet.training.model_restore import restore_model


def load_model_and_checkpoint_files(folds):
    """
    used for if you need to ensemble the five models of a cross-validation. This will restore the model from the
    checkpoint in fold 0, load all parameters of the five folds in ram and return both. This will allow for fast
    switching between parameters (as opposed to loading them form disk each time).
    This is best used for inference and test prediction
    :param folder:
    :param folds:
    :param mixed_precision: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """

    trainer = restore_model(folds[0]+'.pkl', fp16=None)
    trainer.output_folder = str(Path(folds[0]).parent)
    trainer.output_folder_base = str(Path(folds[0]).parent)
    trainer.update_fold(0)
    trainer.initialize(False)
    print("using the following model files: ", folds)
    all_params = [torch.load(i, map_location=torch.device('cpu')) for i in folds]
    return trainer, all_params


def apply_bet(img, bet, out_fname):
    img_itk = sitk.ReadImage(img)
    img_npy = sitk.GetArrayFromImage(img_itk)
    img_bet = sitk.GetArrayFromImage(sitk.ReadImage(bet))
    img_npy[img_bet == 0] = -1024
    out = sitk.GetImageFromArray(img_npy)
    out.CopyInformation(img_itk)
    sitk.WriteImage(out, out_fname)


def run_hd_ctbet(ct_fnames, output_fnames, mode="accurate",
                 device=0, postprocess=False, do_tta=True, keep_mask=True,
                 overwrite=True, step_size=0.5, all_in_gpu=False,
                 mixed_precision=True):
    """

    :param ct_fnames: str or list/tuple of str
    :param output_fnames: str or list/tuple of str. If list: must have the same length as output_fnames
    :param mode: fast or accurate
    :param device: either int (for device id) or 'cpu'
    :param postprocess: whether to do postprocessing or not. Postprocessing here consists of simply discarding all
    but the largest predicted connected component. Default False
    :param do_tta: whether to do test time data augmentation by mirroring along all axes. Default: True. If you use
    CPU you may want to turn that off to speed things up
    :return:
    """

    list_of_param_files = []

    if mode == 'fast':
        params_file = get_params_fname(0)
        maybe_download_parameters(0)

        list_of_param_files.append(params_file)
    elif mode == 'accurate':
        for i in range(5):
            params_file = get_params_fname(i)
            maybe_download_parameters(i)

            list_of_param_files.append(params_file)
    else:
        raise ValueError("Unknown value for mode: %s. Expected: fast or accurate" % mode)

    assert all([os.path.isfile(i) for i in list_of_param_files]), "Could not find parameter files"

    if not isinstance(ct_fnames, (list, tuple)):
        ct_fnames = [ct_fnames]

    if not isinstance(output_fnames, (list, tuple)):
        output_fnames = [output_fnames]

    assert len(ct_fnames) == len(output_fnames), "ct_fnames and output_fnames must have the same length"

    print("emptying cuda cache")
    torch.cuda.empty_cache()

    print("loading parameters for folds")
    trainer, params = load_model_and_checkpoint_files(list_of_param_files)

    print("starting preprocessing generator")
    preprocessing = preprocess_multithreaded(trainer, [ct_fnames], [output_fnames], 6)

    for preprocessed, in_fname, out_fname in zip(preprocessing, ct_fnames, output_fnames):
        mask_fname = out_fname[:-7] + "_mask.nii.gz"
        if overwrite or (not (os.path.isfile(mask_fname) and keep_mask) or not os.path.isfile(out_fname)):
            print("File:", in_fname)
            print("getting data from preprocessor")
            output_filename, (d, dct) = preprocessed
            if isinstance(d, str):
                print("what I got is a string, so I need to load a file")
                data = np.load(d)
                os.remove(d)
                d = data
            data_dict = load_dict(in_fname)

            """
            try:
                data, data_dict = load_and_preprocess(in_fname)
            except RuntimeError:
                print("\nERROR\nCould not read file", in_fname, "\n")
                continue
            except AssertionError as e:
                print(e)
                continue
            """

            print("prediction (CNN id)...")

            trainer.load_checkpoint_ram(params[0], False)
            softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
                d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
                step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
                mixed_precision=mixed_precision)[1]

            for p in params[1:]:
                trainer.load_checkpoint_ram(p, False)
                softmax += trainer.predict_preprocessed_data_return_seg_and_softmax(
                    d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
                    step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
                    mixed_precision=mixed_precision)[1]

            if len(params) > 1:
                softmax /= len(params)

            transpose_forward = trainer.plans.get('transpose_forward')
            if transpose_forward is not None:
                transpose_backward = trainer.plans.get('transpose_backward')
                softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])

            seg = np.argmax(softmax, 0)

            if postprocess:
                seg = postprocess_prediction(seg, d)

            print("exporting segmentation...")
            save_segmentation_nifti(seg, data_dict, mask_fname)

            apply_bet(in_fname, mask_fname, out_fname)

            if not keep_mask:
                os.remove(mask_fname)
