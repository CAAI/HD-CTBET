from urllib.request import urlopen
import torch
from torch import nn
import numpy as np
from skimage.morphology import label
import os
from HD_CTBET.paths import folder_with_parameter_files
import shutil
from multiprocessing import Process, Queue
from skimage import measure
import SimpleITK as sitk


def get_params_fname(fold):
    return os.path.join(folder_with_parameter_files, "CT_%d.model" % fold)


def maybe_download_parameters(fold=0, force_overwrite=False):
    """
    Downloads the parameters for some fold if it is not present yet.
    :param fold:
    :param force_overwrite: if True the old parameter file will be deleted (if present) prior to download
    :return:
    """

    assert 0 <= fold <= 4, "fold must be between 0 and 4"

    if not os.path.isdir(folder_with_parameter_files):
        maybe_mkdir_p(folder_with_parameter_files)

    out_filename = get_params_fname(fold)

    if force_overwrite and os.path.isfile(out_filename):
        os.remove(out_filename)
    if force_overwrite and os.path.isfile(out_filename+'.pkl'):
        os.remove(out_filename+'.pkl')

    url='https://zenodo.org/record/7973721/files/CT_%d.model?download=1' % fold
    if not os.path.isfile(out_filename):
        print("Downloading", url, "...")
        data = urlopen(url).read()
        with open(out_filename, 'wb') as f:
            f.write(data)

    url_pkl='https://zenodo.org/record/7973721/files/CT_%d.model.pkl?download=1' % fold
    if not os.path.isfile(out_filename+'.pkl'):
        print("Downloading", url_pkl, "...")
        data = urlopen(url_pkl).read()
        with open(out_filename+'.pkl', 'wb') as f:
            f.write(data)


def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def postprocess_prediction(seg, ct):

    # Load BET
    labels = measure.label(seg, background=0)

    # Load CT
    labelsCT = measure.label(ct>0, background=0)

    # Keep only largest component that overlaps with CT
    max_val = 0
    max_i = None
    for i in np.unique(labels):
        if i > 0 and (_i_inCT := np.logical_and(labels==i, labelsCT)).sum() > max_val:
            max_val = _i_inCT.sum()
            max_i = i
    return (labels==max_i).astype('int8')


def to_one_hot(seg, all_seg_labels=None):
    """ From https://github.com/MIC-DKFZ/nnUNet/blob/77bc485ee025a61feb91cb0a0ed1c61a32a0f39f/nnunet/utilities/one_hot_encoding.py """
    if all_seg_labels is None:
        all_seg_labels = np.unique(seg)
    result = np.zeros((len(all_seg_labels), *seg.shape), dtype=seg.dtype)
    for i, l in enumerate(all_seg_labels):
        result[i][seg == l] = 1
    return result


def preprocess_save_to_queue(preprocess_fn, q, list_of_lists, output_files, classes,
                             transpose_forward):
    """ Modified from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/inference/predict.py """

    errors_in = []
    for i, l in enumerate(list_of_lists):
        try:
            output_file = output_files[i]
            print("preprocessing", output_file)
            d, _, dct = preprocess_fn(l)
            print(d.shape)
            if np.prod(d.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                print(
                    "This output is too large for python process-process communication. "
                    "Saving output temporarily to disk")
                np.save(output_file[:-7] + ".npy", d)
                d = output_file[:-7] + ".npy"
            q.put((output_file, (d, dct)))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print("error in", l)
            print(e)
    q.put("end")
    if len(errors_in) > 0:
        print("There were some errors in the following cases:", errors_in)
        print("These cases were ignored.")
    else:
        print("This worker has ended successfully, no errors to report")


def preprocess_multithreaded(trainer, list_of_lists, output_files, num_processes=2):
    """ Modified from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/inference/predict.py """

    num_processes = min(len(list_of_lists), num_processes)

    classes = list(range(1, trainer.num_classes))
    q = Queue(1)
    processes = []
    for i in range(num_processes):
        pr = Process(target=preprocess_save_to_queue, args=(trainer.preprocess_patient, q,
                                                            list_of_lists[i::num_processes],
                                                            output_files[i::num_processes],
                                                            classes, trainer.plans['transpose_forward']))
        pr.start()
        processes.append(pr)

    try:
        end_ctr = 0
        while end_ctr != num_processes:
            item = q.get()
            if item == "end":
                end_ctr += 1
                continue
            else:
                yield item

    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()  # this should not happen but better safe than sorry right
            p.join()

        q.close()


def subdirs(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


subfolders = subdirs  # I am tired of confusing those


def maybe_mkdir_p(directory):
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i+1])):
            os.mkdir(os.path.join("/", *splits[:i+1]))
