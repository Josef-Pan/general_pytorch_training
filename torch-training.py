# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch, timm
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time, math, os, copy, tqdm, time, sys, glob, re, shutil, socket, argparse
from datetime import datetime
from torch.utils.data.sampler import SubsetRandomSampler
# import torch.multiprocessing as mp
from src_files.loss_functions.losses import CrossEntropyLS
from volomodel import *
from voloutils import load_pretrained_weights

args, working_path, ip_address = None, None, None


# ✅ 1 Data preparation
def existing_data_set(path_src, path_name_existing):
    def list_dir(train_val, cat):
        dir = os.path.join(path_name_existing, train_val, cat)
        file_list = [os.path.basename(f) for f in sorted(glob.glob(dir + '/*'))]
        return file_list

    path_src_full = os.path.join(working_path, path_src)
    categories = [os.path.basename(d) for d in sorted(glob.glob(path_src_full.rstrip('/') + '/*'))]
    current_img_train = {x: list_dir('train', x) for x in categories}
    current_img_val = {x: list_dir('val', x) for x in categories}
    return current_img_train, current_img_val


def calc_val_number(len_images, split):
    if split < 1.0:  # as percentage
        return int(math.floor(len_images * split / 10) * 10)
    else:  # split > 1.0 # as absolute number
        return int(split)


def do_split(image_set, cat, train_dir, val_dir, unused_dir, train_list_file, val_list_file,
             images_train, images_val, images_train_new=[], images_val_new=[]):
    # print(f'len(image_set) = {len(image_set)}, cat = {cat}, f1={train_list_file}, f2={val_list_file}')
    # print(f'len(images_train) = {len(images_train)}, len(images_val) = {len(images_val)}')
    # print(f'len(images_train_new) = {len(images_train_new)}, len(images_val_new) = {len(images_val_new)}')
    trange_obj = tqdm.tqdm(image_set, bar_format='{l_bar}{bar:40}{r_bar}')
    train, val = 0, 0
    for image_full_path in trange_obj:
        base_name = os.path.basename(image_full_path)
        if base_name in images_train or base_name in images_train_new:
            shutil.copy(image_full_path, os.path.join(train_dir, cat, base_name))
            save_to_file(train_list_file, string_list=[base_name], append=True)
            train += 1
        elif base_name in images_val or base_name in images_val_new:
            shutil.copy(image_full_path, os.path.join(val_dir, cat, base_name))
            save_to_file(val_list_file, string_list=[base_name], append=True)
            val += 1
        else:
            shutil.copy(image_full_path, os.path.join(unused_dir, cat, base_name))
        unused = len(image_set) - train - val
        bar = f"\033[1;36mtrain={train:5d}/val={val:5d}/Unused=\033[31m{unused:5d}\033[0m"
        trange_obj.set_description(bar)
        trange_obj.refresh()  # to show immediately the update
    trange_obj.close()
    print(f'{cat} train={train/len(image_set)*100:.2f}%, val={val/len(image_set)*100:.2f}%,')


def split_into_train_val(path_src, path_target, split=0.2):
    global working_path

    """
    :param path_name: folder containing training pictures, each category in one folder
    :param path_name_splitted: folder containing splitted pictures, will contain 'train' and 'val' subfolder
    :param split: percentage of files for validation dataset
    :return:
    # 🔴 Default action is do incremental slpit unless --wipeall is specfied
    """
    path_src_full = os.path.join(working_path, path_src)
    path_target_full = os.path.join(working_path, path_target)

    print('\033[35mDataset is generated incrementally unless --wipeall is specified\033[0m')
    current_img_train, current_img_val = ({}, {}) if args.wipeall else existing_data_set(path_src, path_target_full)
    print(f'Current dataset train/val')
    for key, value in current_img_train.items():  # current_img_train, current_img_val保存的都是base_name
        print(f'{key}\t{len(current_img_train[key])}/{len(current_img_val[key])}')

    # Delete old subfolders and creat new folders
    [shutil.rmtree(os.path.join(working_path, f), ignore_errors=True) for f in glob.glob(path_target + '/*')]
    os.makedirs(os.path.join(working_path, path_target), exist_ok=True)
    train_dir = os.path.join(working_path, path_target, 'train')
    val_dir = os.path.join(working_path, path_target, 'val')
    unused_dir = os.path.join(working_path, path_target, 'unused')
    [os.makedirs(d) for d in [train_dir, val_dir, unused_dir]]
    subfolders_src = sorted(glob.glob(path_src_full.rstrip('/') + '/*'))
    subfolders_base = [os.path.basename(item) for item in subfolders_src]
    print(subfolders_base)
    for cat in subfolders_base:
        [os.makedirs(os.path.join(d, cat), exist_ok=True) for d in [train_dir, val_dir, unused_dir]]

    image_sets_by_cat = [glob.glob(folder + "/*") for folder in subfolders_src]
    min_length = min([len(set) for set in image_sets_by_cat])  #  train val split on smallest dataset
    val_length = calc_val_number(min_length, split)
    train_length = min_length - val_length
    val_rough = min_length * split if split < 1.0 else split
    print(f'\033[35mmin_length={min_length}, val_length={val_length}, val.org={val_rough:.1f}\033[0m')

    if args.wipeall:
        for folder in subfolders_src:
            cat = os.path.basename(folder)
            train_list_file = os.path.join(working_path, f'train-{cat}.txt')
            val_list_file = os.path.join(working_path, f'val-{cat}.txt')
            os.remove(train_list_file) if os.path.isfile(train_list_file) else ()
            os.remove(val_list_file) if os.path.isfile(val_list_file) else ()
            image_set_1 = sorted(glob.glob(folder + "/*"))
            if split < 1.0:
                print(f'Processing folder {folder} with train {(1-split)*100:.2f}% and val {split*100:.2f}%')
            else:
                train_percentage = (len(image_set_1) - split) / len(image_set_1) * 100
                val_percentage = split / len(image_set_1) * 100
                print(f'Processing folder {folder} with train {train_percentage:.2f}% and val {val_percentage:.2f}%')
            indices_all = np.arange(len(image_set_1))
            indices_val = np.random.choice(indices_all, replace=False, size=(val_length,))
            indices_left = indices_all[np.isin(indices_all, indices_val, invert=True)]
            indices_train = np.random.choice(indices_left, replace=False, size=(train_length,))
            images_train = [os.path.basename(f) for idx, f in enumerate(image_set_1) if idx in indices_train]
            images_val = [os.path.basename(f) for idx, f in enumerate(image_set_1) if idx in indices_val]
            images_train_new, images_val_new = [], []
            do_split(image_set_1, cat, train_dir, val_dir, unused_dir, train_list_file, val_list_file,
                     images_train, images_val, images_train_new=images_train_new, images_val_new=images_val_new)

    elif args.from_listfile:
        for folder in subfolders_src:
            cat = os.path.basename(folder)
            train_list_file = os.path.join(working_path, f'train-{cat}.txt')
            val_list_file = os.path.join(working_path, f'val-{cat}.txt')
            print(f'Processing folder {folder} with train with \033[35m{os.path.basename(val_list_file)}\033[0m')
            if not os.path.isfile(train_list_file):
                print(f'\033[31m{train_list_file} does not exist!\033[0m\nRun with --init ')
                exit(1)
            if not os.path.isfile(val_list_file):
                print(f'\033[31m{val_list_file} does not exist!\033[0m\nRun with --init ')
                exit(1)
            train_images_raw = [line for line in get_file_contents_v3(train_list_file)]
            train_images = [f for f in train_images_raw if os.path.isfile(os.path.join(folder, f))]  # only files exist
            val_images_raw = [line for line in get_file_contents_v3(val_list_file)]
            val_images = [f for f in val_images_raw if os.path.isfile(os.path.join(folder, f))]  # only files exist
            print(f'{train_list_file} has {len(train_images)} valid entries out of {len(train_images_raw)}')
            print(f'{val_list_file} has {len(val_images)} valid entries out of {len(val_images_raw)}')
            os.remove(train_list_file) if os.path.isfile(train_list_file) else ()
            os.remove(val_list_file) if os.path.isfile(val_list_file) else ()
            image_set_1 = sorted(glob.glob(folder + "/*"))
            images_train = [os.path.basename(f) for f in image_set_1 if os.path.basename(f) in train_images]
            images_val = [os.path.basename(f) for f in image_set_1 if os.path.basename(f) in val_images]
            if len(images_train) < train_length:  # 
                number_train_2_append = train_length - len(images_train)
                print(f'Not enough train for {cat}, adding \033[35m{number_train_2_append}\033[0m more')
                bn = os.path.basename
                images_not_used = [bn(f) for f in image_set_1 if bn(f) not in val_images and bn(f) not in train_images]
                idx_add = np.random.choice(len(images_not_used), replace=False, size=(number_train_2_append,))
                images_train_new = [os.path.basename(f) for idx, f in enumerate(images_not_used) if idx in idx_add]
            else:
                images_train_new = []
            if len(images_val) < val_length:  
                number_val_2_append = val_length - len(images_val)
                print(f'Not enough val for {cat}, adding \033[35m{number_val_2_append}\033[0m more')
                bn = os.path.basename
                images_not_used = [bn(f) for f in image_set_1 if bn(f) not in val_images and
                                   bn(f) not in train_images and bn(f) not in images_train_new]
                idx_add = np.random.choice(len(images_not_used), replace=False, size=(number_val_2_append,))
                images_val_new = [os.path.basename(f) for idx, f in enumerate(images_not_used) if idx in idx_add]
            else:
                images_val_new = []
            do_split(image_set_1, cat, train_dir, val_dir, unused_dir, train_list_file, val_list_file,
                     images_train, images_val, images_train_new=images_train_new, images_val_new=images_val_new)

   

    if args.sound:
        try:
            from subprocess import DEVNULL, STDOUT, check_call
            if is_my_mac_on():
                check_call(['ssh', f'josef@{args.my_mac_addr}', '/usr/local/bin/mpg123', args.mp3file],
                           stdout=DEVNULL, stderr=DEVNULL)
            else:
                check_call(['mpg123', args.mp3file], stdout=DEVNULL, stderr=DEVNULL)
        except:
            pass


def data_preparation(data_dir='human.new', batch_sizes={'train': 32, 'val': 16},
                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    from randaugment import RandAugment
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((args.image_w, args.image_w), interpolation=transforms.InterpolationMode.BICUBIC),
            RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((args.image_w, args.image_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    }
    data_full_path = os.path.join(working_path, data_dir)

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_full_path, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    # if args.customsampler:
    if args.balancing:
        from torchsampler import ImbalancedDatasetSampler  # https://github.com/ufoym/imbalanced-dataset-sampler
        dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'],
                                                            sampler=ImbalancedDatasetSampler(image_datasets['train']),
                                                            batch_size=batch_sizes['train'],
                                                            num_workers=0),
                       'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_sizes['val'],
                                                          shuffle=True, num_workers=0), }
    else:
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes[x],
                                                      shuffle=True, num_workers=0)
                       for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    train_images_count = [len(glob.glob(data_full_path.rstrip('/') + '/train/' + c + '/*')) for c in class_names]
    val_images_count = [len(glob.glob(data_full_path.rstrip('/') + '/val/' + c + '/*')) for c in class_names]
    return dataloaders, dataset_sizes, class_names, train_images_count, val_images_count


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ✅ 2 Model preparation
def model_volo_template(nb_classes, pretrained=True, name='volod1'):
    # torch>=1.7.0; torchvision>=0.8.0; timm==0.4.5; tlt==0.1.0; pyyaml; apex-amp
    # volo was trained on imagenet 1000 classes, too weak ‼️
    global img_w
    device = get_device()
    model_selection = {
        'volod1': (volo_d1, 'd1_224_84.2.pth.tar', 224),
        'volod2': (volo_d2, 'd2_224_85.2.pth.tar', 224),
        'volod3': (volo_d3, 'd3_224_85.4.pth.tar', 224),
        'volod4': (volo_d4, 'd4_224_85.7.pth.tar', 224),
        'volod5': (volo_d5, 'd5_224_86.1.pth.tar', 224),
    }

    # create model
    try:
        model_selected, weights_file, img_w = model_selection[name]
    except:
        print(f"\033[35mFound name \033[31m{name} or {weights_file}\033[35m illegal!\033[0m")
        exit(1)

    model = model_selected(num_classes=nb_classes)
    model.aux_logits = False

    # load the pretrained weights
    # change num_classes based on dataset, can work for different image size
    # as we interpolate the position embeding for different image size.
    if pretrained:
        path = os.path.join(args.volo_weight_dir, weights_file)
        load_pretrained_weights(model, path, use_ema=False,
                                strict=False, num_classes=nb_classes)
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)
    return model, name


def model_volod1(nb_classes, pretrained=True):
    return model_volo_template(nb_classes, pretrained, name='volod1')


def model_volod2(nb_classes, pretrained=True):
    return model_volo_template(nb_classes, pretrained, name='volod2')


def model_volod3(nb_classes, pretrained=True):
    return model_volo_template(nb_classes, pretrained, name='volod3')


def model_volod4(nb_classes, pretrained=True):
    return model_volo_template(nb_classes, pretrained, name='volod4')


def model_volod5(nb_classes, pretrained=True):
    return model_volo_template(nb_classes, pretrained, name='volod5')

def unfreeze_from_layer(model, layer_name=''):
    # denseblock4
    unfreeze_from = 9999
    params = [(name, param) for name, param in model.named_parameters()]
    for param in model.parameters():
        param.requires_grad = False
    for idx, (name, param) in enumerate(params):
        # if layer_name in name: unfreeze_from = idx; break
        if name.startswith(layer_name):
            unfreeze_from = idx
            # print(f'\033[31mname={name}, layer_name = {layer_name}\033[0m')
            break
    print(f"Unfreezing from layer = {idx} {name}")
    for idx, (name, param) in enumerate(params):
        if idx >= unfreeze_from:
            param.requires_grad = True


def is_layer_trainable(layer):
    trainable = False
    for param in layer.parameters():  
        if param.requires_grad:
            trainable = True
            break
    return trainable


# ✅ 4 Start training ...
# <editor-fold desc="Training supporting functions">
def mainIPV4Address(server='192.168.1.1'):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((server, 80))
    addr = s.getsockname()[0]
    s.close()
    print(f'mainIPV4Address() = \033[35m{str(addr)}\033[0m')
    return str(addr)


def is_my_mac_on(my_mac_address):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    try:
        s.connect((my_mac_address, 22))
        return True
    except Exception as instance:
        # print(f'\033[31m{str(instance.args)}\033[0m')
        return False


def get_file_contents_v3(file_name, remove_comments=False):  # removing lines starting with '#' if needed
    try:
        with open(file_name, 'r') as f:
            lines = [line.strip() for line in f]  # remove left and right white spaces and '\n'
            lines = [line for line in lines if line]  # exclude empty lines
            lines = [line for line in lines if not line.startswith('#')] if remove_comments else lines
            return lines
    except:
        return []


def changeLRbyFile():
    """
    :return: new lr to be used in next epoch, or None if lr.txt does not contain any legal lr
    """
    # create a file name lr.txt and type new lr, new lr will be loaded next epoch
    # lr.txt will be deleted after using
    file_name = os.path.join(working_path, "lr.txt")
    message = f"[\033[1;35m{os.path.basename(file_name)}\033[0m was deleted afterwards\033[0m]"
    try:
        reader = open(file_name, 'r')
        first_line = reader.readline().replace('\n', ' ')  # Only first line, others ignored
        new_lr = safe_cast_float(first_line)
        if new_lr:
            fmt_string = "\033[32mlr.txt contents = %s\033[1;32m Value = %.2e\033[0m; %s"
            print(fmt_string % (first_line, new_lr, message), end=' ')
            os.remove(file_name);
            return new_lr
        else:
            fmt_string = "\033[32mlr.txt contents = %s\033[1;31m Value illegal\033[0m; %s"
            print(fmt_string % (first_line, message), end=' ')
            os.remove(file_name);
            return None
    except:
        return None


def acc_2_str(array):
    return '[' + ','.join(['%5.2f%%' % (array[n] * 100.0) for n in range(len(array))]) + ']'


def files_2_str(files_kept):
    files_kept = files_kept[-3:]
    return '[' + ','.join(['%s' % files_kept[n] for n in range(len(files_kept))]) + ']'

def log_to_file(string=""):
    file = open(os.path.join(working_path, args.recordfile), 'a')
    file.write(string + '\n')
    file.close()


def save_to_file(file, string_list=[], append=False):
    f = open(file, 'a') if append else open(file, 'w')
    [f.write(line + '\n') for line in string_list]
    f.close()


def log_file_header(epochs, name, optimizer, op_params, unfreeze_from):
    lr, momentum, nesterov = get_optimzer_params(optimizer)
    m = []
    m.append(args.m0) if args.m0 is not None else ()
    m.append(args.m1) if args.m1 is not None else ()
    m.append(args.m2) if args.m2 is not None else ()
    string_m = '[' + ','.join([f'{item:.4f}'.rstrip('0') for item in m]) + ']'
    short_name = name.replace('_in21k', '').replace('_bitm', '')
    config_data = (epochs, short_name, type(optimizer).__name__, lr, momentum, int(nesterov), string_m,
                   args.dropout, args.l2, op_params.gamma, op_params.step)
    str_training_cfg = "epochs=%d @%s %s#%.2e(%.1f,%d),m=%s,drop=%.1f,l2=%.1e,%.2f/%d" % config_data
    dt_string = datetime.now().strftime("\033[1;36m%Y-%m-%d %H:%M:%S\033[0m")
    str_to_log = dt_string + " \033[1;32m" + str_training_cfg + ' ' + unfreeze_from + "\033[0m"
    log_to_file(str_to_log)
    return str_to_log


def safe_cast_float(string, default=0.0):
    try:
        return float(string)
    except (ValueError, TypeError):
        return default


def safe_cast_int(string, default=0):
    try:
        return int(string)
    except (ValueError, TypeError):
        return default


def get_last_epoch(lines):
    if not lines:
        return 0
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    line_last = re.sub(ansi_escape, "", lines[-1])
    splitted = line_last.split(' ')
    if len(splitted) <= 1: return 0
    try:
        return safe_cast_int(splitted[3].split('/')[0])
    except:
        return 0


def get_last_lr(lines):
    if not lines:
        return None
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    line_last = re.sub(ansi_escape, "", lines[-1])
    splitted = line_last.split('=')  
    if len(splitted) <= 1: return None
    try:
        return safe_cast_float(splitted[-1])
    except:
        return None


def get_old_csv_lines(include_header=False, keep_ansi=False):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    try:
        reader = open(os.path.join(working_path, args.recordfile), 'r')
        lines_with_ansi = [line.replace('\n', '') for line in reader.readlines() if len(line) > 0]
        lines_no_ansi = [re.sub(ansi_escape, "", line) for line in lines_with_ansi]
        if include_header and keep_ansi: return lines_with_ansi
        if include_header and (not keep_ansi): return lines_no_ansi
        if (not include_header) and keep_ansi: return lines_with_ansi[1:]
        if (not include_header) and (not keep_ansi): return lines_no_ansi[1:]
        return lines_no_ansi[1:]
    except:
        return []


def hilight_best_acc():
    header_line = get_old_csv_lines(include_header=True, keep_ansi=True)[0]
    if not header_line: return
    lines_splitted = [line.split(' ') for line in get_old_csv_lines()]  # no header, no ansi
    # acc_values = [line[7].strip('%') for line in lines_splitted]
    acc_values = [line[7].split('/')[-1].strip('%') for line in lines_splitted]
    max_acc = max(acc_values)

    safe_remove(os.path.join(working_path, args.recordfile))
    log_to_file(header_line)
    for line_splitted in lines_splitted:
        acc_value = line_splitted[7].split('/')[-1].strip('%')
        if acc_value != str(max_acc):
            new_line = ' '.join(line_splitted)  # old line, not changed
        else:
            new_line = '\033[32m' + ' '.join(line_splitted) + '\033[0m'  # changed to green colour
        log_to_file(new_line)


def get_weights_file():  # 
    working_path = os.path.dirname(os.path.realpath(__file__))
    weights_file = os.path.join(working_path, args.weightsfile)
    return (None, weights_file)[os.path.exists(weights_file)]


def safe_remove(file):
    try:
        os.remove(file) if os.path.isfile(file) else ()
    except Exception as instance:
        print(f'\033[31m{str(instance.args)}\033[0m')
        return False


def keep_only_best_models(files_keep=5):
    #   file format:  resnetv2_50x3_bitm_in21k-005-[78.95,21.05,84.21,63.16,78.95]-73.33@65.26.pth
    weights_dir = os.path.abspath(args.weights_dir)
    save_path = os.path.join(weights_dir, ip_address)

    def del_low_value_files(filename_wild_card, exception):
        all_h5_files = glob.glob(save_path.rstrip('/') + f'/{filename_wild_card}')
        all_h5_files = [f for f in all_h5_files if exception not in all_h5_files]
        all_val_acc = sorted([safe_cast_float(file[-9:-4]) for file in all_h5_files], reverse=True)  # 从大到小排序
        best_acc_values = all_val_acc[0:files_keep]  # 
        print(f'best_acc_values = {best_acc_values}') if args.debug else ()
        [safe_remove(file) for file in all_h5_files if safe_cast_float(file[-9:-4]) not in best_acc_values]

    def fitler_item_for_epoch_acc(file_name):
        file_name_splitted = file_name.split('-')
        epoch_and_acc = [item for item in file_name_splitted if 'pth' in item or re.match('^\d\d\d$', item)]
        return '-'.join(epoch_and_acc)

    del_low_value_files("*.pth", exception=args.weightsfile)  # weight.pth always kept

    all_h5_files = [os.path.basename(file) for file in glob.glob(save_path.rstrip('/') + '/*.pth')]
    all_h5_files = [f for f in all_h5_files if args.weightsfile not in all_h5_files]  # weight.pth excepted
    all_h5_files = [fitler_item_for_epoch_acc(f) for f in all_h5_files]
    print(f'all_h5_files = {all_h5_files}') if args.debug else ()
    all_h5_files.sort(key=lambda filename: safe_cast_float(filename[-9:-4]))
    return all_h5_files


def get_optimzer_params(optimizer):  # should return last group, multiple lr may be used
    try:
        params = optimizer.state_dict()['param_groups'][-1]
        lr = params.get('lr', 0.0)
        momentum = params.get('momentum', 0.0)
        nesterov = params.get('nesterov', False)
        return lr, momentum, nesterov
    except:
        return 0.0, 0.0, False


def set_SGD_params(optimizer, lr, momentum=0.9, nesterov=False):
    new_optimizer = copy.deepcopy(optimizer)
    try:
        state_dict = new_optimizer.state_dict()
        params = state_dict['param_groups'][0]
        params['lr'] = lr
        params['momentum'] = momentum
        params['nesterov'] = nesterov
        new_optimizer.load_state_dict(state_dict)
        return new_optimizer
    except:
        return None


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    untrainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, untrainable


# </editor-fold>

def summary(model, verbose=0):  
    trainable, untrainable = count_parameters(model)
    print(f"Trainable params = \033[32m{trainable:,}\033[0m Untrainable parmas = \033[31m{untrainable:,}\033[0m")
    trainable_layer_names = []
    for name, param in model.named_parameters():
        if ('weight' in name and param.requires_grad):
            # layer_name = name.replace('weight', '').replace('features', '').strip('.')
            layer_name = name.replace('weight', '').strip('.')
            trainable_layer_names.append(layer_name)

    print(f'\033[35m{trainable_layer_names}\033[0m') if args.debug else ()

    def short_layer_name(full_layer_name):
        splitted = full_layer_name.split('.')
        return splitted[0] if splitted[1] == 'weight' else '.'.join([splitted[0], splitted[1]])

    def is_trainable(l):
        return any([item.startswith(l) for item in trainable_layer_names])

    def show_trainable_10():
        print("Trainable layers = ", end=' ')
        for idx, layer in enumerate(trainable_layer_names[:5]):
            print(f"\033[32m{layer}\033[0m", end=', ') if idx == 0 else print(f"{layer}", end=', ')
        print(f"\033[32m ...... \033[0m", end=' ')
        for layer in trainable_layer_names[-5:]:
            print(f"{layer}", end=', ')
        print(f"\033[32mtotal={len(trainable_layer_names)}\033[0m")

    def show_trainable_short():
        all_layer_names = [name for name, param in model.named_parameters() if 'weight' in name]
        layers = [short_layer_name(layer) for layer in all_layer_names]
        layers = list(dict.fromkeys(layers))  # removed duplicates
        [print(f'\033[32m{l}\033[0m' if is_trainable(l) else f'{l}', end=',') for l in layers]
        print('')

    def show_trainale_all():
        print("Trainable layers = ", end=' ')
        for idx, layer in enumerate(trainable_layer_names):
            print(f"\033[32m{layer}\033[0m", end=', ') if idx == 0 else print(f"{layer}", end=', ')
        print(f"\033[32mtotal={len(trainable_layer_names)}\033[0m")

    if verbose == 0:
        show_trainable_short()  # if len(trainable_layer_names) > 10 else show_trainale_all()

    if verbose == 1:
        show_trainale_all()

    if verbose == 2:
        show_trainale_all()
        print("All layers = ")
        all_layer_names = [name for name, param in model.named_parameters() if 'weight' in name]
        for idx, name in enumerate(all_layer_names):
            layer_name = name.replace('weight', '').replace('features', '').strip('.')[:30]
            if ((idx) % 4 == 0): print(f"{idx:03d}", end=':')
            print(f"{layer_name:<30} ", end=' ')
            if ((idx + 1) % 4 == 0): print('')
        print(f"\033[32mtotal={len(all_layer_names)}\033[0m")
    if verbose == 3:
        show_trainale_all()
        print(model)


#########################################################################################################
class OptimizerParams:
    def __init__(self, lr, step, gamma, momentum=0.9, nesterov=True, weight_decay=2e-6, threshold=0.91,
                 min_lr=3e-6):
        self.name = "josef_Optimzer"
        self.lr = lr
        self.step = step
        self.gamma = gamma
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.threshold = threshold
        self.min_lr = min_lr


class MyTrainingConfig:
    def __init__(self, base_model, unfreeze_from, op_params, epochs, batch=None, data_dir='human.new'):
        self.name = "josef_TrainingParams"
        self.base_model = base_model
        self.unfreeze_from = unfreeze_from
        self.op_params = op_params
        self.epochs = epochs
        if not batch:
            self.batch_sizes = {'train': 32, 'val': 16}
        else:
            self.batch_sizes = batch
        self.data_dir = data_dir
        self.last_epoch = 0
        # following params intialized in training_prepare
        self.model_name = None
        self.dataloaders, self.dataset_sizes, self.class_names = None, None, None
        self.device = None
        self.train_loss_fn, self.val_loss_fn = None, None
        self.optimizer, self.scheduler = None, None


# ❤️ SDG multiple lr for layers ...
def sgd_multiple_lr(model, cfg):  # only for Big Transfer Models and volo models
    lr, momentum, nesterov = cfg.op_params.lr, cfg.op_params.momentum, cfg.op_params.nesterov
    weight_decay, weight_decay_all_layers = cfg.op_params.weight_decay, args.l2_all_layers
    weight_decay_last_layer = args.l2_last_layer
    def dict_p(layer_params, layer_lr, layer_w_decay):
        return {"params": layer_params, "lr": layer_lr, "momentum": momentum, "nesterov": nesterov,
                'weight_decay': layer_w_decay}

    resnet_names = ['resnet18', 'resnet34', 'resnet50', 'resnet200d']
    tt = is_layer_trainable
    
    if cfg.model_name.startswith('volo'):  # network.0 - network.4, post_network, aux_head, norm, head
        try:
            networks = model.network
        except:
            return optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        layers, lr_diff = len(networks), args.lr_diff  # lr_diff 
        m = [math.exp(math.log(lr_diff) / (layers - 1) * i) / lr_diff for i in range(layers)]
        m[0], m[1] = args.m0 if args.m0 is not None else m[0], args.m1 if args.m1 is not None else m[1]
        m[2] = args.m2 if args.m2 is not None else m[2]
        m[3] = args.m2 if args.m2 is not None else m[3]
        if not weight_decay_all_layers:
            dicts = [dict_p(model.network[i].parameters(), lr * m[i], 0) for i in range(layers) if tt(model.network[i])]
            dicts.append(dict_p(model.post_network.parameters(), lr, 0)) if tt(model.post_network) else ()
            dicts.append(dict_p(model.aux_head.parameters(), lr, 0)) if tt(model.aux_head) else ()
            # dicts.append(dict_p(model.norm.parameters(), lr, 0))
        else:
            dicts = [dict_p(model.network[i].parameters(), lr * m[i], weight_decay * m[i]) for i in range(layers)
                     if tt(model.network[i])]
            dicts.append(dict_p(model.post_network.parameters(), lr, weight_decay)) if tt(model.post_network) else ()
            dicts.append(dict_p(model.aux_head.parameters(), lr, weight_decay)) if tt(model.aux_head) else ()
        dicts.append(dict_p(model.norm.parameters(), lr, weight_decay)) if tt(model.norm) else ()
        dicts.append(dict_p(model.head.parameters(), lr, weight_decay)) if tt(model.head) else ()
        return optim.SGD(dicts)
    elif cfg.model_name.startswith('vit_') or cfg.model_name.startswith('mixer_'):  # blocks0 - blocks23, norm, head
        try:
            blocks = model.blocks
        except:
            return optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        layers, lr_diff = len(blocks), args.lr_diff  # lr_diff means last trainable layer lr devided by first
        m = [math.exp(math.log(lr_diff) / (layers - 1) * i) / lr_diff for i in range(layers)]
        m[0], m[1] = args.m0 if args.m0 is not None else m[0], args.m1 if args.m1 is not None else m[1]
        m[2] = args.m2 if args.m2 is not None else m[2]
        if not weight_decay_all_layers:
            dicts = [dict_p(model.blocks[i].parameters(), lr * m[i], 0) for i in range(layers) if tt(model.blocks[i])]
            # dicts.append(dict_p(model.norm.parameters(), lr, 0))
        else:
            dicts = [dict_p(model.blocks[i].parameters(), lr * m[i], weight_decay * m[i]) for i in range(layers)
                     if tt(model.blocks[i])]
        dicts.append(dict_p(model.norm.parameters(), lr, weight_decay)) if tt(model.norm) else ()
        dicts.append(dict_p(model.head.parameters(), lr, weight_decay)) if tt(model.head) else ()
        return optim.SGD(dicts)
    elif any([cfg.model_name.startswith(n) for n in resnet_names]):  # layer1 - layer4, fc
        layers, lr_diff = 4, args.lr_diff  # lr_diff means last trainable layer lr devided by first
        m = [math.exp(math.log(lr_diff) / (layers - 1) * i) / lr_diff for i in range(layers)]
        m[0], m[1] = args.m0 if args.m0 is not None else m[0], args.m1 if args.m1 is not None else m[1]
        if not weight_decay_all_layers:
            dicts = [dict_p(model.layer1.parameters(), lr * m[0], 0),
                     dict_p(model.layer2.parameters(), lr * m[1], 0),
                     dict_p(model.layer3.parameters(), lr * m[2], 0),
                     dict_p(model.layer4.parameters(), lr * m[3], 0)]
        else:
            dicts = [dict_p(model.layer1.parameters(), lr * m[0], weight_decay * m[0]),
                     dict_p(model.layer2.parameters(), lr * m[1], weight_decay * m[1]),
                     dict_p(model.layer3.parameters(), lr * m[2], weight_decay * m[2]),
                     dict_p(model.layer4.parameters(), lr * m[3], weight_decay * m[3])]
        dicts.append(dict_p(model.fc.parameters(), lr, weight_decay))
        return optim.SGD(dicts)
    elif cfg.model_name.startswith('densenet169') or cfg.model_name.startswith('densenet201'):
        layers, lr_diff = 4, args.lr_diff  # denseblock1 - denseblock4, norm5, classifier
        m = [math.exp(math.log(lr_diff) / (layers - 1) * i) / lr_diff for i in range(layers)]
        m[0], m[1] = args.m0 if args.m0 is not None else m[0], args.m1 if args.m1 is not None else m[1]
        if not weight_decay_all_layers:
            dicts = [dict_p(model.features.denseblock1.parameters(), lr * m[0], 0),
                     dict_p(model.features.denseblock2.parameters(), lr * m[1], 0),
                     dict_p(model.features.denseblock3.parameters(), lr * m[2], 0),
                     dict_p(model.features.denseblock4.parameters(), lr * m[3], 0)]
        else:
            dicts = [dict_p(model.features.denseblock1.parameters(), lr * m[0], weight_decay * m[0]),
                     dict_p(model.features.denseblock2.parameters(), lr * m[1], weight_decay * m[1]),
                     dict_p(model.features.denseblock3.parameters(), lr * m[2], weight_decay * m[2]),
                     dict_p(model.features.denseblock4.parameters(), lr * m[3], weight_decay * m[3])]
        dicts.append(dict_p(model.features.norm5.parameters(), lr, weight_decay))
        dicts.append(dict_p(model.classifier.parameters(), lr, weight_decay))
        return optim.SGD(dicts)
    else:  # mixed_lr for other models not realised yet
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)


def policy_volod1():
    data_dir, batch_sizes = 'human.new', {'train': args.batch if args.batch else 32, 'val': 16}
    base_model, unfreeze_from = model_volod1, 'network.1'
    lr = args.lr if args.lr else 4.0e-4
    optimizer = OptimizerParams(lr=lr, step=1, gamma=0.99, weight_decay=args.l2)
    training_cfg = MyTrainingConfig(base_model, unfreeze_from, optimizer, 200, batch_sizes, data_dir)
    return training_cfg


def policy_volod2():
    data_dir, batch_sizes = 'human.new', {'train': args.batch if args.batch else 32, 'val': 16}
    base_model, unfreeze_from = model_volod2, 'network.1'
    lr = args.lr if args.lr else 4.0e-4
    optimizer = OptimizerParams(lr=lr, step=2, gamma=0.99, weight_decay=args.l2)
    training_cfg = MyTrainingConfig(base_model, unfreeze_from, optimizer, 200, batch_sizes, data_dir)
    return training_cfg


def policy_volod3():
    data_dir, batch_sizes = 'human.new', {'train': args.batch if args.batch else 32, 'val': 16}
    base_model, unfreeze_from = model_volod3, 'network.0'
    lr = args.lr if args.lr else 3.0e-4
    optimizer = OptimizerParams(lr=lr, step=1, gamma=0.99, weight_decay=args.l2)
    training_cfg = MyTrainingConfig(base_model, unfreeze_from, optimizer, 200, batch_sizes, data_dir)
    return training_cfg


def policy_resnet50():
    data_dir, batch_sizes = 'human.new', {'train': args.batch if args.batch else 32, 'val': 16}
    base_model, unfreeze_from = model_resnet50, 'layer1'
    lr = args.lr if args.lr else 4.0e-4
    optimizer = OptimizerParams(lr=lr, step=2, gamma=0.99, weight_decay=args.l2)
    training_cfg = MyTrainingConfig(base_model, unfreeze_from, optimizer, 12, batch_sizes, data_dir)
    return training_cfg


def policy_resnet34():
    data_dir, batch_sizes = 'human.new', {'train': args.batch if args.batch else 32, 'val': 16}
    base_model, unfreeze_from = model_resnet34, 'layer1'
    lr = args.lr if args.lr else 4.0e-4
    optimizer = OptimizerParams(lr=lr, step=1, gamma=0.99, weight_decay=args.l2)
    training_cfg = MyTrainingConfig(base_model, unfreeze_from, optimizer, 4, batch_sizes, data_dir)
    return training_cfg


def policy_test():
    data_dir, batch_sizes = 'data_hymenoptera', {'train': args.batch if args.batch else 32, 'val': 16}
    base_model, unfreeze_from = model_resnetv2_101x3_bitm_in21k, 'stages.1'
    lr = args.lr if args.lr else 1.68e-5
    optimizer = OptimizerParams(lr=lr, step=2, gamma=0.98, weight_decay=args.l2)
    training_cfg = MyTrainingConfig(base_model, unfreeze_from, optimizer, 100, batch_sizes, data_dir)
    return training_cfg


def policyUnknownPC():
    print("\033[1;31mOrdinateur inconnu, au revoir!\033[0m");
    exit(0)


def set_training_policies():
    training_policies = {
        "192.168.1.181": policy_volod1,  # For testing purpose
        "192.168.1.182": policy_volod1,  # For testing purpose

        "192.168.1.185": policy_volod1,  # For testing
        "192.168.1.186": policy_volod1,  # policy_resnet200d

        "192.168.1.187": policy_volod2,  # policy_bit_mr_152x2_stages3
        "192.168.1.188": policy_volod2,

        "192.168.1.197": policy_volod3,  
        "192.168.1.198": policy_volod3,  

        "192.168.1.191": policy_test,
        "192.168.1.192": policy_test,
    }
    return training_policies


######################################################################################################### ❤️❤️
def process_resume(model, cfg, keep_weights, resume_lr):
    existing_log_lines = []
    if keep_weights:
        existing_log_lines = get_old_csv_lines()
        weight_file = get_weights_file()
        if weight_file:
            try:
                model.load_state_dict(torch.load(weight_file))
            except:
                print(f"\033[1;31m{args.weightsfile} does not match model, exiting...");
                exit(1)
            cfg.last_epoch = get_last_epoch(existing_log_lines)
            last_lr = get_last_lr(existing_log_lines)
            fmt_string = "\033[1;35mLoading %s into model, continuing from \033[1;32mepoch %d\033[0m"
            print(fmt_string % (os.path.basename(weight_file), cfg.last_epoch + 1))
            if resume_lr and last_lr:  # 读取正确，否则是None
                if args.multiple_lr:
                    cfg.op_params.lr = last_lr
                    cfg.optimizer = sgd_multiple_lr(model, cfg)
                else:
                    cfg.optimizer = optim.SGD(model.parameters(), lr=last_lr, momentum=cfg.op_params.momentum,
                                              nesterov=cfg.op_params.nesterov, weight_decay=cfg.op_params.weight_decay)
                cfg.scheduler = lr_scheduler.StepLR(cfg.optimizer, step_size=cfg.op_params.step,
                                                    gamma=cfg.op_params.gamma)
                print(f"\033[1;35mLearning rate resumed as \033[1;32m{last_lr:.2e}\033[0m")
        else:
            print(f"\033[1;31mLoading {args.weightsfile} into model failed, training from start\033[0m")

    safe_remove(os.path.join(working_path, args.recordfile))
    weights_dir = os.path.abspath(args.weights_dir)
    save_path = os.path.join(weights_dir, ip_address)
    if not (keep_weights and cfg.last_epoch > 0):
        [os.remove(f) for f in glob.glob(save_path.rstrip('/') + '/*.pth')]
    return model, cfg, existing_log_lines


def process_log(cfg, keep_weights, last_epoch, existing_log_lines):
    str_logged = log_file_header(cfg.epochs, cfg.model_name, cfg.optimizer, cfg.op_params, cfg.unfreeze_from)
    print(str_logged)
    print(f'\033[35mkeep_weights = {keep_weights}\t last_epoch = {last_epoch}\033[0m')
    if keep_weights and last_epoch > 0:
        # print(f'\033[35m{existing_log_lines}\033[0m')
        [log_to_file(line) for line in existing_log_lines]
        hilight_best_acc()


def process_lr_txt(model, cfg):
    lr_from_lr_txt = changeLRbyFile()  #
    if lr_from_lr_txt:
        print(f'lr_from_lr_txt = {lr_from_lr_txt}')
        if args.multiple_lr:
            cfg.op_params.lr = lr_from_lr_txt
            cfg.optimizer = sgd_multiple_lr(model, cfg)
        else:
            cfg.optimizer = optim.SGD(model.parameters(), lr=lr_from_lr_txt, momentum=cfg.op_params.momentum,
                                      nesterov=cfg.op_params.nesterov, weight_decay=cfg.op_params.weight_decay)
        cfg.scheduler = lr_scheduler.StepLR(cfg.optimizer, step_size=cfg.op_params.step,
                                            gamma=cfg.op_params.gamma)
        lr, momentum, nesterov = get_optimzer_params(cfg.optimizer)
        print("\033[1;35mLR has been changed to %.2e\033[0m" % lr)


def training_prepare(keep_weights=False, resume_lr=True, summary_only=False, verbose=0):
    training_policies = set_training_policies()
    cfg = training_policies.get(ip_address, policy_test)()
    print(ip_address, cfg.data_dir, cfg.batch_sizes)
    _, _, cfg.class_names, _, _ = data_preparation(cfg.data_dir, cfg.batch_sizes)
    cfg.device = get_device()
    model, cfg.model_name = cfg.base_model(len(cfg.class_names))
    if cfg.model_name.startswith('volo'):
        mean = (0.485, 0.456, 0.406)  # IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)  # IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        print(f'\033[32m volo mean={mean}\tstd={std}\033[0m')
        cfg.dataloaders, cfg.dataset_sizes, cfg.class_names, train_images_count, val_images_count = \
            data_preparation(cfg.data_dir, cfg.batch_sizes, mean, std)
    else:
        cfg.dataloaders, cfg.dataset_sizes, cfg.class_names, train_images_count, val_images_count = \
            data_preparation(cfg.data_dir, cfg.batch_sizes)

    print(f"\033[1;35m{cfg.model_name}, {cfg.class_names} img_w = {args.image_w}, dropout rate = {args.dropout}\033[0m")
    unfreeze_from_layer(model, layer_name=cfg.unfreeze_from)
    summary(model, verbose=verbose)
    (lambda x: x, exit)[summary_only](0)
    # cfg.train_loss_fn, cfg.val_loss_fn = CrossEntropyLS(0.2), nn.CrossEntropyLoss()
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    # https://discuss.pytorch.org/t/weights-in-weighted-loss-nn-crossentropyloss/69514/2
    print(f'\033[35mtrain_images_count = {train_images_count}\033[0m')
    print(f'\033[35mval_images_count   = {val_images_count}\033[0m')
    if args.normalize:
        normed_weights = [1 - (x / sum(train_images_count)) for x in train_images_count]
        normed_weights = [x / (sum(normed_weights) / len(normed_weights)) for x in normed_weights]  # make sum to be 5.0
        normed_weights = torch.FloatTensor(normed_weights).to(cfg.device)
        print(f'\033[35mnormed_weights = {normed_weights}\033[0m')
        cfg.train_loss_fn, cfg.val_loss_fn = nn.CrossEntropyLoss(weight=normed_weights), nn.CrossEntropyLoss()
    elif args.crossentropyls:
        cfg.train_loss_fn, cfg.val_loss_fn = CrossEntropyLS(0.2), nn.CrossEntropyLoss()
    else:
        cfg.train_loss_fn, cfg.val_loss_fn = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
    if args.multiple_lr:
        cfg.optimizer = sgd_multiple_lr(model, cfg)
        param_groups = cfg.optimizer.state_dict()['param_groups']
        [print(f"mixed lr {idx:03d}\t{p['lr']:.2e}\t{p['momentum']:.1f}\t{p['nesterov']}\tl2={p['weight_decay']:.2e}")
         for idx, p in enumerate(param_groups)]
    else:
        cfg.optimizer = optim.SGD(model.parameters(), lr=cfg.op_params.lr, momentum=cfg.op_params.momentum,
                                  nesterov=cfg.op_params.nesterov, weight_decay=cfg.op_params.weight_decay)
    cfg.scheduler = lr_scheduler.StepLR(cfg.optimizer, step_size=cfg.op_params.step, gamma=cfg.op_params.gamma)

    model, cfg, existing_log_lines = process_resume(model, cfg, keep_weights, resume_lr)
    print(f'\033[35mexisting_log_lines={existing_log_lines}\033[0m') if args.debug else ()
    process_log(cfg, keep_weights, cfg.last_epoch, existing_log_lines)
    return model, cfg


def actions_after_each_opoch(model, cfg, epoch, correct_pred, total_pred,
                             running_loss, running_corrects, lr, train_acc):
    accuracy_all = []
    for classname, correct_count in correct_pred.items():
        accuracy_per_calss = float(correct_count) / total_pred[classname]
        accuracy_all.append(accuracy_per_calss)
        epoch_loss, val_acc = running_loss / cfg.dataset_sizes['val'], running_corrects / cfg.dataset_sizes['val']

    # Logging ##############################################################
    dt_string = datetime.now().strftime("\033[1;36m%Y-%m-%d %H:%M:%S\033[0m")
    epoch_string = " Epoch %03d/%03d " % (epoch, cfg.last_epoch + cfg.epochs)
    val_string = '%s Loss: %.4f Acc: %05.2f%%/%05.2f%%' % ('val  ', epoch_loss, train_acc * 100.0, val_acc * 100.0)
    lr_string = ' lr = %.2e ' % lr
    val_log = val_string.lstrip('val').lstrip(' ') + ' '
    log_to_file(dt_string + epoch_string + val_log + acc_2_str(accuracy_all) + lr_string)

    # Weights ##############################################################
    acc_all_string = '[' + ','.join(['%5.2f' % (item * 100.0) for item in accuracy_all]) + ']'
    short_name = cfg.model_name.replace('_in21k', '').replace('_bitm', '')
    weight_file = f'{short_name}-{epoch:03d}-{acc_all_string}-{train_acc*100.0:5.2f}@{val_acc*100:5.2f}.pth'
    print(f'\033[35mweight_file = {weight_file}\033[0m') if args.debug else ()
    # resnetv2_50x3_bitm_in21k-005-[78.95,21.05,84.21,63.16,78.95]-73.33@65.26.pth
    weights_dir = os.path.abspath(args.weights_dir)
    save_path = os.path.join(weights_dir, ip_address)
    os.makedirs(save_path, exist_ok=True) if not os.path.isdir(save_path) else ()
    latest_weight = os.path.join(save_path, weight_file)
    torch.save(model.state_dict(), latest_weight)
    resume_weight = os.path.join(working_path, args.weightsfile)  # save to working path
    torch.save(model.state_dict(), resume_weight)  # Make a copy of latest weights to resume from
    pth_files_kept = keep_only_best_models(files_keep=7)
    print(f"\033[32m{val_string} {acc_2_str(accuracy_all)} {files_2_str(pth_files_kept)}\033[0m")
    hilight_best_acc()
    csv_in_working_path = os.path.join(working_path, args.recordfile)
    csv_in_save_path = os.path.join(save_path, args.recordfile)
    shutil.copy(csv_in_working_path, csv_in_save_path)  # make a copy of the csv file
    shutil.copy(csv_in_working_path, os.path.join('/tmp/', args.recordfile))  # make a second copy of the csv file
    if args.multiple_lr:
        print(f'Epoch {epoch:03d} lr for layers = \033[35m[', end=' ')
        param_groups = cfg.optimizer.state_dict()['param_groups']
        [print(f"{value['lr']:.2e},", end=' ') for value in param_groups]
        print(']\033[0m')
    if args.sound:
        try:
            from subprocess import DEVNULL, STDOUT, check_call
            if is_my_mac_on(args.my_mac_addr):
                check_call(['ssh', f'josef@{args.my_mac_addr}', '/usr/local/bin/mpg123', args.mp3file],
                           stdout=DEVNULL, stderr=DEVNULL)
            else:
                check_call(['mpg123', args.mp3file], stdout=DEVNULL, stderr=DEVNULL)
        except:
            pass


def sp_training_worker(model, cfg):
    for epoch in range(cfg.last_epoch + 1, cfg.last_epoch + cfg.epochs + 1):  # loop over the dataset multiple times
        process_lr_txt(model, cfg)
        optimizer, train_loss_fn, val_loss_fn = cfg.optimizer, cfg.train_loss_fn, cfg.val_loss_fn
        # Training #############################################################
        running_loss, running_corrects = 0.0, 0
        str_date = datetime.now().strftime("%m-%d %H:%M")
        lr, _, _ = get_optimzer_params(optimizer)
        trange_obj = tqdm.tqdm(cfg.dataloaders['train'], bar_format='{l_bar}{bar:40}{r_bar}')
        is_volo_model = cfg.model_name.startswith('volo')
        batch_size = cfg.batch_sizes['train']
        model.train()  # Set model to training mode
        for inputs, labels in trange_obj:
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                if is_volo_model:  # volo has multiple outputs
                    outputs = outputs[0]
                _, preds = torch.max(outputs, 1)
                loss = train_loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
            # statistics train
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            l_avg = running_loss / batch_size / (trange_obj.n + 1)
            trange_obj.set_description(f'\033[1;36m{str_date} epoch{epoch:3d} loss={l_avg:.4f} lr={lr:.2e}\033[0m')
            trange_obj.refresh()  # to show immediately the update
            time.sleep(0.001)
        trange_obj.close()
        epoch_loss, epoch_acc = running_loss / cfg.dataset_sizes['train'], running_corrects / cfg.dataset_sizes['train']
        print('\033[36m%s Loss: %.4f Acc: %05.2f%% lr=%.2e\033[0m' % ('train', epoch_loss, epoch_acc * 100.0, lr))

        model.eval()  # Set model to evaluate mode
        trange_obj = tqdm.tqdm(cfg.dataloaders['val'], bar_format='{l_bar}{bar:40}{r_bar}')
        running_loss, running_corrects = 0.0, 0
        str_date = datetime.now().strftime("%m-%d %H:%M")
        batch_size = cfg.batch_sizes['val']
        correct_pred = {classname: 0 for classname in cfg.class_names}
        total_pred = {classname: 0 for classname in cfg.class_names}
        for inputs, labels in trange_obj:
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad()
            torch.no_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = cfg.val_loss_fn(outputs, labels)
            for label, prediction in zip(labels, preds):
                if label == prediction:
                    correct_pred[cfg.class_names[label]] += 1
                total_pred[cfg.class_names[label]] += 1
            # statistics val
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            bar = "\033[1;32m%s epoch%3d loss=%.4f\033[0m" % (
                str_date, epoch, running_loss / batch_size / (trange_obj.n + 1))
            trange_obj.set_description(bar)
            trange_obj.refresh()  # to show immediately the update
            time.sleep(0.001)
        trange_obj.close()

        lr, _, _ = get_optimzer_params(optimizer)
        actions_after_each_opoch(model, cfg, epoch, correct_pred,
                                 total_pred, running_loss, running_corrects, lr, epoch_acc)
        if lr > cfg.op_params.min_lr: cfg.scheduler.step()  # ❗❗❗


#########################################################################################################
def parse_arguments():
    global ip_address
    ip_address = mainIPV4Address()
    parser = argparse.ArgumentParser(description='Generic training based on timm modules',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--init", help="Initialize training dataset from raw data set and exit", action="store_true")
    group.add_argument("--sum", help="Summarize model with verbose level and exit",
                       choices=[0, 1, 2, 3], type=int)
    group.add_argument('--train', default=True, help='Traing the model', action="store_true")
    parser.add_argument("-r", "--resume", default=False, help="Resume training or training from start",
                        action="store_true")
    parser.add_argument("--wipeall", help="Creat train/val from scratch", action="store_true")
    parser.add_argument("--from_listfile", help="Creat train/val from a list file, eg. va-boy.txt", action="store_true")
    parser.add_argument("--from_listval", help="Only use val-*.txt, all rest as train", action="store_true")
    parser.add_argument("--balancing", help="Imbalanced data correction at init and sampling", action="store_true")
    if ip_address in ['192.168.1.187', '192.168.1.188', '192.168.1.197', '192.168.1.198']:
        parser.add_argument("--sound", default=True, help="Playing sound after each epoch", action="store_true")
    else:
        parser.add_argument("--sound", help="Playing sound after each epoch", action="store_true")
    parser.add_argument("--batch", help="Custom batch instead of predefined", type=int)
    parser.add_argument("--l2_all_layers", help="🔴Weight decay for all layers, be careful", action="store_true")
    parser.add_argument("--l2_last_layer", help="🔴Weight decay for last layer, be careful", action="store_true")
    parser.add_argument("--lr", help="Custom learning rate value instead of predefined", type=float)
    parser.add_argument("--lr_diff", default=10.0, help="Ratio between lr of last layer and first layer", type=float)
    if ip_address in ['192.168.1.187', '192.168.1.188', '192.168.1.197', '192.168.1.198']:
        parser.add_argument("--multiple_lr", default=True, help="Use different lr for layers", action="store_true")
        parser.add_argument("--crossentropyls", default=True, help="Use CrossEntropyLS(0.2)", action="store_true")
    else:
        parser.add_argument("--multiple_lr", help="Use different lr for layers", action="store_true")
        parser.add_argument("--crossentropyls", help="Use CrossEntropyLS(0.2)", action="store_true")

    # Important params to adjust for better results ‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️
    parser.add_argument("--m0", help="First layer multiplier", type=float)
    parser.add_argument("--m1", help="Second layer multiplier", type=float)
    parser.add_argument("--m2", help="Third layer multiplier", type=float)
    parser.add_argument("--l2", default=1e-5, help="🔴Weight decay value", type=float)
    parser.add_argument("--dropout", default=0.20, help="Dropout rate for model", type=float)
    # End of important params
    parser.add_argument("--normalize", help="Normalize loss for unbalanced dataset", action="store_true")

    parser.add_argument("--split", default=580, help="Split value for validation dataset", type=float)
    parser.add_argument("--volo_weight_dir", default='/media/usb0/public/volo_weights')
    parser.add_argument("--weights_dir", default='/media/usb0/public/z.training', type=str)
    parser.add_argument("--recordfile", default='torch-training.csv', help='Training record file basename', type=str)
    parser.add_argument("--weightsfile", default='weights.pth', help='Weights file to resume from', type=str)
    if ip_address in ['192.168.1.197', '192.168.1.198']:
        mp3_file = '/usr/local/bin/mp3files/ring-msn-newalert.mp3'
    elif ip_address in ['192.168.1.187', '192.168.1.188']:
        mp3_file = '/usr/local/bin/mp3files/telegraph.mp3'
    else:
        mp3_file = '/usr/local/bin/mp3files/chime.mp3'
    parser.add_argument("--mp3file", default=mp3_file, help='mp3 files to play after each opoch', type=str)
    parser.add_argument("--my_mac_addr", default='192.168.1.192', type=str)
    parser.add_argument("-w", "--image_w", default=224, help="Default image width and height", type=int)
    parser.add_argument("-D", "--debug", help="Show debug information", action="store_true")
    return parser.parse_args()


def main(): 
    global args, working_path, ip_address
    working_path = os.path.abspath(os.path.dirname(sys.argv[0]))
    args = parse_arguments()
    #[print(f'\033[32m{k:<20} -> {v}\033[0m' if v else f'\033[31m{k:<20} -> {v}\033[0m') for k, v in vars(args).items()]
    [print(f'\033[32m{k:<20} -> {v}\033[0m') for k, v in vars(args).items() if v]
    # both dirs are basenames in working_path
    if args.init: split_into_train_val('human', 'human.new', split=args.split); exit(0)
    if args.sum is not None: training_prepare(summary_only=True, verbose=args.sum); exit(0)

    model, cfg = training_prepare(keep_weights=args.resume, resume_lr=True)
    sp_training_worker(model, cfg)


if __name__ == "__main__":
    main()


"""
https://stackoverflow.com/questions/53033556/how-should-the-learning-rate-change-as-the-batch-size-change

Theory suggests that when multiplying the batch size by k, one should multiply the learning rate by sqrt(k) to keep the
 variance in the gradient expectation constant. See page 5 at A. Krizhevsky. One weird trick for parallelizing 
 convolutional neural networks: https://arxiv.org/abs/1404.5997

However, recent experiments with large mini-batches suggest for a simpler linear scaling rule, i.e multiply your 
learning rate by k when using mini-batch size of kN. See P.Goyal et al.: Accurate, Large Minibatch SGD: Training 
ImageNet in 1 Hour https://arxiv.org/abs/1706.02677


"""
