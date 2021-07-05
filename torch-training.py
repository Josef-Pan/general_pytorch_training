# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time, os, copy, tqdm, time, sys, glob, re, shutil, socket
from collections import OrderedDict
from datetime import datetime
from torch.utils.data.sampler import SubsetRandomSampler


# ✅ 1 Data preparation
def split_into_train_val(path_name, path_name_splitted, split = 0.2):
    """
    :param path_name: folder containing training pictures, each category in one folder
    :param path_name_splitted: folder containing splitted pictures, will contain 'train' and 'val' subfolder
    :param split: percentage of files for validation dataset
    :return:
    """
    working_path = os.path.dirname(os.path.realpath(__file__))
    train_dir = working_path + '/' + path_name_splitted + '/train'
    val_dir   = working_path + '/' + path_name_splitted + '/val'
    shutil.rmtree(train_dir, ignore_errors= True)
    shutil.rmtree(val_dir,   ignore_errors= True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    subfolders = sorted(glob.glob(path_name + '/*'))
    subfolders_short = [ os.path.basename(item) for item in subfolders]
    print(subfolders_short)
    for category in subfolders_short:
        os.makedirs(train_dir + '/' + category , exist_ok=True)
        os.makedirs(val_dir + '/' + category , exist_ok=True)

    for folder in subfolders:
        imageSet = glob.glob(folder + "/*")
        numbers_val = int(len(imageSet)*split)
        indices_all = np.arange(len(imageSet))
        indices_val = np.random.choice(len(imageSet), replace=False, size=(numbers_val,))
        indices_train = indices_all[np.isin(indices_all, indices_val, invert=True)]
        for idx, img_path in enumerate(imageSet):
            if idx in indices_train:
                target_path = train_dir + '/' + os.path.basename(folder) + '/' + os.path.basename(img_path)
                print(f'{idx} = TRAIN, {img_path}\t\t{target_path}')
                shutil.copy(img_path, target_path)
            if idx in indices_val:
                target_path = val_dir + '/' + os.path.basename(folder) + '/' + os.path.basename(img_path)
                print(f'{idx} = VAL  , {img_path}\t\t{target_path}')
                shutil.copy(img_path, target_path)

def data_preparation(data_dir, batch_sizes ={'train':32, 'val':16}):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes[x],
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ✅ 2 Model preparation

def model_preparation_densenet169(nb_classes):
    device = get_device()
    model = torchvision.models.densenet169(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    params = [(name, param) for name, param in model.named_parameters()]
    last_layer = params[-2][0] # shoud be 'classifier.weight'
    last_layer_output_shape = model.state_dict()[last_layer].shape

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(last_layer_output_shape[1], 512)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(512, nb_classes)),
        ('Output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    model = model.to(device)
    return model, "densenet169"

def model_preparation_densenet201(nb_classes):
    device = get_device()
    model = torchvision.models.densenet201(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    params = [(name, param) for name, param in model.named_parameters()]
    last_layer = params[-2][0] # shoud be 'classifier.weight'
    last_layer_output_shape = model.state_dict()[last_layer].shape

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(last_layer_output_shape[1], 512)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(512, nb_classes)),
        ('Output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    model = model.to(device)
    return model, "densenet201"

def unfreeze_from_layer(model, layer_name =''):
    unfreeze_from = 9999
    params = [(name, param) for name, param in model.named_parameters()]
    for param in model.parameters():
        param.requires_grad = False
    for idx, (name, param) in enumerate(params):
        if layer_name in name: unfreeze_from = idx; break
    print(f"Unfreezing from layer = {idx} {name}")
    for idx, (name, param) in enumerate(params):
        if idx >= unfreeze_from:
            param.requires_grad = True

class MyOptimizer:
    def __init__(self, lr, step, gamma, momentum, nesterov=False):
        self.name = "josef_Optimzer"
        self.lr = lr
        self.step = step
        self.gamma = gamma
        self.momentum = momentum
        self.nesterov = nesterov

# ✅ 3 Training preparation and training

def mainIPV4Address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    addr = s.getsockname()[0]
    s.close()
    return str(addr)

def changeLRbyFile():
    """
    create a file name lr.txt and type new lr, new lr will be loaded next epoch
    lr.txt will be deleted after using
    :return: new lr to be used in next epoch, or None if lr.txt does not contain any legal lr
    """
    file_name = "lr.txt"
    message = "[\033[1;35m%s\033[0m was deleted afterwards\033[0m]" % file_name
    try:
        reader = open(file_name, 'r')
        first_line = reader.readline().replace('\n', ' ')   # Only first line, other lines just ignored
        new_lr = safe_cast_float(first_line)
        if new_lr:
            fmt_string = "\033[32mlr.txt contents = %s\033[1;32m Value = %.2e\033[0m; %s"
            print( fmt_string % (first_line, new_lr, message) , end =' ')
            os.remove(file_name); return new_lr
        else:
            fmt_string = "\033[32mlr.txt contents = %s\033[1;31m Value illegal\033[0m; %s"
            print( fmt_string % (first_line, message), end =' ')
            os.remove(file_name); return None
    except: return None

def save_weight_as_symlink(latest_weight):
    try:
        os.remove('weights.pth')
        os.symlink(latest_weight, 'weights.pth')  # Make a symlink instead of copying
    except: pass

def acc_2_str(array):
    return '[' + ','.join(['%5.2f%%' % (array[n] * 100.0) for n in range(len(array))]) + ']'

def files_2_str(files_kept):
    files_kept = files_kept[-4:]
    return '[' + ','.join(['%s' % files_kept[n] for n in range(len(files_kept))]) + ']'

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(model, dataloaders, class_names, num_images=6):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def log_to_file(string = ""):
    file = open('torch_training.csv','a')
    file.write(string + '\n')
    file.close()

def log_file_header(epochs, name, optimizer, my_optim, unfreeze_from):
    lr, momentum, nesterov = get_optimzer_params(optimizer)
    config_data = (epochs, name, type(optimizer).__name__, lr, momentum, int(nesterov), my_optim.gamma, my_optim.step)
    str_training_cfg = "epochs=%d @%s %s#%.2e,momentum=%.1f,nesterov=%d,%.2f/%d" % config_data
    dt_string = datetime.now().strftime("\033[1;36m%Y-%m-%d %H:%M:%S\033[0m")
    str_to_log = dt_string + " \033[1;32m" + str_training_cfg + '  ' + unfreeze_from + "\033[0m"
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
    line_last = lines[-1]
    splitted = line_last.split(' ')
    if len(splitted) <= 1: return 0
    try:
        return safe_cast_int(splitted[3].split('/')[0])
    except:
        return 0

def get_last_lr(lines):
    line_last = lines[-1]
    splitted = line_last.split('=')             # 'lr = ' is the last entry of csv record line
    if len(splitted) <= 1: return None
    try:
        return safe_cast_float(splitted[-1])    # 'lr = ' is the last entry of csv record line
    except:
        return None

def get_old_csv_lines():
    try:
        reader = open('torch_training.csv', 'r')
        lines = [line.replace('\n', '') for line in reader.readlines() if len(line) > 0]
    except:
        return []
    return lines[1:]  # first line is header, only other lines contain training history information

def get_weights_file():
    """
    :return: searching for file named weights.pth, will return None if not exists
    """
    working_path = os.path.dirname(os.path.realpath(__file__))
    weights_file = os.path.join(working_path, 'weights.pth')
    return (None, weights_file)[os.path.exists(weights_file)]

def keep_only_best_models(files_keep = 4):
    """
    :param files_keep: how many files with best val_acc will be kept
    :return: a list of string with file names shortened
    """
    def del_low_value_files(filename_wild_card):
        all_h5_files = glob.glob(filename_wild_card)
        all_val_acc = sorted([safe_cast_float(file[-9:-4]) for file in all_h5_files], reverse= True) #best first
        best_acc_values = all_val_acc[0:files_keep]  #N best val_acc values
        for file in all_h5_files:
            acc_indicator = safe_cast_float(file[-9:-4])
            if acc_indicator not in best_acc_values:
                try   : os.remove(file)
                except: pass

    del_low_value_files("torch-weights*.pth")
    all_h5_files = [ file[-13:] for file in glob.glob("torch-weights*.pth") ]
    all_h5_files.sort(key=lambda filename: safe_cast_float(filename[-9:-4]))
    return all_h5_files

def get_optimzer_params(optimizer):
    try:
        params = optimizer.state_dict()['param_groups'][0]
        lr = params.get( 'lr', 0.0)
        momentum = params.get( 'momentum', 0.0)
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

def summary(model, verbose = 0):
    """
    :param model: model to summarize
    :param verbose:
    0, show some trainable layers; 1, show all trainable layers
    2, show all trainable layers and all layers; 3, show all trainable layers and print model
    :return: None
    """
    trainable, untrainable  = count_parameters(model)
    print(f"Trainable params = \033[32m{trainable:,}\033[0m Untrainable parmas = \033[31m{untrainable:,}\033[0m")
    trainable_layer_names =[]
    for name, param in model.named_parameters():
        if ('weight' in name and param.requires_grad):
            layer_name = name.replace('weight', '').replace('features', '').strip('.')
            trainable_layer_names.append(layer_name)
    def show_trainale_10():
        print("Trainable layers = ", end=' ')
        for layer in trainable_layer_names[:5]:
            print(f"{layer}", end=', ')
        print(f"\033[32m ...... \033[0m", end=' ')
        for layer in trainable_layer_names[-5:]:
            print(f"{layer}", end=', ')
        print(f"\033[32mtotal={len(trainable_layer_names)}\033[0m")

    def show_trainale_all():
        print("Trainable layers = ", end=' ')
        for layer in trainable_layer_names:
            print(f"{layer}", end=', ')
        print(f"\033[32mtotal={len(trainable_layer_names)}\033[0m")

    if verbose == 0:
        show_trainale_10() if len(trainable_layer_names) > 10 else show_trainale_all()

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
def policy186():
    data_dir = 'flowers.new'; epochs = 100
    optimizer = MyOptimizer(lr=4e-5, step=2, gamma=0.95, momentum=0.9, nesterov=1)
    batch_sizes ={'train':32, 'val':16}; unfreeze_from = 'denseblock3'
    return data_dir, epochs, optimizer, batch_sizes, unfreeze_from

def policy_test():
    # This dataset is available at https://download.pytorch.org/tutorial/hymenoptera_data.zip
    data_dir = 'data_hymenoptera'; epochs = 20
    optimizer = MyOptimizer(lr=1e-4, step=2, gamma=0.90, momentum=0.9, nesterov=1)
    batch_sizes ={'train':32, 'val':8}; unfreeze_from = 'denseblock4'
    return data_dir, epochs, optimizer, batch_sizes, unfreeze_from

def policyUnknownPC():
    print("\033[1;31mOrdinateur inconnu, au revoir!\033[0m"); exit(0)
######################################################################################################### ❤️❤️
def train_model(keep_weights = False, read_last_lr = True, summary_only=False, verbose=0):
    last_epoch = 0;  # will resume from last record in csv file if keep_weights==True
    training_policies = {
        "192.168.1.187": policy_test,  # Testing purpose only
        "192.168.1.186": policy186,
    }
    ip_address = mainIPV4Address(); print(ip_address)
    policy_func = training_policies.get(ip_address, policyUnknownPC) #Will exit for ip address not defined
    data_dir, epochs, my_optim, batch_sizes, unfreeze_from = policy_func()
    print(data_dir, batch_sizes)
    dataloaders, dataset_sizes, class_names = data_preparation(data_dir, batch_sizes)
    device = get_device()

    model, model_name = model_preparation_densenet201(len(class_names))
    #model, model_name = model_preparation_densenet169(len(class_names))
    print(class_names)
    unfreeze_from_layer(model, layer_name=unfreeze_from);
    summary(model, verbose=verbose)
    (lambda x: x, exit)[summary_only](0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=my_optim.lr, momentum=my_optim.momentum, nesterov=my_optim.nesterov)

    # Decay LR by a factor of 0.9 every 2 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=my_optim.step, gamma=my_optim.gamma)

    if keep_weights:
        existing_log_lines = get_old_csv_lines()
        weight_file = get_weights_file()
        if weight_file:
            try:
                model.load_state_dict(torch.load(weight_file))
            except:
                print("\033[1;31mweights.pth does not match model, exiting..."); exit(1)
            last_epoch = get_last_epoch(existing_log_lines)
            last_lr = get_last_lr(existing_log_lines)
            fmt_string = "\033[1;35mLoading %s into model, continuing from \033[1;32mepoch %d\033[0m"
            print(fmt_string % (os.path.basename(weight_file), last_epoch + 1))
            if read_last_lr and last_lr:  # 读取正确，否则是None
                optimizer = optim.SGD(model.parameters(), lr=last_lr, momentum=my_optim.momentum,
                                      nesterov=my_optim.nesterov)
                scheduler = lr_scheduler.StepLR(optimizer, step_size=my_optim.step, gamma=my_optim.gamma)
                print(f"\033[1;35mLearning rate resumed as \033[1;32m{last_lr:.2e}\033[0m")
        else:
            print("\033[1;31mLoading weights.pth into model failed, training from start\033[0m")

    [os.remove(f) for f in glob.glob("torch*.csv")]
    if not (keep_weights and last_epoch > 0): [os.remove(f) for f in glob.glob("torch*.pth")]

    str_logged = log_file_header(epochs, model_name, optimizer, my_optim, unfreeze_from)
    print(str_logged)

    if keep_weights and last_epoch>0 :
        for line in existing_log_lines:
            log_to_file(line)

    since = time.time()
    for epoch in range(last_epoch+1, last_epoch+epochs+1):  # loop over the dataset multiple times
        str_date = datetime.now().strftime("%m-%d %H:%M")
        lr_from_lr_txt = changeLRbyFile()  # read new lr from lr.txt if it exists
        if lr_from_lr_txt:
            new_lr = lr_from_lr_txt;
            optimizer = optim.SGD(model.parameters(), lr=new_lr, momentum=my_optim.momentum)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=my_optim.step, gamma=my_optim.gamma)
            lr, _, _ = get_optimzer_params(optimizer)
            print("\033[1;35mLR has been changed to %.2e\033[0m" % lr)

        # Training ###########################################################
        model.train()  # Set model to training mode
        batch_size = dataloaders['train'].batch_size
        trange_obj = tqdm.tqdm(dataloaders['train'])
        running_loss = 0.0; running_corrects = 0
        lr, _, _ = get_optimzer_params(optimizer)
        for inputs, labels in trange_obj:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                #print(outputs); print(outputs.shape); exit(0)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            # statistics train
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            loss_on_bar = running_loss/batch_size/(trange_obj.n+1)
            bar = "\033[1;36m%s epoch%3d loss=%.4f lr=%.2e\033[0m"%(str_date, epoch, loss_on_bar, lr)
            trange_obj.set_description(bar)
            trange_obj.refresh()  # to show immediately the update
            time.sleep(0.001)
        trange_obj.close()
        scheduler.step()
        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects * 1.0 / dataset_sizes['train']
        print('\033[36m%s Loss: %.4f Acc: %05.2f%% lr=%.2e\033[0m' % ('train', epoch_loss, epoch_acc * 100.0, lr))

        # Validating ###########################################################
        model.eval()  # Set model to evaluate mode
        batch_size = dataloaders['val'].batch_size
        trange_obj = tqdm.tqdm(dataloaders['val'])
        running_loss = 0.0; running_corrects = 0
        correct_pred = {classname: 0 for classname in class_names}
        total_pred = {classname: 0 for classname in class_names}
        accuracy_all = []
        for inputs, labels in trange_obj:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                for label, prediction in zip(labels, preds):
                    if label == prediction:
                        correct_pred[class_names[label]] += 1
                    total_pred[class_names[label]] += 1
            # statistics val
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            bar = "\033[1;32m%s epoch%3d loss=%.4f\033[0m"%(str_date, epoch, running_loss/batch_size/(trange_obj.n + 1))
            trange_obj.set_description(bar)
            trange_obj.refresh()  # to show immediately the update
            time.sleep(0.001)
        trange_obj.close()
        for classname, correct_count in correct_pred.items():
            accuracy_per_calss = float(correct_count) / total_pred[classname]
            accuracy_all.append(accuracy_per_calss)
        epoch_loss = running_loss / dataset_sizes['val']
        epoch_acc = running_corrects * 1.0 / dataset_sizes['val']

        # Logging #############################################################
        dt_string = datetime.now().strftime("\033[1;36m%Y-%m-%d %H:%M:%S\033[0m")
        epoch_string = " Epoch %03d/%03d " % (epoch, last_epoch + epochs)
        val_string = '%s Loss: %.4f Acc: %05.2f%%' % ('val  ', epoch_loss, epoch_acc*100.0)
        lr_string = ' lr = %.2e ' % lr
        val_log = val_string.lstrip('val').lstrip(' ') + ' '
        log_to_file(dt_string + epoch_string + val_log + acc_2_str(accuracy_all) + lr_string)

        # Weights #############################################################
        latest_weight = "torch-weights-%03d-%05.2f.pth" % (epoch, epoch_acc*100.0)
        torch.save(model.state_dict(), latest_weight)
        #shutil.copyfile(latest_weight, 'weights.pth')  # Make a copy of latest weights to resume from
        save_weight_as_symlink(latest_weight)  # ln -s,   make a link of latest weights to resume from
        pth_files_kept = keep_only_best_models(files_keep=4)
        print(f"\033[32m{val_string} {acc_2_str(accuracy_all)} {files_2_str(pth_files_kept)}")


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model
#########################################################################################################
def evaluate_model(): # Not realized yet
    pass
#########################################################################################################

# ✅ 4 Program entry point, should run with --init for first time
def main(argv):
    """
    :param argv: must run --init first to create folder containing slpitted pictures if your dataset is not splitted
    :return:
    """
    string_argvs = [str(arg) for arg in argv]
    if ('--init' in string_argvs): split_into_train_val('flowers', 'flowers.new'); exit(0)
    if ('--ev'   in string_argvs): evaluate_model(); exit(0)
    if ('--sum'  in string_argvs): train_model(summary_only=True, verbose=0); exit(0)
    if ('--sum0' in string_argvs): train_model(summary_only=True, verbose=0); exit(0)
    if ('--sum1' in string_argvs): train_model(summary_only=True, verbose=1); exit(0)
    if ('--sum2' in string_argvs): train_model(summary_only=True, verbose=2); exit(0)
    if ('--sum3' in string_argvs): train_model(summary_only=True, verbose=3); exit(0)
    if ('--resume' in string_argvs or '-r' in string_argvs):
        train_model(keep_weights=True, read_last_lr = True)  # Continue training, resume last lr
    elif ('--cont' in string_argvs or '-c' in string_argvs):
        train_model(keep_weights=True, read_last_lr = False) # Continue training, new lr
    else:
        train_model(keep_weights=False)# Restart training
if __name__ == "__main__":
    main(sys.argv[1:])

"""

"""
