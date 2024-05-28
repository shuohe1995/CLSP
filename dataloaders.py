import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
#import cv2
#from augment.cutout import Cutout2
#from augment.autoaugment_extra import CIFAR10Policy
import os
import copy
from randaug import *
import numpy as np
from PIL import Image
import torch
import pickle
from sklearn.preprocessing import OneHotEncoder
from model import ResNet18, LeNet
import codecs
from resnet import resnet
import torch.nn.functional as F
import pandas as pd


def unpickle1(file):
    import pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

def unpickle2(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def read_mnist_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8).tolist()
        return parsed

def read_mnist_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(length, num_rows, num_cols)
        return parsed


class PartialDataset(Dataset):
    def __init__(self, root, dataset, mode, transform, partial_rate=0,  imb_rate=0.01, imb_type='exp', reverse=False, shuffle=False):
        self.transform = transform
        self.mode = mode

        if self.mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle1('%s/test_batch' % root)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset == 'cifar100':
                test_dic = unpickle1('%s/test' % root)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
            elif dataset == 'voc':
                with open(os.path.join(root, 'test.pkl'), 'rb') as f:
                    test = pickle.load(f)
                    self.test_data = test['test_image']
                    self.test_label = test['test_label']
            elif dataset == 'tiny-imagenet':
                self.test_data = np.load(os.path.join(root, 'test_data.npy'))
                self.test_label = np.load(os.path.join(root, 'test_targets.npy'))

        else: ####train
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/data_batch_%d' % (root, n)
                    data_dic = unpickle1(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                train_data = np.concatenate(train_data)
                train_data = train_data.reshape((50000, 3, 32, 32))
                train_data = train_data.transpose((0, 2, 3, 1))
            elif dataset == 'cifar100':
                train_dic = unpickle1('%s/train' % root)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
                train_data = train_data.reshape((50000, 3, 32, 32))
                train_data = train_data.transpose((0, 2, 3, 1))
            elif dataset == 'voc':
                with open(os.path.join(root, 'train.pkl'), 'rb') as f:
                    train = pickle.load(f)
                    train_data = train['train_image']
                    train_label = train['train_true_label']
                    partial_label_matrix = train['train_partial_label']
            elif dataset == 'tiny-imagenet':
                train_data = np.load(os.path.join(root, 'train_data.npy'))
                train_label = np.load(os.path.join(root, 'train_targets.npy'))

            if imb_rate != 0:
                cls_num = 100 if dataset == 'cifar100' else 10
                img_num_list = self.get_img_num_per_cls(cls_num, imb_type, imb_rate, reverse, shuffle=shuffle)
                train_data, train_label = self.gen_imbalanced_data(img_num_list, train_data, train_label)

            self.true_label = train_label
            self.train_data = train_data

            if dataset != 'voc':
                if partial_rate == -1.0: # Instance-dependent
                    partial_label_matrix = generate_instance_dependent_candidate_labels(dataset, self.train_data, torch.tensor(self.true_label))
                else:
                    if dataset == 'cifar100' and partial_rate == 0.5: # cifar100 Hierarchical-0.5
                        partial_label_matrix = generate_hierarchical_cv_candidate_labels(dataset, torch.tensor(self.true_label), partial_rate)
                    elif dataset == 'cifar10' and partial_rate == 0.0: # cifar10 Label-dependent
                        partial_label_matrix = generate_label_dependent_cv_candidate_labels(dataset,torch.tensor(self.true_label))
                    else: # Uniform
                        partial_label_matrix = generate_uniform_cv_candidate_labels(torch.tensor(self.true_label), partial_rate)
            self.partial_label_matrix = partial_label_matrix

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_data)
        else:
            return len(self.train_data)

    def __getitem__(self, index):
        if self.mode == 'test':
            each_label = self.test_label[index]
            each_image = self.test_data[index]
            each_image = Image.fromarray(each_image)
            each_image = self.transform(each_image)
            return each_image, each_label
        elif self.mode == 'train':
            each_partial_label = self.partial_label_matrix[index]
            each_true_label = self.true_label[index]
            each_image = self.train_data[index]
            each_image = Image.fromarray(each_image)
            each_image = self.transform(each_image)
            return each_image, each_partial_label, each_true_label, index
    ######
    ######
    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, reverse, shuffle=False):
        #img_max = len(self.data) / cls_num
        img_max = 50000 / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                if reverse:
                    num = img_max * (imb_factor ** ((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
                else:
                    num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        if shuffle:
            random.shuffle(img_num_per_cls)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls, origianl_data, original_label):
        new_data = []
        new_targets = []
        targets_np = np.array(original_label, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(origianl_data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        #self.data = new_data
        #self.targets = new_targets
        return new_data, new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class PartialDataloader():
    def __init__(self, root, dataset, partial_rate, imb_rate, batch_size, num_workers, transform):
        dataset_dir_map = {'cifar10': root + 'cifar-10-batches-py', 'cifar100': root + 'cifar-100-python',
                           'tiny-imagenet': root + 'tiny-imagenet-200/npy', 'voc': root+'voc'}
        self.dataset_dir = dataset_dir_map[dataset]
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.partial_rate = partial_rate
        self.imb_rate = imb_rate
        self.root = root
        if self.dataset == 'cifar10':
            self.transform_train = transform
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ])
        elif self.dataset == 'voc':
            self.transform_train = transform
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2554, 0.2243, 0.2070), (0.2414, 0.2207, 0.2104)),
            ])
        elif self.dataset == 'cifar100':
            self.transform_train = transform
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
        elif self.dataset == 'tiny-imagenet':
            self.transform_train = transform
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])



    def run(self, mode):
        if mode == 'train':
            train_dataset = PartialDataset(dataset=self.dataset, root=self.dataset_dir, partial_rate=self.partial_rate,
                                            imb_rate=self.imb_rate,
                                           transform=self.transform_train, mode="train")
            train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      shuffle=True,
                                      pin_memory=True, drop_last=False)
            return train_loader, train_dataset.partial_label_matrix, torch.tensor(train_dataset.true_label)

        elif mode == 'test':
            test_dataset = PartialDataset(dataset=self.dataset, root=self.dataset_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

            return test_loader

def extract_cub(root):
    """Prepare the data for train/test split and save onto disk."""
    image_path = os.path.join(root, 'images/')
    # Format of images.txt: <image_id> <image_name>
    id2name = np.genfromtxt(os.path.join(root, 'images.txt'), dtype=str)
    # Format of train_test_split.txt: <image_id> <is_training_image>
    id2train = np.genfromtxt(os.path.join(root, 'train_test_split.txt'), dtype=int)

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for id_ in range(id2name.shape[0]):
        image = Image.open(os.path.join(image_path, id2name[id_, 1]))
        label = int(id2name[id_, 1][:3]) - 1  # Label starts with 0

        # Convert gray scale image to RGB image.
        if image.getbands()[0] == 'L':
            image = image.convert('RGB')
        image_np = np.array(image)
        image.close()

        if id2train[id_, 1] == 1:
            train_data.append(image_np)
            train_labels.append(label)
        else:
            test_data.append(image_np)
            test_labels.append(label)

    pickle.dump((train_data, train_labels), open(os.path.join(root, 'processed/train.pkl'), 'wb'))
    pickle.dump((test_data, test_labels), open(os.path.join(root, 'processed/test.pkl'), 'wb'))
    print("saved the processed cub file!")



def generate_hierarchical_cv_candidate_labels(dataname, train_labels, partial_rate=0.1):
    assert dataname == 'cifar100'

    meta = unpickle2('../../datasets/cifar-100-python/meta')

    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]
    label2idx = {fine_label_names[i]: i for i in range(100)}

    x = '''aquatic mammals#beaver, dolphin, otter, seal, whale
fish#aquarium fish, flatfish, ray, shark, trout
flowers#orchid, poppy, rose, sunflower, tulip
food containers#bottle, bowl, can, cup, plate
fruit and vegetables#apple, mushroom, orange, pear, sweet pepper
household electrical devices#clock, keyboard, lamp, telephone, television
household furniture#bed, chair, couch, table, wardrobe
insects#bee, beetle, butterfly, caterpillar, cockroach
large carnivores#bear, leopard, lion, tiger, wolf
large man-made outdoor things#bridge, castle, house, road, skyscraper
large natural outdoor scenes#cloud, forest, mountain, plain, sea
large omnivores and herbivores#camel, cattle, chimpanzee, elephant, kangaroo
medium-sized mammals#fox, porcupine, possum, raccoon, skunk
non-insect invertebrates#crab, lobster, snail, spider, worm
people#baby, boy, girl, man, woman
reptiles#crocodile, dinosaur, lizard, snake, turtle
small mammals#hamster, mouse, rabbit, shrew, squirrel
trees#maple_tree, oak_tree, palm_tree, pine_tree, willow_tree
vehicles 1#bicycle, bus, motorcycle, pickup truck, train
vehicles 2#lawn_mower, rocket, streetcar, tank, tractor'''

    x_split = x.split('\n')
    hierarchical = {}
    reverse_hierarchical = {}
    hierarchical_idx = [None] * 20
    # superclass to find other sub classes
    reverse_hierarchical_idx = [None] * 100
    # class to superclass
    super_classes = []
    labels_by_h = []
    for i in range(len(x_split)):
        s_split = x_split[i].split('#')
        super_classes.append(s_split[0])
        hierarchical[s_split[0]] = s_split[1].split(', ')
        for lb in s_split[1].split(', '):
            reverse_hierarchical[lb.replace(' ', '_')] = s_split[0]

        labels_by_h += s_split[1].split(', ')
        hierarchical_idx[i] = [label2idx[lb.replace(' ', '_')] for lb in s_split[1].split(', ')]
        for idx in hierarchical_idx[i]:
            reverse_hierarchical_idx[idx] = i

    # end generate hierarchical
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    p_1 = partial_rate
    transition_matrix = np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0], dtype=bool))] = p_1
    mask = np.zeros_like(transition_matrix)
    for i in range(len(transition_matrix)):
        superclass = reverse_hierarchical_idx[i]
        subclasses = hierarchical_idx[superclass]
        mask[i, subclasses] = 1

    transition_matrix *= mask
    #print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K):  # for each class
            if jj == train_labels[j]:  # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0
    print("cifar100-H uniform:  Average Candidate Labels:{:.4f}\n".format(partialY.sum() / n))
    return partialY

def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1, noisy_rate=0):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    # partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix = np.eye(K) * (1 - noisy_rate)
    # inject label noise if noisy_rate > 0
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_rate
    #print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        random_n_j = random_n[j]
        while partialY[j].sum() == 0:
            random_n_j = np.random.uniform(0, 1, size=(1, K))
            partialY[j] = torch.from_numpy((random_n_j <= transition_matrix[train_labels[j]]) * 1)

    if noisy_rate == 0:
        partialY[torch.arange(n), train_labels] = 1.0
        # if supervised, reset the true label to be one.
        #print('Reset true labels')

    num_avg = partialY.sum() / n
    print("Uniform generation: the average number of candidate Labels:{:.4f}.".format(num_avg))
    return partialY

def generate_label_dependent_cv_candidate_labels(dataname, train_labels):
    assert dataname == 'cifar10'
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix = get_transition_matrix(K)
    #print('==> Transition Matrix:')
    #print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K): # for each class
            if jj == train_labels[j]: # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0

    num_avg = partialY.sum() / n
    print("cifar10-LD Label-dependent generation: the average number of candidate Labels:{:.4f}.".format(num_avg))
    return partialY

def get_transition_matrix(K):
    transition_matrix = np.zeros((K, K))
    for i in range(K):
        transition_matrix[i, i] = 1
        # set diagonal labels as 1
        transition_matrix[i, (i+1)%K] = 0.5
        transition_matrix[i, (i+2)%K] = 0.4
        transition_matrix[i, (i+3)%K] = 0.3
        transition_matrix[i, (i+4)%K] = 0.2
        transition_matrix[i, (i+5)%K] = 0.1
    return transition_matrix

def generate_instance_dependent_candidate_labels(dataset, data, true_labels):
    num_data = true_labels.size(0)
    num_class = true_labels.max().item() + 1
    true_label_matrix = F.one_hot(true_labels, num_class)
    batch_size = 2000
    rate = 0.04 if dataset in ['cifar100','cifar100-H', 'tiny-imagenet'] else 0.4

    if dataset in ['kmnist', 'fmnist']:
        model = LeNet(out_dim=num_class, in_channel=1, img_sz=28)
        weight_path = './model_path/' + dataset + '_clean_DA1.pth'
    elif dataset in ['cifar10']:
        model = resnet(depth=32, n_outputs=num_class)
        weight_path = './model_path/cifar10_original.pt'
    elif dataset in ['cifar100', 'cifar100-H']:
        model = ResNet18(num_classes=100)
        weight_path = './model_path/cifar100_original.pth'
        #weight_path = './model_path/cifar100_pr=0_model=resnet18_clean_DA1.pth'
    elif dataset in ['tiny-imagenet']:
        model = ResNet18(num_classes=200)
        #weight_path = './model_path/tiny-imagenet_original.pth'
        weight_path = './model_path/tiny-imagenet_clean_DA1.pth'


    model.load_state_dict(torch.load(weight_path, map_location='cuda'))
    model = model.cuda()
    data = torch.from_numpy(np.copy(data))
    train_data, true_label_matrix = data.cuda(), true_label_matrix.cuda()

    if dataset in ['kmnist', 'fmnist']:
        train_data = train_data.unsqueeze(dim=1).to(torch.float32)
    elif dataset in ['cifar10', 'cifar100', 'cifar100-H', 'tiny-imagenet']:
        train_data = train_data.permute(0, 3, 1, 2).to(torch.float32)

    step = num_data // batch_size
    partial_label_matrix = torch.zeros(num_data, num_class)
    with torch.no_grad():
        for i in range(0, step):
            # if dataset in ['kmnist', 'fmnist', 'cifar100', 'cifar100-H', 'tiny-imagenet']:
            #     outputs = model(train_data[i * batch_size:(i + 1) * batch_size])
            # elif dataset in ['cifar10']:
            #     _, outputs = model(train_data[i * batch_size:(i + 1) * batch_size])
            _, outputs = model(train_data[i * batch_size:(i + 1) * batch_size])

            partial_rate_array = torch.softmax(outputs, dim=1).clone().detach()
            partial_rate_array[torch.where(true_label_matrix[i * batch_size:(i + 1) * batch_size] == 1)] = 0
            partial_rate_array = partial_rate_array / torch.max(partial_rate_array, dim=1, keepdim=True)[0]
            partial_rate_array = partial_rate_array / partial_rate_array.mean(dim=1, keepdim=True) * rate
            partial_rate_array[partial_rate_array > 1.0] = 1.0
            partial_rate_array[torch.arange(batch_size), true_labels[i * batch_size:(i + 1) * batch_size]] = 1
            partial_labels = torch.distributions.binomial.Binomial(total_count=1, probs=partial_rate_array).sample()
            ######
            while torch.nonzero(partial_labels.sum(dim=1) < 2).shape[0] != 0:
                temp_index = torch.nonzero(partial_labels.sum(dim=1) < 2).squeeze(dim=1)
                partial_labels[temp_index, torch.randint(0, num_class, (temp_index.shape[0],))] = 1
            ######
            partial_label_matrix[i * batch_size:(i + 1) * batch_size] = partial_labels.detach().clone().cpu()
    assert partial_label_matrix.sum(dim=1).min() > 1
    num_avg = partial_label_matrix.sum() / num_data
    print("Instance_dependent generation: the average number of candidate Labels:{:.4f}.".format(num_avg))
    return partial_label_matrix

def process_voc_image(image):
    height, width = image.size
    if height > width:
        padding = (height - width) // 2
        image = transforms.Pad([0, padding, 0, height - width - padding])(image)

    elif height < width:
        padding = (width - height) // 2
        image = transforms.Pad([padding, 0, width - height - padding, 0])(image)

    new_image = transforms.Resize(128)(image)
    return new_image

def process_voc():
    All_labels = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7,
                  'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
                  'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}
    id2label = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car', 7: 'cat',
                8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
                15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}

    train_images = []
    train_true_labels = []
    train_partial_label_matrix = torch.zeros(11706, 20)
    test_images = []
    test_labels = []

    train_image_path = '../../datasets/voc/train_var/images'
    data = pd.read_csv('../../datasets/voc/train_var/images_PLL.csv')
    train_image_name = data['name'].to_list()
    train_true_label_name = data['label'].to_list()
    train_partial_label_name = data['label set'].to_list()

    ##### train
    for i in range(len(train_image_name)):
        image = Image.open(os.path.join(train_image_path, train_image_name[i]))
        new_image = process_voc_image(image)
        train_images.append(np.array(new_image))
        true_label = All_labels[train_true_label_name[i]]
        train_true_labels.append(true_label)

        partial_label_set = train_partial_label_name[i].split(" ")
        partial_label = [All_labels[label] for label in partial_label_set]
        train_partial_label_matrix[i, partial_label] = 1

    ######test
    test_image_path = '../../datasets/voc/test/images'
    data = pd.read_csv('../../datasets/voc/test/images_uniform.csv')
    test_image_name = data['name'].to_list()
    test_true_label_name = data['label'].to_list()
    for i in range(len(test_image_name)):
        image = Image.open(os.path.join(test_image_path, test_image_name[i]))
        new_image = process_voc_image(image)
        test_images.append(np.array(new_image))
        true_label = All_labels[train_true_label_name[i]]
        test_labels.append(true_label)

    save_train = {}
    save_train['train_image'] = np.array(train_images)
    save_train['train_true_label'] = np.array(train_true_labels)
    save_train['train_partial_label'] = train_partial_label_matrix
    save_test = {}
    save_test['test_image'] = np.array(test_images)
    save_test['test_label'] = np.array(test_labels)

    with open('../../datasets/voc/train.pkl', 'wb') as f:
        pickle.dump(save_train, f)

    with open('../../datasets/voc/test.pkl', 'wb') as f:
        pickle.dump(save_test, f)


if __name__ == "__main__":
    process_voc()
    transform = transforms.Compose(
            [
            transforms.RandomResizedCrop(size=(128,128), scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
               transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.2554, 0.2243, 0.2070), (0.2414, 0.2207, 0.2104))
            ])

    loader = PartialDataloader(root='../../datasets/voc', dataset='voc', partial_type='uniform',
                               partial_rate=0, imb_rate=0,
                               is_LT=0,
                               batch_size=128, num_workers=4, transform=transform)

    data_loader, partial_label_matrix, true_labels = loader.run('train')

    for i, (x, py, y, index) in enumerate(data_loader):

        debug=0









