import argparse
from dataloaders import PartialDataloader
from model import *
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as tv_models
from simclr_model import SimCLR
import os
os.environ['CURL_CA_BUNDLE'] = ''
import faiss
from lavis.models import load_model_and_preprocess, model_zoo

###################
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', help='train batch size ', default=128, type=int)
parser.add_argument('--dataset', help='specify a dataset', default='cifar10', choices=['cifar10', 'voc', 'cifar100', 'tiny-imagenet'], type=str)
parser.add_argument('--dataset_root', help='data', default='../../datasets/', type=str)
parser.add_argument('--seed', help='Random seed', default=7438, type=int, required=False)
parser.add_argument('--gpu', help='used gpu id', default=0, type=int, required=False)
#########
parser.add_argument('--partial_rate', help='partial rate', default=0.4, type=float)
parser.add_argument('--imb_rate', help='partial rate', default=0.0, type=float)
parser.add_argument('--save', help='save partial label matrix', default=False, action='store_true')
parser.add_argument('--model_name', default='blip2', choices=['resnet18_i','resnet18_s','resnet18_c', 'clip', 'albef', 'blip2'], type=str)
parser.add_argument('--model_type', default='pretrain', type=str)
parser.add_argument('--k', help='knn', default=10, type=int)
parser.add_argument('--tau', help='per example pruning ratio', default=0.2, type=float)
#########


#
args = parser.parse_args()
print(args)
#####
###################
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
###################
torch.cuda.set_device(args.gpu)

def get_pretrained_model(num_class):
    if args.model_name == 'resnet18_i':
        pre_model = tv_models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        pre_model.fc = nn.Sequential()
    elif args.model_name == 'resnet18_s':
        base_encoder = tv_models.resnet18(weights=None)
        pre_model = SimCLR(base_encoder, projection_dim=128).cuda()
        pre_model.load_state_dict(torch.load('./model_path/simclr_'+args.dataset+'_resnet18_epoch1000.pt', map_location='cuda:'+str(args.gpu)))
    elif args.model_name == 'resnet18_c':
        pre_model = ResNet18(num_classes=num_class)
        temp = torch.load('./model_path/'+args.dataset+'_model=resnet18_clean_DA1.pth')
        # del temp['head.0.weight']
        # del temp['head.0.bias']
        # del temp['head.2.weight']
        # del temp['head.2.bias']
        # torch.save(temp, './model_path/'+args.dataset+'_model=resnet18_clean_DA1.pth')
        pre_model.load_state_dict(torch.load('./model_path/'+args.dataset+'_model=resnet18_clean_DA1.pth', map_location='cuda:'+str(args.gpu)))
        pre_model.linear = nn.Sequential()
    #############
    if args.dataset in ['cifar10', 'cifar100']:
        crop = transforms.RandomCrop(32, padding=4)
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif args.dataset in ['tiny-imagenet']:
        crop = transforms.RandomCrop(64, padding=4)
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    elif args.dataset in ['voc']:
        crop = transforms.RandomCrop(128, padding=12)
        normalize = transforms.Normalize((0.2554, 0.2243, 0.2070), (0.2414, 0.2207, 0.2104))
    ############
    if '_i' in args.model_name:
        transform = transforms.Compose([
                                        crop,
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Resize(224),
                                        transforms.ToTensor(),
                                        normalize
                                    ])
    elif '_s' in args.model_name:
        transform = transforms.Compose([
                                        crop,
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                    ])
    elif '_c' in args.model_name:
        transform = transforms.Compose([
                                        crop,
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                        ])

    return pre_model.cuda(), transform

def get_feature(model, dataloader):
    model.eval()
    features = torch.zeros(num_data, feature_dim)
    for batch_idx, (images, partial_label, true_label, index) in enumerate(dataloader):
        images = images.cuda()
        if args.model_name == 'clip':
            feature = model.encode_image(images)
        elif args.model_name in ['blip', 'blip2', 'albef']:
            sample = {"image": images, "text_input": None}
            feature = model.extract_features(sample, mode="image").image_embeds[:, 0 ,:]
        else:
            if '_i' in args.model_name:
                feature = model(images)
            else:
                feature, _ = model(images)

        for i in range(images.size(0)):
            features[index[i]] = feature[i].detach().clone().cpu()
    return features

def eval_quality(features):
    features = F.normalize(features, dim=1).numpy()

    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    _, I5 = index.search(features, 5+1)
    n_indices_5 = torch.from_numpy(I5[:, 1:6])
    _, I20 = index.search(features, 20+1)
    n_indices_20 = torch.from_numpy(I20[:, 1:21])
    _, I50 = index.search(features, 50+1)
    n_indices_50 = torch.from_numpy(I50[:, 1:51])
    _, I100 = index.search(features, 100+1)
    n_indices_100 = torch.from_numpy(I100[:, 1:101])
    _, I150 = index.search(features, 150+1)
    n_indices_150 = torch.from_numpy(I150[:, 1:151])
    _, I200 = index.search(features, 200+1)
    n_indices_200 = torch.from_numpy(I200[:, 1:201])

    delta_5 = (true_label_matrix[n_indices_5].sum(dim=1) * true_label_matrix).sum(dim=1).float().mean() / 5
    delta_20 = (true_label_matrix[n_indices_20].sum(dim=1) * true_label_matrix).sum(dim=1).float().mean() / 20
    delta_50 = (true_label_matrix[n_indices_50].sum(dim=1) * true_label_matrix).sum(dim=1).float().mean() / 50
    delta_100 = (true_label_matrix[n_indices_100].sum(dim=1) * true_label_matrix).sum(dim=1).float().mean() / 100
    delta_150 = (true_label_matrix[n_indices_150].sum(dim=1) * true_label_matrix).sum(dim=1).float().mean() / 150
    delta_200 = (true_label_matrix[n_indices_200].sum(dim=1) * true_label_matrix).sum(dim=1).float().mean() / 200

    rho_5 = (partial_label_matrix[n_indices_5].sum(dim=1) * false_partial_label_matrix).max(dim=1)[0].float().mean() / 5
    rho_20 = (partial_label_matrix[n_indices_20].sum(dim=1) * false_partial_label_matrix).max(dim=1)[0].float().mean()  / 20
    rho_50 = (partial_label_matrix[n_indices_50].sum(dim=1) * false_partial_label_matrix).max(dim=1)[0].float().mean()  / 50
    rho_100 = (partial_label_matrix[n_indices_100].sum(dim=1) * false_partial_label_matrix).max(dim=1)[0].float().mean()  / 100
    rho_150 = (partial_label_matrix[n_indices_150].sum(dim=1) * false_partial_label_matrix).max(dim=1)[0].float().mean()  / 150
    rho_200 = (partial_label_matrix[n_indices_200].sum(dim=1) * false_partial_label_matrix).max(dim=1)[0] .float().mean() / 200

    print('1-delta_5:{:.4f} 1-delta_20:{:.4f} 1-delta_50:{:.4f} 1-delta_100:{:.4f} 1-delta_150:{:.4f} 1-delta_200:{:.4f}'.format(delta_5, delta_20, delta_50, delta_100,delta_150, delta_200))
    print('rho_5:{:.4f} rho_20:{:.4f} rho_50:{:.4f} rho_100:{:.4f} rho_150:{:.4f} rho_200:{:.4f}'.format(rho_5, rho_20, rho_50, rho_100,rho_150, rho_200))

def cal_down_votes(features):
    features = F.normalize(features, dim=1).numpy()
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    D, I = index.search(features, args.k+1)
    n_indices = torch.from_numpy(I[:, 1:args.k+1])
    down_votes = (reversed_partial_label_matrix[n_indices] * partial_label_matrix.unsqueeze(dim=1)).sum(dim=1)

    return down_votes

def clsp(down_votes):

    num_candidate = partial_label_matrix.sum(dim=1) - 1
    num_del1 = (num_candidate * args.tau).ceil().long()
    values1, indices1 = down_votes.sort(dim=1, descending=True)
    threshold_values1 = values1[torch.arange(num_data), num_del1].unsqueeze(dim=1)
    pruned_label_matrix = (down_votes > threshold_values1).float()

    return pruned_label_matrix

def eval_clsp(del_matrix):
    error = torch.nonzero((del_matrix * true_label_matrix).sum(dim=1) != 0).shape[0] / num_data
    coverage = del_matrix.sum() / (partial_label_matrix.sum() - num_data)
    F1 = F1_beta_scroe(0.5, 1-error, coverage)
    print("pruning_coverage:{:.4f}  pruning_error:{:.4f} F1:{:.4f}".format(coverage , error, F1))


def F1_beta_scroe(beta, precision, recall):
    return (1+beta**2)*precision*recall / ((beta**2)*precision+recall)

feature_dim_map = {'blip2': 768, 'clip': 512, 'blip': 768, 'albef': 768, 'resnet18_s': 512, 'resnet18_c': 512, 'resnet18_i': 512}

if args.dataset in ['cifar10', 'cifar10-LD']:
    num_class = 10
elif args.dataset in ['cifar100', 'cifar100-H']:
    num_class = 100
elif args.dataset == 'voc':
    num_class = 20
elif args.dataset == 'tiny-imagenet':
    num_class = 200

if args.model_name in ['clip', 'blip', 'blip2', 'albef']:
    model, vis_processors, txt_processors = load_model_and_preprocess(name=args.model_name+'_feature_extractor', model_type=args.model_type, is_eval=True, device='cuda:'+str(args.gpu))
    transform = vis_processors['eval']
elif args.model_name in ['resnet18_i', 'resnet18_s', 'resnet18_c']:
    model, transform = get_pretrained_model(num_class)

loader = PartialDataloader(root=args.dataset_root, dataset=args.dataset,
                           partial_rate=args.partial_rate, imb_rate=args.imb_rate,
                           batch_size=args.batch_size, num_workers=8, transform=transform)
data_loader, partial_label_matrix, true_labels = loader.run('train')
reversed_partial_label_matrix = (partial_label_matrix.float() + 1) % 2

num_data = partial_label_matrix.size(0)
feature_dim = feature_dim_map[args.model_name]
true_label_matrix = F.one_hot(true_labels, num_class)
false_partial_label_matrix = partial_label_matrix-true_label_matrix

features = get_feature(model, data_loader)
#eval_quality(features) # empirically calculate 1-delta_k and pho_k
down_votes = cal_down_votes(features)
pruned_label_matrix = clsp(down_votes)
eval_clsp(pruned_label_matrix)

if args.save:
    log_path = os.path.join('saved', args.dataset)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    original_file_path = os.path.join(log_path, 'pr='+str(args.partial_rate)+'_ir='+str(args.imb_rate)+'_tau=0.0_k=0_model=0.pt')
    if not os.path.isfile(original_file_path):
        torch.save(partial_label_matrix, original_file_path)
    new_partial_label_matrix = partial_label_matrix-pruned_label_matrix
    assert new_partial_label_matrix.sum(dim=1).min() > 0
    new_file_path = 'pr='+str(args.partial_rate)+'_ir='+str(args.imb_rate)+'_tau='+str(args.tau)+'_k='+str(args.k)+'_model='+args.model_name+'.pt'
    torch.save(new_partial_label_matrix, os.path.join(log_path, new_file_path))
    print('saved!')










