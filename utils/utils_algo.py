import numpy as np
import torch
import torch.nn.functional as F
import random
import pickle
#from imageio import imwrite
from sklearn.mixture import GaussianMixture

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
def accuracy_check(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            outputs,_ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)
    return total / num_samples

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape((-1, )).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    p_1 = partial_rate
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))]=p_1
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K): # for each class
            if jj == train_labels[j]: # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0

    print("Finish Generating Candidate Label Sets!\n")
    return partialY

def get_random_negative_label(y):
    # get a random label from non-candidate labels
    bs=y.size(0)
    negative_labels=torch.zeros(bs)
    for i in range(bs):
        all_non_candidate_labels = torch.nonzero(y[i]==0).squeeze(dim=1).cpu().numpy()
        if all_non_candidate_labels.size==0:
            a_random_non_candidate_label=np.random.randint(10,size=1).item()
        else:
            a_random_non_candidate_label = np.random.choice(all_non_candidate_labels,1).item()
        negative_labels[i]=a_random_non_candidate_label
    return negative_labels

def get_random_negative_labe2(ty):
    # get a random label from non-candidate labels
    bs=ty.size(0)
    negative_labels=torch.zeros(bs)
    for i in range(bs):
        temp=np.arrange(10)
        true_label_index=ty[i].item()-1
        all_candidate_labels = temp.delete(temp,true_label_index)
        a_random_candidate_label = np.random.choice(all_candidate_labels,1).item()
        negative_labels[i]=a_random_candidate_label
    return negative_labels

def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

def generate_hierarchical_cv_candidate_labels(dataname, train_labels, partial_rate=0.1):
    assert dataname == 'cifar100-H'

    meta = unpickle('../datasets/cifar-100-python/meta')

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
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K):  # for each class
            if jj == train_labels[j]:  # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0
    print("Finish Generating Candidate Label Sets!\n")
    return partialY

def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    p_1 = partial_rate
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))]=p_1
    #print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K): # for each class
            if jj == train_labels[j]: # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0

    #print("Finish Generating Candidate Label Sets!\n")
    return partialY

def generate_ooc(data,os_data,partialY,true_labels,cs_rate=0.1,os_rate=0.1,partial_rate=0.3):
    num1=data.shape[0]
    num2=os_data.shape[0]
    num_cs=int(num1*cs_rate)
    num_os=int(num1*os_rate)
    random_index=torch.randperm(num1)
    index_cs=random_index[0:num_cs]
    index_normal=random_index[num_cs:num1]
    index_os=torch.randperm(num2)[0:num_os]
    #####
    index_of_no_noncandidate=[]
    for i,index in enumerate(index_cs):
        non_candidate_labels=torch.nonzero(partialY[index]==0).squeeze(dim=1)
        if non_candidate_labels.shape[0]==0: ### no non-candidate label
            index_of_no_noncandidate.append(i)
            continue
        else:
            ooc_label_index=random.randint(0,non_candidate_labels.shape[0]-1)
            partialY[index][true_labels[index]]=0
            partialY[index][non_candidate_labels[ooc_label_index]]=1
    #####
    temp=np.delete(index_cs.numpy(),index_of_no_noncandidate)
    index_cs=torch.from_numpy(temp)

    new_data=np.concatenate((data,os_data[index_os]),axis=0)
    os_partialY=generate_random_candidate_labels(num_os,10,partial_rate,False)
    new_partialY=torch.cat([partialY,os_partialY])
    index_os=torch.arange(num1,num1+num_os)
    temp=generate_one_random_label(num_os,10)
    os_random_true_labels=temp.max(dim=1)[1]

    #####
    return new_data,new_partialY,index_normal,index_cs,index_os,os_random_true_labels

def return_same_idx(a,b):
    uniset,cnt=torch.cat([a,b]).unique(return_counts=True)
    result=torch.nonzero(cnt==2).squeeze(dim=1)
    return uniset[result]

def cnt_same_idx(a,b):
    uniset,cnt=torch.cat([a,b]).unique(return_counts=True)
    result=torch.nonzero(cnt==2)
    return len(result)

def entropy(p, dim = -1, keepdim = None):
   return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim = dim, keepdim = keepdim)

def cal_entropy(outputs,Y):
    #sm_outputs=torch.softmax(outputs,dim=1)
    reversedY = ((1 + Y) % 2)
    candidate_outputs=outputs*Y
    noncandidate_outputs=outputs * reversedY
    candidate_entropy=entropy(candidate_outputs,1,False)
    noncandidate_entropy = entropy(noncandidate_outputs,1,False)

    #confidence
    #candidate_confidence=candidate_outputs/candidate_outputs.sum(dim=1,keepdim=True)
    #noncandidate_confidence=noncandidate_outputs/noncandidate_outputs.sum(dim=1,keepdim=True)

    #normalize
    #num_candidate=Y.sum(dim=1)
    #num_noncandidate=reversedY.sum(dim=1)
    # candidate_entropy=entropy(candidate_outputs,1,False) / num_candidate
    # noncandidate_entropy = entropy(noncandidate_outputs,1,False) / num_noncandidate
    #candidate_entropy=entropy(candidate_confidence,1,False)/ num_candidate
    #noncandidate_entropy = entropy(noncandidate_confidence,1,False) / num_noncandidate

    ## some examples have no candidates / all candidates
    # idx_temp=torch.nonzero(reversedY.sum(dim=1)==0)
    # noncandidate_entropy[idx_temp]=2
    # candidate_entropy[idx_temp]=0

    return candidate_entropy,noncandidate_entropy

# def cal_wood_loss(outputs,Y):
#     reversedY = ((1 + Y) % 2)
#     logsm_outputs = F.log_softmax(outputs, dim=1)
#     candidate_loss= (-logsm_outputs * Y).sum(dim=1)
#     noncandidate_loss = (-logsm_outputs * reversedY).sum(dim=1)
#
#     return candidate_loss, noncandidate_loss

def cal_wood_loss(outputs,Y):
    max_value=1000
    reversedY = ((1 + Y) % 2)
    logsm_outputs = F.log_softmax(outputs, dim=1)
    candidate_loss= -logsm_outputs * Y
    candidate_loss[candidate_loss==0]=max_value
    min_candidate_loss=candidate_loss.min(dim=1)[0]
    noncandidate_loss = -logsm_outputs * reversedY
    noncandidate_loss[noncandidate_loss==0]=max_value
    min_noncandidate_loss=noncandidate_loss.min(dim=1)[0]

    return min_candidate_loss, min_noncandidate_loss

def cal_gmm(candidate,noncandidate):
    #fit a two-component GMM to the entropy of candidates
    gmm_candiadte = GaussianMixture(n_components=2,max_iter=20,tol=1e-2,reg_covar=5e-4) # convergence warning
    gmm_candiadte.fit(candidate)
    prob_candidate = gmm_candiadte.predict_proba(candidate)
    prob_candidate = prob_candidate[:,gmm_candiadte.means_.argmin()]
    # fit a two-component GMM to the entropy of non-candidates
    gmm_noncandiadte = GaussianMixture(n_components=2,max_iter=20,tol=1e-2,reg_covar=5e-4)
    gmm_noncandiadte.fit(noncandidate)
    prob_noncandidate = gmm_noncandiadte.predict_proba(noncandidate)
    prob_noncandidate = prob_noncandidate[:,gmm_noncandiadte.means_.argmin()]
    # ######
    return torch.from_numpy(prob_candidate),torch.from_numpy(prob_noncandidate)

def generate_random_candidate_labels(num_sample,num_class,a=0.5,normalize=True):
    prob=np.random.uniform(0,1,size=(num_sample,num_class))
    random_targets=torch.zeros(num_sample,num_class)
    for i in range(num_sample):
        for j in range(num_class):
            if prob[i][j]<a:
                random_targets[i][j]=1
        if random_targets[i].sum()==0: # if no candidate label
            random_index=random.randint(0,num_class-1)
            random_targets[i][random_index]=1
    if normalize:
        random_targets=random_targets/random_targets.sum(dim=1,keepdim=True)
    return random_targets

def generate_one_random_label(num_sample,num_class):
    random_targets = torch.zeros(num_sample, num_class)
    for i in range(num_sample):
        random_label=random.randint(0,num_class-1)
        random_targets[i][random_label] = 1
    return random_targets

def log_args(args,log):
    for key,value in vars(args).items():
        log.write(key+":{}\n".format(value))
    log.flush()

def show_img(imgs,true_labels):
    classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
    all_outputs=torch.load('./save/all_outputs.pt')
    history_outputs = torch.from_numpy(np.vstack(all_outputs)).view(len(all_outputs), 70000, -1)
    partialY = torch.load('./save/partialY.pt')
    reversedY=(1+partialY)%2
    true_normal_index = torch.load('./save/true_normal_index.pt')
    true_cs_index = torch.load('./save/true_cs_index.pt')
    true_os_index = torch.load('./save/true_os_index.pt')
    outputs = history_outputs[-5:].mean(dim=0)
    candidate_loss, noncandidate_loss = cal_loss(outputs, partialY)
    sm_outputs=F.softmax(outputs,dim=1)*partialY
    revered_sm_outputs=F.softmax(outputs,dim=1)*reversedY
    confidence=sm_outputs/sm_outputs.sum(dim=1,keepdim=True)
    reversed_confidence = revered_sm_outputs / revered_sm_outputs.sum(dim=1, keepdim=True)
    #######
    selected_normal_index=(noncandidate_loss-candidate_loss).sort(descending=True)[1][0:int(40000)]
    selected_cs_index=(noncandidate_loss-candidate_loss).sort(descending=False)[1][0:int(10000)]
    selected_os_index=(noncandidate_loss+candidate_loss).sort(descending=True)[1][0:int(20000)]
    #######
    ######
    show_normal_index=return_same_idx(true_normal_index,selected_normal_index)
    selected_normal_idx=show_normal_index[0]
    normal_confidence_name='_confidence:'
    for i in range(10):
        if confidence[selected_normal_idx][i]!=0:
            normal_confidence_name+=classes[i]+'='+str(confidence[selected_normal_idx][i].item())
    normal_name='normal_ty:'+classes[true_labels[selected_normal_idx]]+normal_confidence_name+'.png'
    ######
    ######
    show_cs_index=return_same_idx(true_cs_index,selected_cs_index)
    selected_cs_idx=show_cs_index[1]
    cs_confidence_name='_confidence:'
    for i in range(10):
        if reversed_confidence[selected_cs_idx][i]!=0:
            cs_confidence_name+=classes[i]+'='+str(reversed_confidence[selected_cs_idx][i].item())
    cs_name='cs_ty:'+classes[true_labels[selected_cs_idx]]+cs_confidence_name+'.png'
    ######
    ######
    show_os_index=return_same_idx(true_os_index,selected_os_index)
    selected_os_idx=show_os_index[1]
    os_confidence_name='_confidence:'
    for i in range(10):
        if confidence[selected_os_idx][i]!=0:
            os_confidence_name+=classes[i]+'='+str(confidence[selected_os_idx][i].item())
    os_name='os_ty:'+classes[true_labels[selected_os_idx]]+os_confidence_name+'.png'
    ######
    imwrite('imgs/'+normal_name,imgs[selected_normal_idx])
    imwrite('imgs/' + cs_name, imgs[selected_cs_idx])
    imwrite('imgs/' + os_name, imgs[selected_os_idx])
    return

