from torch.utils.data import Dataset
import torchvision.transforms as transforms

class gen_index_dataset(Dataset):
    def __init__(self, images, given_label_matrix, true_labels):
        self.images = images
        self.given_label_matrix = given_label_matrix
        self.true_labels = true_labels
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        
    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        each_image = self.images[index]
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]

        each_image_1 = self.transform(each_image)
        each_image_2 = self.transform(each_image)
        
        return each_image_1,each_image_2, each_label, each_true_label, index
    '''
    def __init__(self, user_input, item_input, ratings):
        self.user_input = user_input
        self.item_input = item_input
        self.ratings = ratings
    
    def __len__(self):
        return len(self.user_input)
    
    def __getitem__(self, index):
        user_id = self.user_input[index]
        item_id = self.item_input[index]
        rating = self.ratings[index]
        
        return {'user_id': user_id,
                'item_id': item_id,
                'rating': rating}
    '''