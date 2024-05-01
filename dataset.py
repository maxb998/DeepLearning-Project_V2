import torch, os
import numpy as np
import torchvision as tv
from utils import Converter
from tqdm import tqdm

anchor_boxes_yaml_location = './anchor_boxes.yaml'

class RisikoDataset(torch.utils.data.Dataset):

    is_trainset:bool
    gauss_kernels:tuple[tuple[int,int], ...]
    gauss_sigmas:tuple[tuple[float,float], ...]
    salt_pepper_hval:int
    cv:Converter
    images:list[torch.Tensor]
    images_shape:list[torch.Size]
    labels:list[torch.Tensor]
    loss_labels:list[torch.Tensor]
    img_paths:list[str]
    norm = tv.transforms.Normalize(mean=(138.330, 116.681, 88.420), std=(51.176, 47.243, 48.665), inplace=False)
    invnorm = tv.transforms.Normalize(mean=(-138.330/51.176, -116.681/47.243, -88.420/48.665), std=(1/51.176, 1/47.243, 1/48.665), inplace=False) # EXTERNAL USE ONLY


    def __init__(self,
                 dataset_dir:str,
                 cv:Converter,
                 is_trainset:bool=False,
                 gauss_kernels:tuple[tuple[int,int], ...] = ( (3,3), (1,3), (3,1) ),
                 gauss_sigmas:tuple[tuple[float,float], ...] = ( (0.1, 0.75), (0.05, 1.), (0.01, 1.25) ),
                 salt_pepper_hval:int = 200,
                 labels_extension:str='.csv',
                 ):
        
        self.is_trainset = is_trainset
        self.gauss_kernels = gauss_kernels
        self.gauss_sigmas = gauss_sigmas
        self.salt_pepper_hval = salt_pepper_hval
        self.cv = cv

        self.images = []

        self.labels = []
        self.loss_labels = []
        
        imgs_dir = os.path.join(dataset_dir, 'images')
        labels_dir = os.path.join(dataset_dir, 'labels')

        self.img_paths = sorted( filter( lambda x: os.path.isfile(os.path.join(imgs_dir, x)), os.listdir(imgs_dir) ) )

        prog_bar = tqdm(total=len(self.img_paths))


        for i in range(len(self.img_paths)):
            self.img_paths[i] = os.path.join(imgs_dir, self.img_paths[i])

            # load image and label
            img = tv.io.read_image(self.img_paths[i], tv.io.ImageReadMode.RGB)

            label_fname = os.path.join(labels_dir, os.path.splitext(os.path.basename(self.img_paths[i]))[0]) + labels_extension
            if os.path.isfile(label_fname):
                label = torch.from_numpy(np.genfromtxt(fname=label_fname, delimiter=' ', dtype=np.float32))

                if label.shape[0] > 0:
                    if len(label.shape) == 1: # avoid errors (dimension errors with images containing only one object)
                        label.unsqueeze_(0)
                else:
                    label = torch.empty((0,5))
            else:
                label = torch.empty((0,5))

            if img.shape[1] > cv.netin_img_shape or img.shape[2] > cv.netin_img_shape:
                imgs = cv.split_big_image(img)
                labels = cv.split_big_labels(label, img.shape)
            else:
                letterboxed_img = cv.apply_letterbox_to_image(img)
                cv.apply_letterbox_to_labels(label, img.shape)
                imgs, labels = [letterboxed_img,], [label,]
            
            self.images.extend(imgs)
            self.labels.extend(labels)

            if is_trainset:
                for lbl in labels:
                    self.loss_labels.append(cv.convert_labels_to_losslabels(lbl))
            
            prog_bar.update(1)
        
        prog_bar.close()

                
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx:int) -> tuple[torch.Tensor, torch.Tensor]:

        img = self.images[idx].clone()

        # agumentation step
        if self.is_trainset:
            agumentation_modes_array = torch.randperm(2, dtype=torch.int)

            for i in agumentation_modes_array:
                if torch.randint(low=0, high=2, size=(1,))[0] == 1:
                    if i == 0: # gaussian blur
                        gauss_kernel = self.gauss_kernels[torch.randint(low=0, high=len(self.gauss_kernels), size=(1,))[0]]
                        gauss_sigma = self.gauss_sigmas[torch.randint(low=0, high=len(self.gauss_sigmas), size=(1,))[0]]
                        gauss_transform = tv.transforms.GaussianBlur(gauss_kernel, gauss_sigma)
                        img = gauss_transform(img)

                    elif i == 1: # salt and pepper
                        #generate random tensor, threshold it and apply to some pixel a randomly generated color
                        rnd_tensor = torch.randint_like(input=img, low=0, high=self.salt_pepper_hval+1)
                        mask = rnd_tensor == self.salt_pepper_hval
                        rnd_tensor = torch.randint_like(input=img[mask], low=0, high=255)
                        img[mask] = rnd_tensor

        img = self.norm(img.type(self.cv.netout_dtype))

        if self.is_trainset:
            return img, self.loss_labels[idx]
        else:
            return img, self.labels[idx]
    

    def get_img_shape(self, index:int) -> torch.Size:
            
        img_shape = self.images[index].shape

        return img_shape
