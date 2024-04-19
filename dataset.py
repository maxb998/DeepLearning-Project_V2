import torch, os
import numpy as np
import torchvision as tv
from utils import Converter
from tqdm import tqdm

anchor_boxes_yaml_location = './anchor_boxes.yaml'

class RisikoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir:str, cv:Converter,
                 train_mode:bool=False,
                 gauss_kernels:tuple[tuple[int,int], ...] = ( (3,3), (5,5), (1,3), (3,1), (1,5), (5,1) ),
                 gauss_sigmas:tuple[tuple[float,float], ...] = ( (0.1, 2.), (0.05, 2.75), (0.01, 3.5) ),
                 salt_pepper_hval:int = 200,
                 load_all_images_in_memory:bool = False,
                 ):
        
        self.train_mode:bool = train_mode
        self.gauss_kernels:tuple[tuple[int,int], ...] = gauss_kernels
        self.gauss_sigmas:tuple[tuple[float,float], ...] = gauss_sigmas
        self.salt_pepper_hval:int = salt_pepper_hval
        self.imgs_loaded:bool = load_all_images_in_memory
        self.cv = cv

        self.images:list[torch.Tensor] = []
        self.images_shape:list[torch.Size] = []

        self.basic_labels:list[torch.Tensor] = []
        self.labels:list[torch.Tensor] = []
        
        imgs_dir = os.path.join(dataset_dir, 'images')
        labels_dir = os.path.join(dataset_dir, 'labels')

        self.img_paths = sorted( filter( lambda x: os.path.isfile(os.path.join(imgs_dir, x)), os.listdir(imgs_dir) ) )

        prog_bar = tqdm(total=len(self.img_paths))

        for i in range(len(self.img_paths)):
            self.img_paths[i] = os.path.join(imgs_dir, self.img_paths[i])

            # load image to get shape
            img = tv.io.read_image(self.img_paths[i], tv.io.ImageReadMode.RGB)

            # set and check shape(only in train mode)
            old_netin = self.cv.netin_img_shape
            self.cv.set_img_original_shape(img.shape)
            if self.train_mode and old_netin != self.cv.netin_img_shape and i > 0:
                prog_bar.close()
                print('Error all images must have the same netin_img_shape, but image ' + str(i) + ' has a different netin_img_shape compared to the previous image')
                exit()

            if self.imgs_loaded:
                self.images.append(cv.apply_letterbox_to_image(img))
            self.images_shape.append(img.shape)

            # load all labels here and format them correctly to spare time during training
            label_fname = os.path.join(labels_dir, os.path.splitext(os.path.basename(self.img_paths[i]))[0]) + '.csv'

            if os.path.isfile(label_fname):
                label_file_content = np.genfromtxt(fname=label_fname, delimiter=' ', dtype=np.float32)

                if label_file_content.shape[0] > 0:

                    if len(label_file_content.shape) == 1: # avoid errors (dimension errors with images containing only one object)
                        self.basic_labels.append(torch.from_numpy(label_file_content).unsqueeze(0))
                    else:
                        self.basic_labels.append(torch.from_numpy(label_file_content))

                    self.cv.apply_letterbox_to_labels(self.basic_labels[i])

                else:
                    self.basic_labels.append(None)
            else:
                self.basic_labels.append(None)
            
            if self.train_mode:
                self.labels.append(self.cv.convert_labels_to_netout(self.basic_labels[i], check_coordinate_overlap=True))
            
            prog_bar.update(1)
        
        prog_bar.close()

                
    def __len__(self) -> int:
        return len(self.img_paths)
    
    def __getitem__(self, idx:int, return_basic_labels:bool=False) -> tuple[torch.Tensor, torch.Tensor]:

        if self.imgs_loaded:
            img = self.images[idx]
        else:
            img = tv.io.read_image(self.img_paths[idx], tv.io.ImageReadMode.RGB)
            self.cv.set_img_original_shape(img.shape)
            img = self.cv.apply_letterbox_to_image(img)

        # agumentation step
        if self.train_mode:
            agumentation_modes_array = np.arange(2, dtype=int)
            agumentation_modes_array = np.random.permutation(agumentation_modes_array) # allows for effect to apply on top of one another in a random way

            for i in agumentation_modes_array:
                if np.random.randint(2) == 1:
                    if i == 0: # gaussian blur
                        gauss_kernel = self.gauss_kernels[np.random.randint(len(self.gauss_kernels))]
                        gauss_sigma = self.gauss_sigmas[np.random.randint(len(self.gauss_sigmas))]
                        gauss_transform = tv.transforms.GaussianBlur(gauss_kernel, gauss_sigma)
                        img = gauss_transform(img)

                    elif i == 1: # salt and pepper
                        #generate random tensor, threshold it and apply to some pixel a randomly generated color
                        rnd_tensor = torch.randint_like(input=img, low=0, high=self.salt_pepper_hval+1)
                        mask = rnd_tensor == self.salt_pepper_hval
                        rnd_tensor = torch.randint_like(input=img, low=0, high=255)
                        img[mask] = rnd_tensor[mask]

        # convert image type and normalize from 0 to 1
        img = img.type(self.cv.netout_dtype) / 256

        if return_basic_labels:
            return img, self.basic_labels[idx]
        else:
            return img, self.labels[idx]
    

    def get_img_shape(self, index:int) -> torch.Size:
            
        if self.imgs_loaded:
            img_shape = self.images[index].shape
        else:
            img_shape = tv.io.read_image(self.img_paths[index], tv.io.ImageReadMode.RGB).size()

        return img_shape
