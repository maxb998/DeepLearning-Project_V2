import torch, os
import numpy as np
import torchvision as tv


class RisikoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir:str,
                 train_mode:bool=False,
                 gauss_kernels:list = [ (3,3), (5,5), (1,3), (3,1), (1,5), (5,1) ],
                 gauss_sigmas:list = [ (0.2, 5.), (0.1, 2.), (0.05, 4.) ],
                 salt_pepper_hval:int = 20,
                 rnd_keypoints_number:int = 10,
                 max_agumenting_lines:int = 30
                 ):
        
        imgs_dir = dataset_dir + "/images"
        labels_dir = dataset_dir + "/labels"

        self.images = sorted( filter( lambda x: os.path.isfile(os.path.join(imgs_dir, x)), os.listdir(self.imgs_dir) ) )
        self.labels = map(lambda x: os.path.join(labels_dir, os.path.splitext(os.path.basename(x))[0]) + ".txt", self.images)

        for lbl in self.labels:
            if not os.path.isfile(lbl):
                print("Error: file \"" + lbl + "\" does not exists")
                exit()
            
        self.train_mode = train_mode
        self.gauss_kernels = gauss_kernels
        self.gauss_sigmas = gauss_sigmas
        self.salt_pepper_hval = salt_pepper_hval
        self.rnd_keypoints_number = rnd_keypoints_number
        self.max_agumenting_lines = max_agumenting_lines


    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx:int, mode_plot:bool=False) -> tuple[torch.Tensor, torch.Tensor]:
        annotations_file_data = np.genfromtxt(fname= self.annots_dir + "/" + self.annotations[idx], delimiter=' ', dtype=np.float32)
        classes, bboxes = np.hsplit(annotations_file_data, np.array([1]))

        basic_annotations = torch.cat([torch.from_numpy(bboxes), torch.from_numpy(classes)], 1)
        img = tv.io.read_image(self.imgs_dir + "/" + self.images[idx], tv.io.ImageReadMode.RGB)

        if mode_plot:
            return img, basic_annotations

        # agumentation step
        if self.train_mode:
            agumentation_modes_array = np.arange(3, dtype=int)
            agumentation_modes_array = np.random.permutation(agumentation_modes_array) # allows for effect to apply on top of one another in a random way

            for i in agumentation_modes_array:
                if np.random.randint(2) == 1:
                    if i == 0: # gaussian blur
                        gauss_kernel = self.gauss_kernels[np.random.randint(len(self.gauss_kernels))]
                        gauss_sigma = self.gauss_sigmas[np.random.randint(len(self.gauss_sigmas))]
                        img = tv.transforms.GaussianBlur(gauss_kernel, gauss_sigma)
                    elif i == 1: # salt and pepper
                        #generate random tensor, threshold it and apply to some pixel a randomly generated color
                        rnd_tensor = torch.randint_like(input=img, low=0, high=self.salt_pepper_hval+1)
                        mask = rnd_tensor < self.salt_pepper_hval
                        rnd_tensor = torch.randint_like(input=img, low=0, high=255)
                        img[mask] = rnd_tensor[mask]
                    elif i == 2: # random geometric shapes of random color
                        rnd_color = torch.randint(low=0, high=255, size=3, dtype=img.dtype)

                        # random points
                        rnd_keypoints = torch.zeros(size=(self.rnd_keypoints_number, 2), dtype=torch.int)
                        rnd_keypoints[0:] = torch.randint(low=0, high=img.size(2), dtype=rnd_keypoints.dtype, size=self.rnd_keypoints_number) # X width
                        rnd_keypoints[1:] = torch.randint(low=0, high=img.size(1), dtype=rnd_keypoints.dtype, size=self.rnd_keypoints_number) # Y height

                        # connect points at random
                        connections_count = torch.randint(low=0,high=self.max_agumenting_lines, dtype=torch.int)
                        rnd_connections = torch.randint(low=0, high=self.rnd_keypoints_number, size=(connections_count, 2), dtype=torch.int)
                        img = tv.utils.draw_keypoints(img, keypoints=rnd_keypoints, connectivity=rnd_connections, colors=rnd_color, radius=1, width=1)

        # normalize image from 0 to 1
        #img = img.to(torch.float16) / 256

        annotations = torch.ones([300,5], dtype=torch.float32) * -1
        annotations[:basic_annotations.size()[0], ...] = basic_annotations

        return img, annotations