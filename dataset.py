import torch, os
import numpy as np
from PIL import Image
from torchvision import transforms


class RisikoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir:str, mode:str, transform=None):
        if mode != "train" and mode != "val" and mode != "test" and mode != "real":
            raise Exception("Mode value of dataset not valid")

        imgs_dir = dataset_dir + "/" + mode + "/images"
        labels_dir = dataset_dir + "/" + mode + "/labels"

        self.images = sorted( filter( lambda x: os.path.isfile(os.path.join(imgs_dir, x)), os.listdir(self.imgs_dir) ) )
        self.labels = map(lambda x: os.path.join(labels_dir, os.path.splitext(os.path.basename(x))[0]) + ".txt", self.images)

        for lbl in self.labels:
            if not os.path.isfile(lbl):
                print("Error: file \"" + lbl + "\" does not exists")
                exit()

        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx:int, mode_plot:bool=False) -> tuple[torch.Tensor, torch.Tensor]:
        annotations_file_data = np.genfromtxt(fname= self.annots_dir + "/" + self.annotations[idx], delimiter=' ', dtype=np.float32)
        classes, bboxes = np.hsplit(annotations_file_data, np.array([1]))
        #classes[:] = 0 #to check if there are issues with classification

        basic_annotations = torch.cat([torch.from_numpy(bboxes), torch.from_numpy(classes)], 1)
        
        img = Image.open(self.imgs_dir + "/" + self.images[idx]).convert("RGB")

        if self.transform: img = self.transform(img)
        
        if mode_plot: return img, basic_annotations

        pil_to_tensor = transforms.Compose([transforms.PILToTensor()])
        img:torch.Tensor = pil_to_tensor(img)

        # normalize image from 0 to 1
        #img = img.to(torch.float16) / 256

        annotations = torch.ones([300,5], dtype=torch.float32) * -1
        annotations[:basic_annotations.size()[0], ...] = basic_annotations

        return img, annotations