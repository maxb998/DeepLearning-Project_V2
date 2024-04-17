import torch, yaml, math
import torchvision as tv
from sys import exit

class Converter:
    big_grid_res = 4

    original_img_shape:torch.Size
    img_upscale_coeff:float # multiplier applied to image to scale to netin_img_shape
    netin_img_shape:int # size of the image input of the net (no downscaling)
    grids_count:int # number of grids of anchors applied on top of the image for detection
    grids_shape:torch.Tensor    # number of cell each row/colums(equals since shape is square) of each grid
    grids_relative_area:torch.Tensor # size in pixels of grids cell for all grids
    abox_count:int  # number of anchor boxes used
    abox_default_scale_ratio:torch.Tensor # tensor of size [2, abox_count] containing reference(default) values for scale and ratio of each anchor box
    netout_dtype:torch.dtype

    # both of these consider an netin_img_shape of at most 5120 (no need to go above) which are 8 grids
    max_netin:int=5120
    max_grids:int
    netout_grid_limits:torch.Tensor # tensor containing the number of the limiting row of each grid in netout
    netout_center_multiplier:torch.Tensor
    netout_center_offset:torch.Tensor
    netout_abox_multiplier:torch.Tensor # tensor used to multiply netout anchor boxes to convert their values into pixels values (except for ratio)
    netout_abox_offsets:torch.Tensor # tensor containg the value of the offset to make each center match the cell location on the grid

    def __init__(self, netout_dtype:torch.dtype=torch.float32, abox_def_file_path:str='./anchor_boxes.yaml') -> None:

        assert (
            netout_dtype == torch.float16 or
            netout_dtype == torch.float32
        )
        self.netout_dtype = netout_dtype

        # read anchor box file
        with open(abox_def_file_path) as f:
            anchor_dict = yaml.load(f, Loader=yaml.FullLoader)

        self.abox_count = len(anchor_dict)
        self.abox_default_scale_ratio = torch.zeros(size=[self.abox_count, 2], dtype=torch.float32)
        for i in range(self.abox_count):
            self.abox_default_scale_ratio[i,1] = anchor_dict[i]['ratio']
            self.abox_default_scale_ratio[i,0] = 1 / self.abox_default_scale_ratio[i,1].sqrt() # keeps the area of the default anchor the same value as the grid_cell_area

        self.netin_img_shape = -1 # to work with init function

        # setup here using self.max_netin
        self.max_grids:int=round(math.log2(self.max_netin / self.big_grid_res) - 3) + 1
        self.grids_shape = torch.round(torch.pow(2, torch.arange(self.max_grids)) * self.big_grid_res).type(torch.int32)
        self.grids_relative_area = torch.divide(1., self.grids_shape.square())

        # setup netout related data(computationally more expensive so only do once)
        max_grids_cells_count = self.grids_shape.square()
        self.netout_grid_limits = torch.zeros(self.max_grids+1, dtype=torch.int32)
        for i in range(self.max_grids):
            self.netout_grid_limits[i+1] = self.netout_grid_limits[i] + max_grids_cells_count[i]
        
        mem_format = torch.contiguous_format
        self.netout_center_offset = torch.empty(size=(self.netout_grid_limits[-1],2), dtype=netout_dtype, memory_format=mem_format)
        self.netout_center_multiplier = torch.empty(size=(self.netout_grid_limits[-1],2), dtype=netout_dtype, memory_format=mem_format)
        self.netout_abox_offset = torch.empty(size=(self.netout_grid_limits[-1],self.abox_count,2), dtype=netout_dtype, memory_format=mem_format)
        self.netout_abox_multiplier = torch.empty(size=(self.netout_grid_limits[-1],self.abox_count,2), dtype=netout_dtype, memory_format=mem_format)
        for i in range(self.max_grids):
            l1, l2 = self.netout_grid_limits[i], self.netout_grid_limits[i+1] # to make the code more readable

            # set multiplier tensor
            self.netout_center_multiplier[l1:l2,:] = 1. / self.grids_shape[i].type(torch.float32)
            self.netout_abox_multiplier[l1:l2,:,0] = 2. / self.grids_shape[i].type(torch.float32)
            self.netout_abox_multiplier[l1:l2,:,1] = 1 # ratio does not scale with grid size

            # set offset tensor
            aranged_tensor = torch.arange(start=self.netout_center_multiplier[l1, 0]/2, end=1, step=self.netout_center_multiplier[l1, 0], dtype=netout_dtype)
            self.netout_center_offset[l1:l2,0] = aranged_tensor.repeat(self.grids_shape[i])
            self.netout_center_offset[l1:l2,1] = aranged_tensor.repeat_interleave(self.grids_shape[i])
            self.netout_abox_offset[l1:l2,:,0] = self.abox_default_scale_ratio[:,0] * self.netout_abox_multiplier[l1,:,0] # scale offset
            self.netout_abox_offset[l1:l2,:,1] = self.abox_default_scale_ratio[:,1] * self.netout_abox_multiplier[l1,:,1] # ratio offset


    def set_img_original_shape(self, original_img_shape:torch.Size):

        assert original_img_shape[1] <= self.max_netin and original_img_shape[2] <= self.max_netin

        self.original_img_shape = original_img_shape
        self.netin_img_shape = int(max(pow(2, math.ceil(math.log2(max(original_img_shape)/self.big_grid_res))) * self.big_grid_res, 160))
        self.img_upscale_coeff = self.netin_img_shape / max(self.original_img_shape)
        self.grids_count = round(math.log2(self.netin_img_shape / self.big_grid_res) - 3) + 1


    # Given an input image resizes it to match netin shape, without stretching the image(apply letterbox effect)
    def apply_letterbox_to_image(self, img:torch.Tensor) -> torch.Tensor:

        if img.shape != self.original_img_shape:
            self.set_img_original_shape(img.shape)

        if not (img.shape[1] == self.netin_img_shape and img.shape[2] == self.netin_img_shape):

            letterboxed_img = torch.ones(size=[3,self.netin_img_shape,self.netin_img_shape], dtype=img.dtype)
            rnd_color = torch.randint(low=0, high=255, size=3, dtype=img.dtype)
            torch.multiply(input=letterboxed_img, other=rnd_color, out=letterboxed_img)

            if self.img_upscale_coeff != 1:
                # Upscale image if necessary
                upscaler = tv.transforms.Compose(tv.transforms.Resize(size=img.shape * self.img_upscale_coeff))
                img = upscaler(img)

            start_pt = torch.empty(size=2, dtype=int)
            torch.floor(((img.shape[1:3] - self.netin_img_shape) / 2), out=start_pt)
            letterboxed_img[:, start_pt[0]:start_pt[0]+self.netin_img_shape, start_pt[1]:start_pt[1]+self.netin_img_shape] = img
        
            return letterboxed_img
        
        else:
            return img.clone()
    
    # Re-fit bounding boxes to letterbox scaled image
    def apply_letterbox_to_labels(self, labels:torch.Tensor):
        if not (self.original_img_shape[1] == self.netin_img_shape and self.original_img_shape[2] == self.netin_img_shape):

            upscale_coeff = self.netin_img_shape / max(self.original_img_shape)

            if upscale_coeff != 1:
                # scale boxes width and height
                torch.multiply(labels[:,3:5], upscale_coeff, out=labels[:,3:5])
            if self.original_img_shape[0] == self.original_img_shape[1]:
                # offset centers
                boxes_offset = labels[:,1:3] * (self.netin_img_shape - self.original_img_shape[1:][::-1]) / 2
                torch.add(labels[:,1:3], boxes_offset, out=labels[:,1:3])
    


    # Converts the labels values to be absolute in pixel instead of being relative to image width and height
    def convert_labels_from_relative_to_absolute_values(self, labels:torch.Tensor, use_netin_shape:bool=False):
        if use_netin_shape:
            torch.multiply(labels[:,1:], self.netin_img_shape, out=labels[:,1:])
        else:
            torch.multiply(labels[:,1], self.original_img_shape[2], out=labels[:,1])
            torch.multiply(labels[:,3], self.original_img_shape[2], out=labels[:,3])
            torch.multiply(labels[:,2], self.original_img_shape[1], out=labels[:,2])
            torch.multiply(labels[:,4], self.original_img_shape[1], out=labels[:,4])
        
        torch.round(labels[:,3:], out=labels[:,3:]) # numerical stability


    # Convert labels(with values relative to image size(between 0 and 1)) to network output shape
    def convert_labels_to_netout(self, labels:torch.Tensor, check_coordinate_overlap:bool=False) -> torch.Tensor:

        netout = torch.zeros((self.netout_grid_limits[self.grids_count+1], 8), dtype=self.netout_dtype)

        if labels == None:
            return netout
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(0)

        #labels = labels.clone() # preserve labels

        boxes_relative_area = labels[:,3] * labels[:,4]
        boxes_ratio = labels[:,3] / labels[:,4]

        chosen_grids_idxs = torch.abs(boxes_relative_area.repeat((self.grids_relative_area.shape[0],1)).t() - self.grids_relative_area).argmin(-1)
        chosen_grids_shape = self.grids_shape[chosen_grids_idxs]

        chosen_abox_idxs = torch.abs(boxes_ratio.repeat((self.abox_count,1)).t() - self.abox_default_scale_ratio[:,1]).argmin(-1)

        grid_x = torch.multiply(labels[:,1], chosen_grids_shape).floor()
        grid_y = torch.multiply(labels[:,2], chosen_grids_shape).floor()
        torch.multiply(grid_y, chosen_grids_shape, out=grid_y)
        netout_coords = torch.add(grid_x, grid_y).type(torch.int32)
        torch.add(netout_coords, self.netout_grid_limits[chosen_grids_idxs], out=netout_coords)

        for i in range(labels.shape[0]):
            coord = netout_coords[i]
            abox_idx = chosen_abox_idxs[i]

            if check_coordinate_overlap:
                if coord in netout_coords[:i]:
                    print('There is already a object at the selected coordinate[' + str(i) + '] of ' + str(netout_coords[i]))
                    print('Coords already occupied ' + str(netout_coords))
                    exit()
            
            netout[coord, 0] = 1
            netout[coord, 1] = int(torch.floor(labels[i,0] / 6)) # 0 for tank, 1 for flag
            netout[coord, 2] = labels[i,0] % 6 # color class
            netout[coord, 3:5] = (labels[i,1:3] - self.netout_center_offset[coord]) / self.netout_center_multiplier[coord] # center
            netout[coord, 5] = abox_idx # anchor box identifier
            netout[coord, 6] = (labels[i,3] - self.netout_abox_offset[coord, abox_idx , 0]) / self.netout_abox_multiplier[coord, abox_idx, 0] # scale
            netout[coord, 7] = (boxes_ratio[i] - self.netout_abox_offset[coord, abox_idx, 1]) / self.netout_abox_multiplier[coord, abox_idx, 1] #ratio

        return netout

    def fit_netout(self, netout:torch.Tensor, preserve_original:bool=False):

        assert netout.shape[-1] == 8

        if preserve_original:
            netout = netout.clone()

        torch.multiply(netout[..., 3:5], self.netout_center_multiplier[:netout.shape[-2]], out=netout[..., 3:5])
        torch.add(netout[..., 3:5], self.netout_center_offset[:netout.shape[-2]], out=netout[..., 3:5])

        aboxes_idx = netout[:,5].type(torch.int32)
        aranged_tensor = torch.arange(netout.shape[-2], dtype=torch.int32)
        torch.multiply(netout[..., 6:8], self.netout_abox_multiplier[:netout.shape[-2]][aranged_tensor, aboxes_idx], out=netout[..., 6:8])
        torch.add(netout[..., 6:8], self.netout_abox_offset[:netout.shape[-2]][aranged_tensor, aboxes_idx], out=netout[..., 6:8])

        return netout

    def convert_netout_to_labels(self, netout:torch.Tensor, probability_threshold:float=0.6, max_iou:float=0.6, apply_nms:bool=True) -> torch.Tensor:

        assert len(netout.shape) == 2 # cannot do all batch at once since every image has a different number of boxes
        assert netout.shape == (self.netout_grid_limits[self.grids_count+1], 8)

        thresholded_netout = netout[netout[:, 0] > probability_threshold]

        torch.add(thresholded_netout[:, 2], ( 6 * thresholded_netout[:, 1].round()), out=thresholded_netout[:, 1])

        # choose anchor box for each thresholded detection
        # thresholded_best_aboxes_idx = thresholded_netout[:, 5].type(torch.int32)

        aranged_tensor = torch.arange(thresholded_netout.shape[0], dtype=torch.int32)
        #thresholded_netout[:, 6] = thresholded_netout[aranged_tensor, 3 + thresholded_best_aboxes_idx] # abox chance (don't know if it's even needed so it's at the end)
        # torch.multiply(thresholded_best_aboxes_idx, 2, out=thresholded_best_aboxes_idx)
        thresholded_netout[:, 2] = thresholded_netout[aranged_tensor, 3] #cx
        thresholded_netout[:, 3] = thresholded_netout[aranged_tensor, 4] #cy
        thresholded_netout[:, 4] = thresholded_netout[aranged_tensor, 6] #scale (after appling multiplier above this is the same as width)
        thresholded_netout[:, 5] = thresholded_netout[aranged_tensor, 7] #ratio

        # convert labels from [p,class,cx,cy,r,s] notation to nms_labels[x1,y1,x2,y2] (needed for nms) and labels[p, class, cx, cy, width, height] (needed as output)
        labels = thresholded_netout[:, :6].clone()
        torch.divide(labels[:,4], labels[:,5], out=labels[:,5]) # get height

        if apply_nms:
            nms_labels = labels[:,2:4].repeat((1,2))
            torch.add(nms_labels, torch.cat((-labels[:,4:6], labels[:,4:6]), dim=1), out=nms_labels)

            remaining_boxes = tv.ops.batched_nms(boxes=nms_labels, scores=labels[:,0], idxs=labels[:,1], iou_threshold=max_iou)

            labels = labels[remaining_boxes]

        return labels

