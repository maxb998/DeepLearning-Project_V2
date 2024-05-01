import torch, yaml, math
import torchvision as tv

class Converter:
    big_grid_res = 4

    netin_img_shape:int # size of the image input of the net (no downscaling)

    grids_count:int # number of grids of anchors applied on top of the image for detection
    grids_shape:torch.Tensor    # number of cell each row/colums(equals since shape is square) of each grid
    grids_relative_area_sqrt:torch.Tensor # size in pixels of grids cell for all grids
    abox_count:int  # number of anchor boxes used
    abox_default_ratio_sqrt:torch.Tensor # tensor of size [2, abox_count] containing reference(default) values for ratio (under sqrt) of each anchor box
    netout_dtype:torch.dtype

    big_image_split_overlap:int=int(1)

    # both of these consider an netin_img_shape of at most 5120 (no need to go above) which are 8 grids
    netout_grid_limits:torch.Tensor # tensor containing the number of the limiting row of each grid in netout
    netout_center_multiplier:torch.Tensor
    netout_center_offset:torch.Tensor
    netout_abox_multiplier:torch.Tensor # tensor used to multiply netout anchor boxes to convert their values into pixels values (except for ratio)
    netout_abox_offset:torch.Tensor # tensor containg the value of the offset to make each center match the cell location on the grid

    image_scaler:tv.transforms.Resize

    def __init__(self, netin_img_shape:int=512, netout_dtype:torch.dtype=torch.float32, abox_def_file_path:str='./anchor_boxes.yaml') -> None:

        self.netout_dtype = netout_dtype

        assert math.log2(netin_img_shape) % 1 == 0 and netin_img_shape >= 64
        self.netin_img_shape = netin_img_shape

        # read anchor box file
        with open(abox_def_file_path) as f:
            anchor_dict = yaml.load(f, Loader=yaml.FullLoader)

        self.abox_count = len(anchor_dict)
        self.abox_default_ratio_sqrt = torch.zeros(size=[self.abox_count], dtype=netout_dtype)
        for i in range(self.abox_count):
            self.abox_default_ratio_sqrt[i] = math.sqrt(anchor_dict[i]['ratio'])

        self.netin_img_shape = netin_img_shape # to work with init function

        # setup here using self.max_netin
        self.grids_count = int(round(math.log2(self.netin_img_shape / self.big_grid_res) - 3) + 1)
        self.grids_shape = torch.pow(2, torch.arange(self.grids_count, dtype=torch.int32)).mul_(self.big_grid_res)#.round_().int()
        self.grids_relative_area_sqrt = torch.divide(2., self.grids_shape)

        # setup netout related data(computationally more expensive so only do once)
        grids_cells_count = self.grids_shape.square()
        self.netout_grid_limits = torch.zeros(self.grids_count+1, dtype=torch.int32)
        for i in range(self.grids_count):
            self.netout_grid_limits[i+1] = self.netout_grid_limits[i] + grids_cells_count[i]
        
        mem_format = torch.contiguous_format
        self.netout_center_offset = torch.empty(size=(self.netout_grid_limits[-1],2), dtype=netout_dtype, memory_format=mem_format)
        self.netout_center_multiplier = torch.empty(size=(self.netout_grid_limits[-1],2), dtype=netout_dtype, memory_format=mem_format)
        self.netout_abox_offset = torch.empty(size=(self.netout_grid_limits[-1],self.abox_count,2), dtype=netout_dtype, memory_format=mem_format)
        self.netout_abox_multiplier = torch.empty(size=(self.netout_grid_limits[-1],self.abox_count,2), dtype=netout_dtype, memory_format=mem_format)
        for i in range(self.grids_count):
            l1, l2 = self.netout_grid_limits[i], self.netout_grid_limits[i+1] # to make the code more readable

            # set multiplier tensor
            self.netout_center_multiplier[l1:l2,:] = 1. / self.grids_shape[i].type(self.netout_dtype)
            self.netout_abox_multiplier[l1:l2,:,0] = 3 / self.grids_shape[i].type(self.netout_dtype) # 3
            self.netout_abox_multiplier[l1:l2,:,1] = 1 # ratio does not scale with grid size

            # set offset tensor
            # aranged_tensor = torch.arange(start=self.netout_center_multiplier[l1, 0]/2, end=1, step=self.netout_center_multiplier[l1, 0], dtype=netout_dtype) # makes center_offsets have range [-0.5,0.5]
            aranged_tensor = torch.arange(start=1e-6, end=1-self.netout_center_multiplier[l1, 0]/2, step=self.netout_center_multiplier[l1, 0], dtype=netout_dtype) # makes center_offsets have range [0,1]
            self.netout_center_offset[l1:l2,0] = aranged_tensor.repeat(self.grids_shape[i])
            self.netout_center_offset[l1:l2,1] = aranged_tensor.repeat_interleave(self.grids_shape[i])
            self.netout_abox_offset[l1:l2,:,0] = 1/self.grids_shape[i].type(self.netout_dtype) # 1 # area offset
            self.netout_abox_offset[l1:l2,:,1] = self.abox_default_ratio_sqrt - 0.5 #* self.netout_abox_multiplier[l1,:,1] # ratio offset
        
        # flatten to work best with multiplication
        self.netout_abox_offset = self.netout_abox_offset.flatten(1,2)
        self.netout_abox_multiplier = self.netout_abox_multiplier.flatten(1,2)

        self.image_scaler = tv.transforms.Resize((512,512))


    # Given an input image, scales it to match netin shape, without stretching the image(apply letterbox effect)
    def apply_letterbox_to_image(self, img:torch.Tensor) -> torch.Tensor:

        # assert img.shape[1] <= self.netin_img_shape and img.shape[2] <= self.netin_img_shape

        if not (img.shape[1] == self.netin_img_shape and img.shape[2] == self.netin_img_shape):

            letterboxed_img = torch.ones(size=[3,self.netin_img_shape,self.netin_img_shape], dtype=img.dtype)
            rnd_color = torch.randint(low=0, high=255, size=(3,1,1), dtype=img.dtype)
            letterboxed_img.mul_(rnd_color)

            img_scale_coeff = self.netin_img_shape / max(img.shape)

            if img_scale_coeff != 1:
                # Scale image if necessary
                self.image_scaler.size = (int(round(img.shape[1] * img_scale_coeff)), int(round(img.shape[2] * img_scale_coeff)))
                img = self.image_scaler.forward(img)

            start_pt = int(round((self.netin_img_shape - img.shape[1]) / 2)) , int(round((self.netin_img_shape - img.shape[2]) / 2))
            letterboxed_img[:, start_pt[0]:start_pt[0]+img.shape[1], start_pt[1]:start_pt[1]+img.shape[2]] = img
        
            return letterboxed_img
        
        else:
            return img
    
    # Re-fit bounding boxes to letterbox scaled image
    def apply_letterbox_to_labels(self, labels:torch.Tensor, original_img_shape:torch.Size):
        if not original_img_shape[1] == original_img_shape[2] and not labels.shape[0] == 0:

            scale_coeff = self.netin_img_shape / max(original_img_shape)

            scaled_img_rel_shape = torch.tensor((original_img_shape[2], original_img_shape[1]), dtype=labels.dtype).mul_(scale_coeff).div_(self.netin_img_shape)

            labels[:,3:5].mul_(scaled_img_rel_shape)
            labels[:,1:3].mul_(scaled_img_rel_shape)

            lsr = torch.sub(1, scaled_img_rel_shape).div_(2)
            labels[:,1:3].add_(lsr)

    # Converts the labels values to be absolute in pixel instead of being relative to image width and height
    def convert_labels_from_relative_to_absolute_values(self, labels:torch.Tensor, original_img_shape:torch.Size=None):
        if original_img_shape is None:
            labels[...,1:].mul_(self.netin_img_shape)
        else:
            labels[...,1:].mul_(torch.tensor((original_img_shape[2], original_img_shape[1], original_img_shape[2], original_img_shape[1]), dtype=labels.dtype))

        # round allowing for .25 .5 .75 or integers
        # labels.mul_(4)
        # labels.round_()
        # labels.div_(4)

    def convert_labels_from_absolute_to_relative_values(self, labels:torch.Tensor, original_img_shape:torch.Size=None):
        if original_img_shape is None:
            labels[...,1:].div_(self.netin_img_shape)
        else:
            labels[...,1:].div_(torch.tensor((original_img_shape[2], original_img_shape[1], original_img_shape[2], original_img_shape[1]), dtype=labels.dtype))

    
    def convert_boxes_cxcywh_to_x1y1x2y2(self, cxcywh:torch.Tensor) -> torch.Tensor:
        assert cxcywh.shape[-1] == 4

        x1y1x2y2 = cxcywh[:,:2].repeat((1,2))

        half_wh = cxcywh[:,2:].mul(0.5)
        x1y1x2y2[:,:2].sub_(half_wh)
        x1y1x2y2[:,2:].add_(half_wh)

        return x1y1x2y2

    def find_repeats_pts_for_big_imgs(self, img_shape:int) -> torch.Tensor:
        if img_shape <= self.netin_img_shape:
            return torch.tensor((0,), dtype=torch.int)
        
        remains = img_shape - 2 * self.netin_img_shape * (self.big_grid_res - 1) / self.big_grid_res
        if remains <= 0:
            return torch.tensor((0,img_shape-self.netin_img_shape), dtype=torch.int)
        
        extra_repeats = int(math.ceil(remains / self.netin_img_shape * (self.big_grid_res - 2) / self.big_grid_res))
        retVal = torch.zeros((extra_repeats+2,), dtype=torch.int)
        retVal[-1] = img_shape - self.netin_img_shape
        offset = int(math.floor((img_shape - self.netin_img_shape) / (extra_repeats + 1)))
        for i in range(1, extra_repeats+1, 1):
            retVal[i] = retVal[i-1] + offset

        return retVal

    def split_big_image(self, img:torch.Tensor) -> list[torch.Tensor]:

        assert max(img.shape) > self.netin_img_shape

        x_repeats = self.find_repeats_pts_for_big_imgs(img.shape[2])
        y_repeats = self.find_repeats_pts_for_big_imgs(img.shape[1])

        # find img
        img_split:list[torch.Tensor] = []
        for yi in y_repeats:
            for xi in x_repeats:
                letterboxed_img = self.apply_letterbox_to_image(img[:, yi:min(yi+self.netin_img_shape, img.shape[1]), xi:min(xi+self.netin_img_shape, img.shape[2])])
                img_split.append(letterboxed_img)
            
        return img_split
    
    def split_big_labels(self, labels:torch.Tensor, img_shape:torch.Size) -> list[torch.Tensor]:

        assert max(img_shape) > self.netin_img_shape

        x_repeats = self.find_repeats_pts_for_big_imgs(img_shape[2])
        y_repeats = self.find_repeats_pts_for_big_imgs(img_shape[1])

        labels_absolute = labels.clone()
        self.convert_labels_from_relative_to_absolute_values(labels_absolute, img_shape)

        labels_split:list[torch.Tensor] = []
        for yi in y_repeats:
            for xi in x_repeats:
                limits_low = torch.tensor((xi,yi), dtype=labels_absolute.dtype)
                limits_high = limits_low.add(self.netin_img_shape)

                mask = labels_absolute[:,1:3] > limits_low
                mask.logical_and_(labels_absolute[:,1:3] < limits_high)
                mask = torch.logical_and(mask[:,0], mask[:,1])

                labels_split.append(labels_absolute[mask].clone(memory_format=torch.contiguous_format))
                labels_split[-1][:,1:3].sub_(torch.tensor((xi,yi), dtype=labels.dtype))
                cur_shape = (3, min(self.netin_img_shape, img_shape[1]-yi), min(self.netin_img_shape, img_shape[2]-xi))
                self.convert_labels_from_absolute_to_relative_values(labels_split[-1], cur_shape)
                self.apply_letterbox_to_labels(labels_split[-1], cur_shape)
        
        return labels_split


    # Convert labels(with values relative to image size(between 0 and 1)) to network output shape(USEFUL TO LOSS FUNCTION, SO A BIT DIFFERENT)
    def convert_labels_to_losslabels (self, labels:torch.Tensor) -> torch.Tensor:

        loss_labels = torch.zeros((self.netout_grid_limits[self.grids_count], 8), dtype=self.netout_dtype)

        if labels == None:
            return loss_labels
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(0)

        expanded_labels = labels.repeat_interleave(repeats=self.grids_count, dim=0) # preserve labels

        boxes_relative_area_sqrt = torch.sqrt(labels[:,3] * labels[:,4]).repeat_interleave(repeats=self.grids_count)
        boxes_ratio_sqrt = torch.sqrt(labels[:,3] / labels[:,4]).repeat_interleave(repeats=self.grids_count)

        grids_map = torch.logical_and(boxes_relative_area_sqrt > self.grids_relative_area_sqrt[:self.grids_count].repeat(labels.shape[0]) / 2, 
                                      boxes_relative_area_sqrt < self.grids_relative_area_sqrt[:self.grids_count].repeat(labels.shape[0]) * 2)
        
        ones = torch.zeros(size=(labels.shape[0], self.grids_count), dtype=torch.int32)
        ones[grids_map.reshape(labels.shape[0], self.grids_count)] = 1
        
        chosen_abox_idxs = boxes_ratio_sqrt.repeat((self.abox_count,1)).t_().sub_(self.abox_default_ratio_sqrt).abs_().argmin(-1)

        grid_x = self.grids_shape[:self.grids_count].repeat(labels.shape[0]).float().mul_(expanded_labels[:,1]).floor_()
        grid_y = self.grids_shape[:self.grids_count].repeat(labels.shape[0]).float().mul_(expanded_labels[:,2]).floor_()
        grid_y.mul_(self.grids_shape[:self.grids_count].repeat(labels.shape[0]))
        netout_coords = torch.add(grid_x, grid_y).type(torch.int32)
        netout_coords.add_(self.netout_grid_limits[:self.grids_count].repeat(labels.shape[0]))

        boxes_relative_area_sqrt = boxes_relative_area_sqrt[grids_map]
        netout_coords = netout_coords[grids_map]
        basic_expanded_labels = expanded_labels[grids_map]
        chosen_abox_idxs = chosen_abox_idxs[grids_map]
        boxes_ratio_sqrt = boxes_ratio_sqrt[grids_map]

        for i in range(basic_expanded_labels.shape[0]):

            coord = netout_coords[i]
            abox_idx = chosen_abox_idxs[i]
            
            loss_labels[coord, 0] = 1
            loss_labels[coord, 1] = int(torch.floor(basic_expanded_labels[i,0] / 6)) # 0 for tank, 1 for flag
            loss_labels[coord, 2] = basic_expanded_labels[i,0] % 6 # color class
            loss_labels[coord, 3:5] = (basic_expanded_labels[i,1:3] - self.netout_center_offset[coord]) / self.netout_center_multiplier[coord] # center
            loss_labels[coord, 5] = abox_idx # anchor box identifier
            loss_labels[coord, 6] = (boxes_relative_area_sqrt[i] - self.netout_abox_offset[coord, abox_idx * 2]) / self.netout_abox_multiplier[coord, abox_idx * 2] # area
            loss_labels[coord, 7] = (boxes_ratio_sqrt[i] - self.netout_abox_offset[coord, abox_idx * 2 + 1]) / self.netout_abox_multiplier[coord, abox_idx * 2 + 1] # ratio

        return loss_labels

    def convert_losslabels_to_labels(self, loss_labels:torch.Tensor) -> torch.Tensor:

        assert loss_labels.shape == (self.netout_grid_limits[self.grids_count], 8)

        threshold_mask = loss_labels[:, 0] == 1  
        thresh_loss_labels = loss_labels[threshold_mask].clone()
        
        # fit centers, areas and ratios
        thresh_loss_labels[:,3:5].mul_(self.netout_center_multiplier[:loss_labels.shape[-2]][threshold_mask])
        thresh_loss_labels[:,3:5].add_(self.netout_center_offset[:loss_labels.shape[-2]][threshold_mask])

        abox_idxs = torch.multiply(thresh_loss_labels[:,5], 2).type(torch.int32)
        aranged_tensor = torch.arange(thresh_loss_labels.shape[-2])
        thresh_loss_labels[:,6].mul_(self.netout_abox_multiplier[:loss_labels.shape[-2]][threshold_mask][aranged_tensor, abox_idxs])
        thresh_loss_labels[:,7].mul_(self.netout_abox_multiplier[:loss_labels.shape[-2]][threshold_mask][aranged_tensor, abox_idxs+1])
        thresh_loss_labels[:,6].add_(self.netout_abox_offset[:loss_labels.shape[-2]][threshold_mask][aranged_tensor, abox_idxs])
        thresh_loss_labels[:,7].add_(self.netout_abox_offset[:loss_labels.shape[-2]][threshold_mask][aranged_tensor, abox_idxs+1])

        thresh_loss_labels[:,1].mul_(6)
        thresh_loss_labels[:,1].add_(thresh_loss_labels[:, 2])

        aranged_tensor = torch.arange(thresh_loss_labels.shape[0], dtype=torch.int32)
        thresh_loss_labels[:, 2] = thresh_loss_labels[aranged_tensor, 3] #cx
        thresh_loss_labels[:, 3] = thresh_loss_labels[aranged_tensor, 4] #cy
        thresh_loss_labels[:, 4] = thresh_loss_labels[aranged_tensor, 6] #area (after appling multiplier above this is the same as width)
        thresh_loss_labels[:, 5] = thresh_loss_labels[aranged_tensor, 7] #ratio

        # convert labels from [p,class,cx,cy,r,s] notation to nms_labels[x1,y1,x2,y2] (needed for nms) and labels[p, class, cx, cy, width, height] (needed as output)
        labels = thresh_loss_labels[:, 1:6].clone(memory_format=torch.contiguous_format)
        w = torch.multiply(labels[:,3], labels[:,4]) # get width
        torch.divide(labels[:,3], labels[:,4], out=labels[:,4]) # get height
        labels[:,3] = w

        return labels


    def convert_netout_to_labels(self, netout:torch.Tensor, probability_threshold:float=0.6, max_iou:float=0.6, apply_nms:bool=True) -> torch.Tensor:

        assert netout.shape == (self.netout_grid_limits[self.grids_count], 1+1+6+2+self.abox_count*(1+2))

        if netout.requires_grad:
            netout = netout.detach().clone()

        threshold_mask = netout[:, 0] > probability_threshold
        thresholded_netout = netout[threshold_mask]
        
        thresholded_netout[:, 1].round_()
        thresholded_netout[:, 1].mul_(6)
        argmax_tensor = torch.argmax(thresholded_netout[:, 2:8], dim=-1)
        thresholded_netout[:, 1].add_(argmax_tensor)
        
        # fit center
        thresholded_netout[:, 8:10].mul_(self.netout_center_multiplier[:netout.shape[-2]][threshold_mask])
        thresholded_netout[:, 8:10].add_(self.netout_center_offset[:netout.shape[-2]][threshold_mask])

        # fit area and ratio
        thresholded_netout[:, 10+self.abox_count:].mul_(self.netout_abox_multiplier[:netout.shape[-2]][threshold_mask])
        thresholded_netout[:, 10+self.abox_count:].add_(self.netout_abox_offset[:netout.shape[-2]][threshold_mask])

        # move centers
        thresholded_netout[:, 2:4] = thresholded_netout[:, 8:10]

        # identify best abox and move it
        argmax_tensor = torch.argmax(thresholded_netout[:, 10:10+self.abox_count], dim=-1)
        thresholded_netout[:, 6] = argmax_tensor
        argmax_tensor.mul_(2)
        aranged_tensor = torch.arange(thresholded_netout.shape[-2])
        thresholded_netout[:, 4] = thresholded_netout[aranged_tensor, argmax_tensor + 10 + self.abox_count]
        thresholded_netout[:, 5] = thresholded_netout[aranged_tensor, argmax_tensor + 11 + self.abox_count]

        # convert labels from [p,class,cx,cy,r,s] notation to nms_labels[x1,y1,x2,y2] (needed for nms) and labels[p, class, cx, cy, width, height] (needed as output)
        labels = thresholded_netout[:, :6].clone()
        w = torch.multiply(labels[:,4], labels[:,5]) # get width
        torch.divide(labels[:,4], labels[:,5], out=labels[:,5]) # get height
        labels[:,4] = w

        if apply_nms:
            nms_labels = labels[:,2:4].repeat((1,2))
            nms_labels.add_(torch.cat((-labels[:,4:6], labels[:,4:6]), dim=1))

            remaining_boxes = tv.ops.batched_nms(boxes=nms_labels, scores=labels[:,0], idxs=labels[:,1], iou_threshold=max_iou)

            labels = labels[remaining_boxes].clone(memory_format=torch.contiguous_format)

        return labels

