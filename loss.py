import torch
import torch.nn as nn

# Since this loss is only useful during training, it's assumed that all images processed by this loss HAVE THE SAME SHAPE

class RisikoLoss(nn.Module):
    lambda_abox:float
    lambda_coord:float
    lambda_no_obj:float
    lambda_class_obj:float
    lambda_class_color:float
    abox_count:int
    device:str
    mse:nn.MSELoss
    CrossEntropyLoss:nn.CrossEntropyLoss
    aranged_tensor:torch.Tensor

    def __init__(self,
                 lambda_abox:float, 
                 lambda_coord:float, 
                 lambda_no_obj:float, 
                 lambda_class_obj:float, 
                 lambda_class_color:float, 
                 abox_count:int, 
                 batch_size:int=1, 
                 max_obj:int=600, 
                 device:str='cpu'):
        
        super(RisikoLoss, self).__init__()

        self.lambda_abox = lambda_abox
        self.lambda_coord = lambda_coord
        self.lambda_no_obj = lambda_no_obj
        self.lambda_class_obj = lambda_class_obj
        self.lambda_class_color = lambda_class_color

        self.abox_count = abox_count
        self.device = device

        self.mse = nn.MSELoss(reduction='sum')
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='sum')

        self.aranged_tensor = torch.arange(batch_size * max_obj, dtype=torch.int32).to(self.device)


    def forward(self, predictions:torch.Tensor, label:torch.Tensor):

        targets_mask = label[...,0]==1
        predictions_targets = predictions[..., targets_mask, :]
        actual_targets = label[..., targets_mask, :]

        # abox_loss, box_loss, class_obj_loss, class_color_loss, obj_loss = 0,0,0,0,0

        # if (actual_targets.shape[0] > 0):
        # Abox Loss: loss on the selection of the anchor box
        abox_loss = self.CrossEntropyLoss(predictions_targets[..., 10: 10 + self.abox_count], actual_targets[..., 5].type(torch.long))

        # Box Loss: loss on the center offset, ratio and scale multipliers
        box_loss = self.mse(predictions_targets[..., 8:10], actual_targets[..., 3:5])
        abox_base_pos = actual_targets[..., 5].type(torch.int32) * 2 + 10 + self.abox_count
        scale_ratio_tensor = torch.stack(
            (predictions_targets[self.aranged_tensor[:predictions_targets.shape[0]], abox_base_pos],
             predictions_targets[self.aranged_tensor[:predictions_targets.shape[0]], abox_base_pos+1])).t()
        box_loss += self.mse(scale_ratio_tensor, actual_targets[..., 6:8])

        # Class Loss: color loss and class obj loss which is a boolean: tank=1, flag=0
        class_obj_loss = self.mse(predictions_targets[..., 1], actual_targets[..., 1])
        class_color_loss = self.CrossEntropyLoss(predictions_targets[..., 2:8], actual_targets[..., 2].type(torch.long))
            
        # Obj Loss: loss on the probability of the object being in a cell (positive cases)
        obj_loss = self.mse(predictions_targets[..., 0], actual_targets[..., 0])

        # No Obj Loss: opposite of Obj Loss (negative cases, no obj in cells)
        no_obj_mask = ~targets_mask
        no_obj_preditions = predictions[..., no_obj_mask, :]
        no_obj_actual = label[..., no_obj_mask, :]
        no_obj_loss = self.mse(no_obj_preditions[..., 0], no_obj_actual[..., 0])

        loss = abox_loss * self.lambda_abox + self.lambda_coord * box_loss + obj_loss + self.lambda_no_obj * no_obj_loss + self.lambda_class_obj * class_obj_loss + self.lambda_class_color * class_color_loss

        return loss