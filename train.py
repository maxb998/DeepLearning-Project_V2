import torch, argparse, os
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from dataset import RisikoDataset
from utils import Converter
from net import DetektorNet
from loss import RisikoLoss

class TrainingParams:
    epochs:int
    batch:int
    lr:float
    dataset_path:str
    weights_path:str
    load_all_imgs:bool
    load_weights:bool


def argParser() -> TrainingParams:

    parser = argparse.ArgumentParser(prog='dataset_generator', epilog='Risko dataset generator', description='Generate a synthetic dataset for the risiko problem')
    parser.add_argument('-e', '--epochs', metavar='INT', type=int, required=True, help='Number of epochs')
    parser.add_argument('-b', '--batch', metavar='INT', type=int, required=True, help='Batch size')
    parser.add_argument('-lr', metavar='FLOAT', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weights_path', metavar='PATH', type=str, help='Path to directory in which the weights are saved', default='weights')
    parser.add_argument('--dataset_path', metavar='STR', type=str, required=True, help='Path to the dataset. Inside the specified directory there must directories \'train\', \'val\' and \'test\' containing the respective datasets')
    parser.add_argument('--load_all_imgs', action='store_true', help='Specify wether or not loading all images into memory')
    parser.add_argument('--new_weights', action='store_false', help='Specify wether or not loading existing weights if any')

    args = parser.parse_args()

    assert args.epochs > 0 and args.batch > 0 and args.lr > 0
    assert os.path.isdir(args.weights_path) and os.path.isdir(args.dataset_path)
    for d in ['train', 'val', 'test']:
        d = os.path.join(args.dataset_path, d)
        assert os.path.isdir(d) and os.path.isdir(os.path.join(d, 'images')) and os.path.isdir(os.path.join(d, 'labels'))

    t = TrainingParams()
    t.epochs = args.epochs
    t.batch = args.batch
    t.lr = torch.tensor(args.lr, dtype=float)
    t.dataset_path = args.dataset_path
    t.weights_path = args.weights_path
    t.load_all_imgs = args.load_all_imgs
    t.load_weights = not args.new_weights

    return t

def get_score(model:DetektorNet, dataset:RisikoDataset, mean_ap_obj:MeanAveragePrecision, device:str, prob_threshold:float=0.5, iou_threshold:float=0.7) -> float:

    preds, target = list(), list()
    mAP = 0

    pbar = tqdm(len(dataset), desc='\t Evaluating mAP on validation set', unit='sample', leave=True)
    pbar.set_postfix_str('mAP = ' + str(mAP))

    for i in range(len(dataset)):
        img_tensor, target_labels = dataset.__getitem__(i, return_basic_labels=True)
        img_tensor = img_tensor.unsqueeze(0)
        target_labels = target_labels.clone() # clone to prevent modifications on original

        preds_i, target_i = dict(), dict()

        netout = model.forward(img_tensor.to(device)).to('cpu')

        pred_labels = dataset.cv.convert_netout_to_labels(netout.squeeze(0), probability_threshold=prob_threshold, max_iou=iou_threshold, apply_nms=True)

        # get labels absolute values
        target_i['boxes'] = target_labels[:,1:] * dataset.cv.netin_img_shape
        target_i['labels'] = target_labels[:,0].type(torch.int32)

        preds_i['boxes'] = pred_labels[:,2:] * dataset.cv.netin_img_shape
        preds_i['scores'] = pred_labels[:,0]
        preds_i['labels'] = pred_labels[:,1].type(torch.int32)

        target.append(target_i)
        preds.append(preds_i)

        mAP += mean_ap_obj.forward([preds_i], [target_i])['map']

        pbar.set_postfix_str('mAP = ' + str(mAP.item() / (i+1)))
        pbar.update(1)

    return mAP.item()


def main():

    p = argParser()

    cv = Converter(netout_dtype=torch.float32)
    print('Loading training set...')
    trainset = RisikoDataset(dataset_dir=os.path.join(p.dataset_path, 'train'), cv=cv, train_mode=True, load_all_images_in_memory=p.load_all_imgs)
    print('Loading validation set...')
    validationset = RisikoDataset(dataset_dir=os.path.join(p.dataset_path, 'val'), cv=cv, train_mode=False, load_all_images_in_memory=p.load_all_imgs)
    print('Loading test set...')
    testset = RisikoDataset(dataset_dir=os.path.join(p.dataset_path, 'test'), cv=cv, train_mode=False, load_all_images_in_memory=p.load_all_imgs)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DetektorNet(cv.abox_count).to(device)
    loss_func = RisikoLoss(lambda_abox=0.1,
                      lambda_coord=1.,
                      lambda_no_obj=0.01,
                      lambda_class_obj=1,
                      lambda_class_color=0.1,
                      abox_count=cv.abox_count,
                      batch_size=p.batch)
    
    # load existing weights if any
    if p.load_weights:
        weights_fnames = os.listdir(p.weights_path).sort()
        for w_name in weights_fnames:
            if 'last' in w_name:
                model = torch.load(os.path.join(p.weights_path, w_name)).to(device)
    

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0)
    mean_ap_obj = MeanAveragePrecision(box_format='cxcywh', iou_type='bbox')
    mean_ap_obj.warn_on_many_detections = False

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=p.batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    best_map = 0
    loss_sum = int(0)

    for epoch in range(p.epochs):
        pbar = tqdm(train_loader, desc='Training Epoch ' + str(epoch), unit='batch', leave=True)
        pbar.set_postfix(loss=loss_sum/p.batch)

        model.train()
        
        loss_sum = int(0)
        for batch_idx, (x,y) in enumerate(pbar):

            optimizer.zero_grad()

            netout = model.forward(x.to(device)).to('cpu')

            loss = loss_func.forward(netout, y)
            loss_sum += loss.item()
            loss.backward()

            optimizer.step()

            pbar.set_postfix(loss=loss_sum/(batch_idx+1))

        pbar.close()

        model.eval()

        mAP = get_score(model=model, dataset=validationset, mean_ap_obj=mean_ap_obj, device=device, prob_threshold=0.5, iou_threshold=0.7)
        print('Current mAP = ' + str(mAP))
        if mAP > best_map:
            best_map = mAP
            torch.save(model, os.path.join(p.weights_path, 'best_model'))
        torch.save(model, os.path.join(p.weights_path, 'last_model'))

    return 0

if __name__ == "__main__":
    main()