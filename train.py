import torch, argparse, os
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from dataset import RisikoDataset
from utils import Converter
from net import GridNet, GridNetBackbone, GridNetHead
from loss import RisikoLoss

class TrainingParams:
    epochs:int
    batch:int
    lr:float
    model_size:str
    dataset_path:str
    weights_path:str
    load_weights:bool


def argParser() -> TrainingParams:

    parser = argparse.ArgumentParser(prog='Training Script', epilog='DetektorNet training script', description='Trains DetektorNet')
    parser.add_argument('-e', '--epochs', metavar='INT', type=int, required=True, help='Number of epochs')
    parser.add_argument('-b', '--batch', metavar='INT', type=int, required=True, help='Batch size')
    parser.add_argument('-lr', metavar='FLOAT', type=float, required=True, help='Learning rate')
    parser.add_argument('--model_size', metavar='STR', type=str, required=True, help='Size of the model specified as \'M\', \'L\' or a supported combination like \'M-XXL\' or \'L-XL\'')
    parser.add_argument('--weights_path', metavar='PATH', type=str, help='Path to directory in which the weights are saved', default='weights')
    parser.add_argument('--dataset_path', metavar='STR', type=str, required=True, help='Path to the dataset. Inside the specified directory there must directories \'train\', \'val\' and \'test\' containing the respective datasets')
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
    t.model_size = args.model_size
    t.dataset_path = args.dataset_path
    t.weights_path = args.weights_path
    t.load_weights = args.new_weights

    return t

def get_score(model:GridNet, dataset:RisikoDataset, mean_ap_obj:MeanAveragePrecision, device:str='cpu', prob_threshold:float=0.5, iou_threshold:float=0.75) -> float:

    mAP = 0
    map_50 = 0
    mAP_75 = 0
    detections = 0
    effective_objs = 0

    assert dataset.is_trainset == False

    pbar = tqdm(len(dataset), desc='\t Evaluating mAP on validation set', unit='sample', leave=True)
    pbar.set_postfix(mAP=0, map_50=0, mAP_75=0, avg_detect=0, avg_objs=0)

    for i in range(len(dataset)):
        img_tensor, target_labels = dataset.__getitem__(i)
        img_tensor = img_tensor.unsqueeze(0)
        target_labels = target_labels.clone() # clone to prevent modifications on original
        effective_objs += target_labels.shape[0]

        preds_i, target_i = dict(), dict()

        netout = model.forward(img_tensor.to(device)).to('cpu')

        pred_labels = dataset.cv.convert_netout_to_labels(netout.squeeze(0), probability_threshold=prob_threshold, max_iou=iou_threshold)
        dataset.cv.convert_labels_from_relative_to_absolute_values(pred_labels[...,1:])
        dataset.cv.convert_labels_from_relative_to_absolute_values(target_labels)
        detections += pred_labels.shape[0]

        # get labels absolute values
        target_i['boxes'] = target_labels[:,1:]
        target_i['labels'] = target_labels[:,0].type(torch.int32)

        preds_i['boxes'] = pred_labels[:,2:]
        preds_i['scores'] = pred_labels[:,0]
        preds_i['labels'] = pred_labels[:,1].type(torch.int32)

        mAP_dict = mean_ap_obj.forward([preds_i], [target_i])
        mAP += mAP_dict['map']
        map_50 += mAP_dict['map_50']
        mAP_75 += mAP_dict['map_75']

        pbar.set_postfix(mAP=mAP.item()/(i+1), map_50=map_50.item()/(i+1), mAP_75=mAP_75.item()/(i+1), avg_detect=detections/(i+1), avg_objs=effective_objs/(i+1))
        pbar.update(1)

    return mAP.item()/len(dataset)


def main():

    p = argParser()

    sizes = p.model_size.split('-')
    assert sizes[0] in GridNetBackbone.ch.keys()
    assert sizes[-1] in GridNetHead.ch.keys()

    weights_fnames = os.listdir(p.weights_path)
    best_map = float(0)
    for w_name in weights_fnames:
        if 'best_' + p.model_size in w_name:
            best_map = float(w_name.split('_')[-1])

    cv = Converter(netout_dtype=torch.float32)
    print('Loading training set...')
    trainset = RisikoDataset(os.path.join(p.dataset_path, 'train'), cv, is_trainset=True)
    print('Loading validation set...')
    validationset = RisikoDataset(os.path.join(p.dataset_path, 'val'), cv)
    print('Loading test set...')
    testset = RisikoDataset(os.path.join(p.dataset_path, 'test'), cv)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GridNet(cv.abox_count, p.model_size)
    loss_func = RisikoLoss(lambda_abox=0.1,
                      lambda_coord=1.,
                      lambda_no_obj=0.4,
                      lambda_class_obj=1,
                      lambda_class_color=0.1,
                      abox_count=cv.abox_count,
                      batch_size=p.batch)
    
    # load existing weights if any
    if p.load_weights:
        weights_fnames = os.listdir(p.weights_path)
        for w_name in weights_fnames:
            if 'best_' + p.model_size in w_name:
                print('loading previous weights: ' + w_name)
                model = torch.load(os.path.join(p.weights_path, w_name))
                break
    
    model.to(device)    

    adam = torch.optim.Adam(model.parameters(), weight_decay=0, lr=p.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=adam, gamma=0.95)
    mean_ap_obj = MeanAveragePrecision(box_format='cxcywh', iou_type='bbox')
    mean_ap_obj.warn_on_many_detections = False

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=p.batch, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    loss_sum = int(0)

    for epoch in range(p.epochs):
        pbar = tqdm(train_loader, desc='Epoch ' + str(epoch), unit='batch', leave=True)
        pbar.set_postfix(loss=loss_sum/p.batch)

        model.train()
        
        loss_sum = int(0)
        for batch_idx, (x,y) in enumerate(pbar):

            adam.zero_grad()

            netout = model.forward(x.to(device)).to('cpu')

            loss = loss_func.forward(netout, y)
            loss_sum += loss.item() / p.batch
            loss.backward()

            adam.step()

            pbar.set_postfix(loss=loss_sum/(batch_idx+1), lr=scheduler.get_last_lr()[0].item())

        pbar.close()
        scheduler.step()

        model.eval()

        mAP = get_score(model, validationset, mean_ap_obj, device)
        if mAP > best_map:
            best_map = mAP
            weights_fnames = os.listdir(p.weights_path)
            for w_name in weights_fnames:
                if 'best_' + p.model_size in w_name:
                    os.remove(os.path.join(p.weights_path, w_name))
            torch.save(model, os.path.join(p.weights_path, 'best_' + p.model_size + '_' + 'e' + str(epoch) + '_' + str(mAP).replace('.', '-')))

        weights_fnames = os.listdir(p.weights_path)
        for w_name in weights_fnames:
            if 'last_' + p.model_size in w_name:
                os.remove(os.path.join(p.weights_path, w_name))
        torch.save(model, os.path.join(p.weights_path, 'last_' + p.model_size + '_' + 'e' + str(epoch) + '_' + str(mAP).replace('.', '-')))
    
    print()
    print('Computing score on testset using latest weights')
    print('Score on testset is: ' + str(get_score(model, testset, mean_ap_obj, device)))

    weights_fnames = os.listdir(p.weights_path)
    for w_name in weights_fnames:
        if 'best_' + p.model_size in w_name:
            model = torch.load(os.path.join(p.weights_path, w_name))
            print()
            print('Computing score on testset using best weights: ' + w_name)
            print('Score on testset is: ' + str(get_score(model, testset, mean_ap_obj, device)))

    

    return 0

if __name__ == "__main__":
    main()