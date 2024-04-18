import torch, torchvision, argparse, os

from dataset import RisikoDataset
from utils import Converter
from net import DetektorNet
from loss import RisikoLoss

class TrainingParams:
    epochs:int
    batch:int
    dataset_path:str
    weights_dir:str
    device:str
    trainset:RisikoDataset
    validationset:RisikoDataset
    testset:RisikoDataset


def argParser() -> TrainingParams:
    parser = argparse.ArgumentParser(prog='dataset_generator', epilog='Risko dataset generator', description='Generate a synthetic dataset for the risiko problem')
    parser.add_argument('-e', '--epochs', metavar='INT', type=int, required=True, help='Number of epochs')
    parser.add_argument('-b', '--batch', metavar='INT', type=int, required=True, help='Batch size')
    parser.add_argument('--weightsDir', metavar='PATH', type=str, help='Path to directory in which the weights are saved', default='weights')
    parser.add_argument('--datasetPath', metavar='STR', type=str, required=True, help='Path to the dataset. Inside the specified directory there must directories \'train\', \'val\' and \'test\' containing the respective datasets')

    args = parser.parse_args()

    assert args.epochs > 0
    assert args.batch > 0
    assert os.path.isdir(args.weightsDir)
    assert os.path.isdir(args.datasetPath)

    t = TrainingParams()
    t.epochs, t.batch, t.dataset_path, t.weights_dir = args.epochs, args.batch, args.datasetPath, args.weightsDir

    return t

def initParams() -> TrainingParams:
    p = argParser()
    p.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert os.path.isdir


def main():

    params = initParams()

    

    return 0

if __name__ == "__main__":
    main()