import os
import argparse
import random
from datetime import datetime

def arg_parser() -> tuple[ str, int, int]:
    # parse arguments
    parser = argparse.ArgumentParser(prog='shuffle_full_dataset', epilog='Risko dataset shuffler', description='Shuffle the full dataset before splitting it into train-val-test')
    parser.add_argument('dataset_path', type=str, help='Path to full dataset without differentiation between ')
    parser.add_argument('-s', '--seed', metavar='SEED', type=int, help='Set random seed')
    parser.add_argument('-i', '--iter', metavar='ITERATIONS', type=int, help='Number of \"Exchanges\" to be done')

    args = parser.parse_args()
    
    dataset_path: str = args.dataset_path
    if not os.path.isdir(dataset_path):
        print('Invalid path for the dataset')
        exit()

    # get random seed
    seed = int(round(datetime.now().timestamp()))
    if args.seed != None:
        if args.seed <= 0:
            print('Invalid seed value')
            exit()
        seed: int = args.seed

    iter_count = 12000
    if args.iter != None:
        iter_count = args.iter

    return [dataset_path, seed, iter_count]

def main():
    dataset_path, seed, iter_count = arg_parser()
    
    print(dataset_path)
    print(seed)
    print(iter_count)

    random.seed(seed)

    imgs_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")

    imgs = sorted( filter( lambda x: os.path.isfile(os.path.join(imgs_dir, x)), os.listdir(imgs_dir) ) )
    labels = sorted( filter( lambda x: os.path.isfile(os.path.join(labels_dir, x)), os.listdir(labels_dir) ) )

    dataset_size = len(imgs)
    assert dataset_size == len(labels)

    temp_img = os.path.join(imgs_dir, "temp.jpg")
    temp_label = os.path.join(labels_dir, "temp.txt")

    for i in range(iter_count):
        # pick 2 different random indices
        index1, index2 = random.randint(0,dataset_size-1), random.randint(0,dataset_size-1)
        while index1 == index2 : index2 = random.randint(0,dataset_size-1)

        os.rename(os.path.join(imgs_dir, imgs[index1]), temp_img)
        os.rename(os.path.join(imgs_dir, imgs[index2]), os.path.join(imgs_dir, imgs[index1]))
        os.rename(temp_img, os.path.join(imgs_dir, imgs[index2]))

        os.rename(os.path.join(labels_dir, labels[index1]), temp_label)
        os.rename(os.path.join(labels_dir, labels[index2]), os.path.join(labels_dir, labels[index1]))
        os.rename(temp_label, os.path.join(labels_dir, labels[index2]))


if __name__ == "__main__":
    main()

