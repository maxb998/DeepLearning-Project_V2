import argparse, os
import numpy as np


def argParser() -> tuple[str, np.ndarray, str]:
    parser = argparse.ArgumentParser(prog='dataset_splitter', epilog='Dataset splitter', description='Split dataset specified in original directory into destination directory as train, val and test')
    parser.add_argument('-d', '--dataset_path', metavar='STR', type=str, required=True, help='Path to the unseparated dataset')
    parser.add_argument('--train', metavar='FLOAT', type=float, default=0.6, help='Percentage of elements in the original dataset that go into the training set')
    parser.add_argument('--val', metavar='FLOAT', type=float, default=0.2, help='Percentage of elements in the original dataset that go into the validation set')
    parser.add_argument('--test', metavar='FLOAT', type=float, default=0.2, help='Percentage of elements in the original dataset that go into the test set')
    parser.add_argument('--labels_extension', metavar='STR', type=str, default='.csv', help='Extension of the labels files')

    args = parser.parse_args()

    assert args.train >= 0 and args.val >= 0 and args.test >= 0
    assert os.path.isdir(args.dataset_path)

    split_vars = np.array((args.train, args.val, args.test), dtype=float)
    np.divide(split_vars, split_vars.sum(), out=split_vars)

    return (args.dataset_path, split_vars, args.labels_extension)


def main():
    d_path, split_vars, lbl_ext = argParser()
    src_images_dir, src_labels_dir = os.path.join(d_path, 'images'), os.path.join(d_path, 'labels')

    img_paths = os.listdir(src_images_dir)
    n = len(img_paths)
    
    shuffler = np.random.permutation(n)
    path_names = ('train', 'val', 'test')
    split_vars = np.multiply(split_vars, n).round().astype(int)

    for i in range(3):
        dest_dir = os.path.join(d_path, path_names[i])
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        
        dest_images_dir, dest_labels_dir = os.path.join(dest_dir, 'images'), os.path.join(dest_dir, 'labels')

        if not os.path.isdir(dest_images_dir):
            os.mkdir(dest_images_dir)
        if not os.path.isdir(dest_labels_dir):
            os.mkdir(dest_labels_dir)

        start_idx, end_idx = split_vars[:i].sum(), split_vars[:i+1].sum()
        
        print('Moving ' + str(end_idx-start_idx) + ' samples to ' + path_names[i] + ' set')
        
        for j in shuffler[start_idx:end_idx]:
            label_j_name = os.path.splitext(img_paths[j])[0] + lbl_ext
            src_label_j_path = os.path.join(src_labels_dir, label_j_name)

            if os.path.isfile(src_label_j_path):
                dest_label_j_path = os.path.join(dest_labels_dir, label_j_name)
                os.rename(src_label_j_path, dest_label_j_path) # move label file
            else: # label file missing
                print('WARNING: label file \"' + src_label_j_path + '\" does not exist')

            src_image_j_path = os.path.join(src_images_dir, img_paths[j])
            dest_image_j_path = os.path.join(dest_images_dir, img_paths[j])
            os.rename(src_image_j_path, dest_image_j_path) # move image

    return 0

if __name__ == "__main__":
    main()