import argparse
import numpy as np
import sklearn.model_selection



def select_data_from_index(data, mask, index=None, pos_index=None, neg_index=None, balance=False):
    assert (data.shape[0] == mask.shape[0])
    if balance:
        data_mask = np.zeros_like(mask)
        data_mask[index] = 1
        data_selected = np.zeros_like(data)
        data_selected[index] = data[index]
    else:
        data_mask = np.zeros_like(mask)
        data_mask[pos_index] = 1
        data_mask[neg_index] = 1
        data_selected = np.zeros_like(data)
        data_selected[pos_index] = data[pos_index]
    return data_selected, data_mask


def construct_train_val_data(data, mask, validation_size):
    assert (data.shape[0] == mask.shape[0])
    mask_index = np.where(mask == 1)[0]
    train_index, val_index = sklearn.model_selection.train_test_split(mask_index,
                                                                      test_size=validation_size,
                                                                      stratify=data[mask == 1, 0]
                                                                      )
    data_train, mask_train = select_data_from_index(data, mask, train_index,balance =True)
    data_val, mask_val = select_data_from_index(data, mask, val_index, balance=True)
    return data_train, mask_train, data_val, mask_val


def construct_cross_validation_data(data, mask, folds, balance = False):
    if balance:
        label_idx = np.where(mask == 1)[0]  # get indices of labeled genes
        kf = sklearn.model_selection.StratifiedKFold(n_splits=folds, shuffle=True)
        splits = kf.split(label_idx, data[label_idx])
        k_sets = []
        for train, test in splits:
            # get the indices in the real y and mask realm
            train_idx = label_idx[train]
            test_idx = label_idx[test]
            # construct y and mask for the two sets
            data_train, train_mask = select_data_from_index(data, mask, train_idx, balance = True)
            data_test, test_mask = select_data_from_index(data, mask, test_idx, balance = True)
            k_sets.append((data_train, data_test, train_mask, test_mask))
    else:
        pos = data
        neg = mask
        k_sets = []
        pos_label_idx = np.where(pos == 1)[0]
        pos_kf = sklearn.model_selection.StratifiedKFold(n_splits=folds, shuffle=True)
        pos_splits = pos_kf.split(pos_label_idx, pos[pos_label_idx])
        pos_train_index = []
        pos_test_index = []

        for train, test in pos_splits:
            pos_train_idx = pos_label_idx[train]
            pos_test_idx = pos_label_idx[test]
            pos_train_index.append(pos_train_idx)
            pos_test_index.append(pos_test_idx)

        neg_label_idx = np.where(neg == 1)[0]
        neg_kf = sklearn.model_selection.StratifiedKFold(n_splits=folds, shuffle=True)
        neg_splits = neg_kf.split(neg_label_idx, neg[neg_label_idx])
        neg_train_index = []
        neg_test_index = []

        for train, test in neg_splits:
            neg_train_idx = neg_label_idx[train]
            neg_test_idx = neg_label_idx[test]
            neg_train_index.append(neg_train_idx)
            neg_test_index.append(neg_test_idx)

        for i in range(folds):
            data_train, train_mask = select_data_from_index(pos, neg, pos_index=pos_train_index[i], neg_index=neg_train_index[i], balance=False)
            data_test, test_mask = select_data_from_index(pos, neg, pos_index=pos_test_index[i], neg_index=neg_test_index[i], balance=False)
            k_sets.append((data_train, data_test, train_mask, test_mask))

    return k_sets


def parse_args():
    # for specific cancer type, it's better to set lower epoch number
    # pancancer 6000; brca 1500; gbm 2000; luad 1000
    parser = argparse.ArgumentParser(description='The params in training procedure')
    parser.add_argument('-e', '--epochs', help='Number of Epochs',
                        dest='epochs',
                        default=4000,
                        type=int
                        )
    parser.add_argument('-lr', '--learningrate', help='Learning Rate',
                        dest='lr',
                        default=.001,
                        type=float
                        )
    parser.add_argument('-s', '--support', help='Neighborhood Size in Convolutions',
                        dest='support',
                        default=1,
                        type=int
                        )
    parser.add_argument('-hd', '--hidden_dims',
                        help='Hidden Dimensions (number of filters per layer. Also determines the number of hidden layers.',
                        nargs='+',
                        dest='hidden_dims',
                        default=[50, 100])
    parser.add_argument('-wd', '--weight_l2', help='Weight Decay',
                        dest='l2_weight',
                        default=5e-2,
                        type=float
                        )
    parser.add_argument('-wl', '--weight_l1', help='Weight L1',
                        dest='l1_weight',
                        default=1e-4,
                        type=float
                        )
    parser.add_argument('-do', '--dropout', help='Dropout Percentage',
                        dest='dropout',
                        default=.5,
                        type=float
                        )
    parser.add_argument('-d', '--data', help='Path to HDF5 container with data',
                        dest='data',
                        type=str,
                        required=True
                        )
    parser.add_argument('-cv', '--cv_runs', help='Number of cross validation runs',
                        dest='cv_runs',
                        default=3,
                        type=int
                        )
    args = parser.parse_args()
    return args





