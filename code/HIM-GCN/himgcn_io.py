import h5py
import os
import pandas as pd
import time



def load_data_from_container(data_path, load_without=None):
    with h5py.File(data_path, 'r') as data_container:
        network = data_container['network'][:]
        nodes = data_container['gene_name'][:]
        pos = data_container['pos'][:]
        neg = data_container['neg'][:]
        if load_without == 'expr':
            pass
        else:
            expr = data_container['expr'][:]
        if load_without == 'mut':
            pass
        else:
            mut = data_container['mut'][:]
        if load_without == 'cn':
            pass
        else:
            cn = data_container['cn'][:]
        if load_without == 'methy':
            pass
        else:
            methy = data_container['methy'][:]

        return network, nodes, expr, mut, cn, methy, pos, neg


def create_model_path():
    model_path = './code/HIM-GCN/model/training'
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
        print('Create model path')
    t = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    os.mkdir(os.path.join(model_path,t))
    dir = os.path.join(model_path,t)

    return dir


def str_to_num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def write_hypers(hypers, input_file, path):
    with open(path,'w') as f:
        for hyper in hypers:
            f.write('{}\t{}\n'.format(hyper, hypers[hyper]))
        f.write('{}\n'.format(input_file))
    print("Save the hyperparameters to {}".format(path))


def load_hypers(path):
    with open(path, 'r') as f:
        hypers = {}
        for line in f.readlines():
            if '\t' in line:
                key, value = line.split('\t')
                if value.startswith('['):  # list of hidden dimensions
                    f = lambda x: "".join(c for c in x if c not in ['\"', '\'', ' ', '\n', '[', ']'])
                    l = [int(f(i)) for i in value.split(',')]
                    hypers[key.strip()] = l
                else:
                    hypers[key.strip()] = str_to_num(value.strip())
            else:
                input = line.strip()
    return hypers, input


def write_prediction_result(path, nodes, predict_results):
    with open(os.path.join(path, 'predict_results.tsv'), 'w') as f:
        f.write('Ensembl_ID\tGeneName\tPositive\n')
        for index in range(predict_results.shape[0]):
            f.write('{}\t{}\t{}\n'.format(nodes[index, 0], nodes[index, 1], predict_results[index, 0]))


def write_prediction_label(path, nodes, predict_results, y_train, y_test, mask_train,mask_test):
    with open(os.path.join(path, 'predict_label.tsv'), 'w') as f:
        f.write('Ensembl_ID\tGeneName\tPositive\ty_train\ty_test\tmask_train\tmask_test\n')
        for index in range(predict_results.shape[0]):
            f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(nodes[index, 0],
                                                          nodes[index, 1],
                                                          predict_results[index, 0],
                                                          y_train[index, 0],
                                                          y_test[index, 0],
                                                          mask_train[index, 0],
                                                          mask_test[index, 0]))
    pass


def write_final_prediction(path, args):
    for cv in range(args):
        model_path = os.path.join(path, 'cv_{}'.format(cv))
        pred_path = os.path.join(model_path, 'predict_results.tsv')
        #print(pred_path)
        if cv == 0:
            df = pd.read_csv(pred_path, sep='\t', header=0)
        else:
            cv_res = pd.read_csv(pred_path, sep='\t', header=0)
            df = pd.merge(df, cv_res, on=['Ensembl_ID', 'GeneName'], how='left')
    df.insert(loc=2, column='average', value=[0] * (len(df)))
    df.loc[:, 'average'] = df.iloc[:, 3:3 + args].mean(axis=1)
    df = df.iloc[:, 0:3]
    df = df.rename(columns={'average': 'Positive'})
    final_pred_path = os.path.join(path, 'predict_results.tsv')
    df.to_csv(final_pred_path, sep='\t')




