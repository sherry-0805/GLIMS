
import numpy as np
import os, sys
import tensorflow as tf

import himgcn_io, preprocessing, utils
from himgcn import HIMGCN


os.environ['CUDA_VISIBLE_DEVICES'] = '6'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7

with tf.Session(config=config) as sess:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


save_after_training = True


def predict(sess, model, expr, mut, cn, methy, support, labels, labels_mask, placeholders):
    feed_dict = utils.construct_feed_dict(expr, mut, cn, methy, support, labels, labels_mask, placeholders)
    prediction = sess.run(model.predict(), feed_dict=feed_dict )
    return prediction


def fit_model(model, sess, expr, mut, cn, methy, support, placeholders,
              epochs, dropout, y_train, mask_train, y_val, mask_val, path):
    model_path = os.path.join(path, 'model.ckpt')
    model_performance = model.performance()
    running_paras = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="evaluation")
    paras_initialize = tf.variables_initializer(var_list=running_paras)

    merged = tf.summary.merge_all()

    # initialize parameters
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    for epoch in range(epochs):
        feed_dict = utils.construct_feed_dict(expr, mut, cn, methy, support, y_train, mask_train, placeholders)
        feed_dict.update({placeholders['dropout']: dropout})
        _ = sess.run(model.opt_op, feed_dict=feed_dict)
        train_loss, train_acc, train_aupr, train_auroc = sess.run(model_performance, feed_dict=feed_dict)

        if model.logging:
            print("Epoch:", '%04d' % (epoch + 1),
                    "Train Loss=", "{:.5f}".format(train_loss),
                    "Train Acc=", "{:.5f}".format(train_acc),
                    "Train AUROC={:.5f}".format(train_auroc),
                    "Train AUPR: {:.5f}".format(train_aupr))

        if epoch % 10 == 0 or epoch-1 == epochs:
            if model.logging:
                d = utils.construct_feed_dict(expr,mut,cn,methy, support, y_val,
                                              mask_val, placeholders)
                sess.run(paras_initialize)
                val_loss, val_acc, val_aupr, val_auroc = sess.run(model_performance, feed_dict=d)
                print("Epoch:", '%04d' % (epoch + 1),
                      "Test Loss=", "{:.5f}".format(val_loss),
                      "Test Acc=", "{:.5f}".format(val_acc),
                      "Test AUROC={:.5f}".format(val_auroc),
                      "Test AUPR: {:.5f}".format(val_aupr))

    print('The total epoch is', epochs, ', optimization finished.')

    if save_after_training:
        print("Save model to {}".format(model_path))
        model.save(model_path, sess=sess)
    else:
        model = HIMGCN(placeholders=placeholders,
                    learning_rate=0.001,
                    featureless=False,
                    logging=True)
        model.load(model_path, sess=sess)

    return model


def single_cv(sess, support, num_supports, expr, mut, cn, methy,y_train, y_test, mask_train,
              mask_test, gene, args, pos_weight, neg_weight, path):
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32, name='support_{}'.format(i)) for i in range(num_supports)],
        'expr': tf.placeholder(tf.float32, shape=expr.shape, name='expr'),
        'mut': tf.placeholder(tf.float32, shape=mut.shape, name='mutr'),
        'cn': tf.placeholder(tf.float32, shape=cn.shape, name='cn'),
        'methy': tf.placeholder(tf.float32, shape=methy.shape, name='methy'),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1]), name='Labels'),
        'labels_mask': tf.placeholder(tf.int32, shape=mask_train.shape, name='LabelsMask'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='Dropout'),
        'num_features_nonzero': tf.placeholder(tf.int32, shape=())
    }

    model = HIMGCN(placeholders=placeholders,
                 learning_rate=args['lr'],
                 weight_l2=args['l2_weight'],
                 weight_l1=args['l1_weight'],
                 pos_weight=pos_weight,
                 neg_weight=neg_weight,
                 logging=True)

    model = fit_model(model, sess, expr, mut, cn, methy, support, placeholders, args['epochs'],
                      args['dropout'], y_train, mask_train, y_test, mask_test, path)

    model_performance = model.performance()
    sess.run(tf.local_variables_initializer())
    f_dict = utils.construct_feed_dict(expr, mut, cn, methy, support, y_test, mask_test, placeholders)
    test_performance = sess.run(model_performance, feed_dict=f_dict)
    print("Validataion/test set results:",
          "loss=", "{:.5f}".format(test_performance[0]),
          "accuracy=", "{:.5f}".format(
            test_performance[1]), "aupr=", "{:.5f}".format(test_performance[2]),
          "auroc=", "{:.5f}".format(test_performance[3]))

    # predict
    predictions = predict(sess, model, expr, mut, cn, methy, support, y_test, mask_test, placeholders)

    himgcn_io.write_prediction_result(path, gene, predictions)
    print(np.shape(y_train), np.shape(y_test), np.shape(mask_train), np.shape(mask_test), np.shape(predictions))
    himgcn_io.write_prediction_label(path, gene, predictions, y_train, y_test, mask_train, mask_test)

    return test_performance


def cross_validation(adj, gene, expr, mut, cn, methy, pos, neg, args, path):
    support, num_supports = utils.get_support_matrices(adj, args['support'])

    pos_number = pos.sum()
    neg_number = neg.sum()
    pos_weight = (pos_number + neg_number) / pos_number
    neg_weight = (pos_number + neg_number) / neg_number

    k_sets = preprocessing.construct_cross_validation_data(pos, neg, folds=args['cv_runs'], balance=False)

    performance_measures = []
    for cv in range(args['cv_runs']):
        model_path = os.path.join(path, 'cv_{}'.format(cv))
        train_y, test_y, train_mask, test_mask, = k_sets[cv]
        print(train_y.sum(), test_y.sum(), train_mask.sum(), test_mask.sum())
        with tf.Session() as sess:
            cv_performance = single_cv(sess, support, num_supports, expr, mut, cn, methy, train_y, test_y, train_mask,
                                       test_mask, gene, args, pos_weight, neg_weight, model_path)
            performance_measures.append(cv_performance)
        tf.reset_default_graph()

    data_rel_to_model = os.path.relpath(args['data'], path)
    args['data'] = data_rel_to_model
    himgcn_io.write_hypers(args, args['data'], os.path.join(path, 'hyperparamters.txt'))
    himgcn_io.write_final_prediction(path, args['cv_runs'])
    print(performance_measures)
    return performance_measures


args = preprocessing.parse_args()
path = himgcn_io.create_model_path()
data_input = himgcn_io.load_data_from_container(args.data)
network, gene, expr, mut, cn, methy, pos, neg = data_input
print('Load data container from', format(args.data))
args_dict = vars(args)
print(args_dict)
cross_validation(network, gene, expr, mut, cn, methy, pos, neg, args_dict, path)


