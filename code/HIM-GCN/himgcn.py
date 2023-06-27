
import tensorflow as tf

from GCN.layers import GraphConvolution
from GCN.models import Model
from himgcn_module import himgcn_module


class HIMGCN():
    def __init__(self, placeholders, learning_rate=0.001, featureless=False, weight_l2=5e-4,
                 weight_l1=1e-4, pos_weight=10, neg_weight=1, logging=True, **kwargs):
        super(HIMGCN, self).__init__(**kwargs)

        self.placeholders = placeholders
        self.outputs = None

        # definite four models trained on specific omic-data
        self.himgcn_expr = None
        self.himgcn_mut = None
        self.himgcn_cn = None
        self.himgcn_methy = None
        # definite model trained on concatenated multiomics data
        self.himgcn_multi = None

        # definite adam optimizer
        self.featureless = featureless
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # parameters of loss function
        self.loss = 0
        self.opt_op = None
        self.weight_l2 = weight_l2
        self.weight_l1 = weight_l1
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

        self.logging = logging
        self.construct()

    @tf.function
    # definite the function of concatenate multiomics data
    def concat(self, a, b, c, d):
        return tf.concat([a, b, c, d], axis=1)

    def construct(self):
        self.himgcn_expr = himgcn_module(placeholders=self.placeholders, data_type='expr')
        expr = self.himgcn_expr.predict()
        self.himgcn_mut = himgcn_module(placeholders=self.placeholders, data_type='mut')
        mut = self.himgcn_mut.predict()
        self.himgcn_cn = himgcn_module(placeholders=self.placeholders, data_type='cn')
        cn = self.himgcn_cn.predict()
        self.himgcn_methy = himgcn_module(placeholders=self.placeholders, data_type='methy')
        methy = self.himgcn_methy.predict()
        multi_data = self.concat(expr, mut, cn, methy)
        self.himgcn_multi = himgcn_module(placeholders=self.placeholders, data_type='multi_data', multi_data=multi_data)
        multi = self.himgcn_multi.predict()

        self.outputs = multi
        self._loss()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        if self.outputs.shape[1] > 1:
            return tf.nn.softmax(self.outputs)
        else:
            return tf.nn.sigmoid(self.outputs)

    def performance(self):
        with tf.variable_scope("evaluation"):
            pred = self.predict()
            acc_pre, acc = tf.metrics.accuracy(labels=self.placeholders['labels'],
                                         predictions=tf.greater(pred, 0.5),
                                         weights=self.placeholders['labels_mask']
                                         )
            auroc_pre, auroc = tf.metrics.auc(labels=self.placeholders['labels'],
                                      predictions=pred,
                                      weights=self.placeholders['labels_mask'],
                                      curve='ROC'
                                      )
            aupr_pre, aupr = tf.metrics.auc(labels=self.placeholders['labels'],
                                     predictions=pred,
                                     weights=self.placeholders['labels_mask'],
                                     curve='PR',
                                     summation_method='careful_interpolation'
                                     )
            if self.logging:
                    tf.summary.scalar('LOSS', self.loss)
                    tf.summary.scalar('ACC', acc)
                    tf.summary.scalar('AUPR', aupr)
                    tf.summary.scalar('AUROC',auroc)
        return self.loss, acc, aupr, auroc

    def _loss(self):
        # l1 and l2 regularization
        for var in self.himgcn_expr.layers[0].vars.values():
            self.loss += self.weight_l2 * tf.nn.l2_loss(var)+tf.keras.regularizers.l1(l=self.weight_l1)(var)
        for var in self.himgcn_mut.layers[0].vars.values():
            self.loss += self.weight_l2 * tf.nn.l2_loss(var)+tf.keras.regularizers.l1(l=self.weight_l1)(var)
        for var in self.himgcn_cn.layers[0].vars.values():
            self.loss += self.weight_l2 * tf.nn.l2_loss(var)+tf.keras.regularizers.l1(l=self.weight_l1)(var)
        for var in self.himgcn_methy.layers[0].vars.values():
            self.loss += self.weight_l2 * tf.nn.l2_loss(var)+tf.keras.regularizers.l1(l=self.weight_l1)(var)

        self.loss += self.masked_weighted_focal_loss(self.outputs,
                                                     self.placeholders['labels'],
                                                     self.placeholders['labels_mask'])

    def masked_weighted_focal_loss(self, logit,label, mask, alpha=0.25, r=2):
        if logit.shape[1] > 1:
            fl = tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=label)
        else:
            # definite the weighted focal loss to solve the data sample unbalance
            prediction = tf.nn.sigmoid(logit)
            prediction_fl = tf.where(tf.equal(label, 1), prediction, 1.-prediction)
            logits = tf.cast(logit, tf.float32)
            labels = tf.cast(label, tf.float32)
            cross_en = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

            # calculate the alpha and alpha weight for focal loss
            alpha_fl = tf.scalar_mul(alpha, tf.ones_like(label, dtype=tf.float32))
            alpha_weight = tf.where(tf.equal(labels, 1.0), self.pos_weight * alpha_fl, self.neg_weight * (1 - alpha_fl))
            fl = cross_en * tf.pow(1 - prediction_fl, r) * alpha_weight

        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        fl *= mask
        return tf.reduce_mean(fl)

    def _accuracy(self):
        pass

    def save(self, path, sess=None):
        if not sess:
            raise AttributeError("attribute error.")
        saver = tf.train.Saver()
        save_path = saver.save(sess, path)
        print("Model saved in file: %s" % save_path)

    def load(self, path, sess=None):
        if not sess:
            raise AttributeError("attribute error.")
        saver = tf.train.Saver(self.vars)
        saver.restore(sess, path)
