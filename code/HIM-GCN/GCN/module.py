import tensorflow as tf

from GCN.layers import GraphConvolution, dot
from GCN.models import Model


class gcn_module(Model):
    def __init__(self, placeholders, hidden_dims=[50, 100], sparse_network=False, featureless=False,
                 data_type=None, multi_data=None, **kwargs):
        super(gcn_module, self).__init__(**kwargs)

        self.data_type = data_type

        self.placeholders = placeholders
        self.sparse_network = sparse_network
        self.featureless = featureless
        if self.data_type == "expr":
            # 取第0维度的数据
            self.inputs = placeholders['expr']
        if self.data_type == "mut":
            # 取第一维度数据
            self.inputs = placeholders['mut']
        if data_type == "cn":
            self.inputs = placeholders['cn']
        if data_type == "methy":
            self.inputs = placeholders['methy']
        elif self.data_type == "multi_data":
            # 取全部数据
            self.inputs = multi_data
        if featureless:
            self.input_dim = self.inputs.get_shape().as_list()[0]
        else:
            # 不用get_shape(), 不能输出第一维度
            self.input_dim = self.inputs.get_shape().as_list()[1]
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]

        self.hidden_dims = hidden_dims
        self.build()

    def build(self):
        # 重定义build()
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)

        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def _accuracy(self):
        pass

    def _loss(self):
        pass

    def _build(self):
        if self.data_type == "multi_data":
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.hidden_dims[1],
                                                # 100维
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                sparse_inputs=False,
                                                logging=self.logging,
                                                featureless=self.featureless))
            self.layers.append(GraphConvolution(input_dim=self.hidden_dims[1],
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=lambda x: x,
                                                dropout=True,
                                                # 第一层
                                                logging=self.logging,
                                                featureless=self.featureless))
        else:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.hidden_dims[0],
                                                # 20维
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                sparse_inputs=False,
                                                logging=self.logging,
                                                featureless=self.featureless))

    def predict(self):
        return self.outputs


















