import pdb
import threading
import numpy as np
import tensorflow as tf
import tensorflow.keras as tf_keras
import tensorflow.keras.models as tf_models
import tensorflow.keras.layers as tf_layers
import tensorflow.keras.regularizers as tf_regularizers
import tensorflow.keras.initializers as tf_initializers
import tensorflow.keras.activations as tf_activations
from tensorflow.keras.optimizers import Adam, Adagrad
from keras import metrics

from made.made_object import *
from misc.utils import *
from modules.layers import *

class NetModule:
    """ This module manages model networks and parameters
    Saves and loads all states whenever client is switched.
    Created by:
        Wonyong Jeong (wyjeong@kaist.ac.kr)
    """
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.initializer = tf_initializers.VarianceScaling(seed=args.seed)
        self.curr_task = 0
        self.state = {}
        self.models = []
        self.heads = []
        self.decomposed_layers = {}
        self.initial_body_weights = []
        self.initial_heads_weights = []

        self.lid = 0
        self.adaptive_factor = 3
        if self.args.base_network == 'lenet':
            self.input_shape = (32,32,3)
        elif self.args.base_network == 'made':
            self.input_shape = 784 # input pixels

        if self.args.base_network == 'lenet':
            self.shapes = [
                (5,5,3,20),
                (5,5,20,50),
                (3200,800),
                (800,500)]

        elif self.args.base_network == 'made':
            layers = [self.input_shape]
            [layers.append(hidden_layer) for hidden_layer in self.args.hidden_layers]
            layers.append(self.input_shape)
            self.shapes = []
            [self.shapes.append((layers[i], layers[i+1])) for i in range(0, len(layers)-1)]

        if self.args.model in ['fedweit']:
            self.decomposed_variables = {
                'shared': [],
                'adaptive':{},
                'mask':{},
                'bias':{},
            }
            if self.args.model == 'fedweit':
                self.decomposed_variables['atten'] = {}
                self.decomposed_variables['from_kb'] = {}

    def init_state(self, cid):
        if self.args.model in ['fedweit']:
            self.state = {
                'client_id':  cid,
                'decomposed_weights': {
                    'shared': [],
                    'adaptive':{},
                    'mask':{},
                    'bias':{},
                },
                'heads_weights': self.initial_heads_weights,
            }
            if self.args.model == 'fedweit':
                self.state['decomposed_weights']['atten'] = {}
                self.state['decomposed_weights']['from_kb'] = {}
        else:
            self.state = {
                'client_id':  cid,
                'body_weights': self.initial_body_weights,
                'heads_weights': self.initial_heads_weights,
            }

    def save_state(self):
        self.state['heads_weights'] = []
        for h in self.heads:
            self.state['heads_weights'].append(h.get_weights())
        if self.args.model in ['fedweit']:
            for var_type, layers in self.decomposed_variables.items():
                self.state['decomposed_weights'] = {
                    'shared': [layer.numpy() for layer in self.decomposed_variables['shared']],
                    'adaptive':{tid: [layer.numpy() for lid, layer in self.decomposed_variables['adaptive'][tid].items()] for tid in self.decomposed_variables['adaptive'].keys()},
                    'mask':{tid: [layer.numpy() for lid, layer in self.decomposed_variables['mask'][tid].items()] for tid in self.decomposed_variables['mask'].keys()},
                    'bias':{tid: [layer.numpy() for lid, layer in self.decomposed_variables['bias'][tid].items()] for tid in self.decomposed_variables['bias'].keys()},
                }
                if self.args.model == 'fedweit':
                    self.state['decomposed_weights']['from_kb'] = {tid: [layer.numpy() for lid, layer in self.decomposed_variables['from_kb'][tid].items()] for tid in self.decomposed_variables['from_kb'].keys()}
                    self.state['decomposed_weights']['atten'] = {tid: [layer.numpy() for lid, layer in self.decomposed_variables['atten'][tid].items()] for tid in self.decomposed_variables['atten'].keys()}
        else:
            self.state['body_weights'] = self.model_body.get_weights()

        np_save(self.args.state_dir, '{}_net.npy'.format(self.state['client_id']), self.state)

    def load_state(self, cid):
        self.state = np_load(os.path.join(self.args.state_dir, '{}_net.npy'.format(cid))).item()

        for i, h in enumerate(self.state['heads_weights']):
                self.heads[i].set_weights(h)

        if self.args.model in ['fedweit']:
            for var_type, values in self.state['decomposed_weights'].items():
                if var_type == 'shared':
                    for lid, weights in enumerate(values):
                        self.decomposed_variables['shared'][lid].assign(weights)
                else:
                    for tid, layers in values.items():
                        for lid, weights in enumerate(layers):
                            self.decomposed_variables[var_type][tid][lid].assign(weights)
        else:
            self.model_body.set_weights(self.state['body_weights'])

    def init_global_weights(self):
        if self.args.base_network == 'made':
            global_weights = {}
            if self.args.only_federated == True:
                zeros_initializer = tf.keras.initializers.Zeros()
                global_weights['W'] = []
                for i in range(len(self.shapes)):
                    global_weights['W'].append(self.initializer(self.shapes[i]).numpy())
                global_weights['bias'] = []
                for i in range(len(self.shapes)):
                    global_weights['bias'].append(zeros_initializer(self.shapes[i][1]).numpy()) #just take output dims for use_bias
                if self.args.connectivity_weights:
                    global_weights['U'] = []
                    for i in range(len(self.shapes)):
                        global_weights['U'].append(self.initializer(self.shapes[i]).numpy())
                if self.args.direct_input:
                    global_weights['D'] = [self.initializer([self.input_shape, self.shapes[-1][1]]).numpy()]
            else:
                global_weights["W_global"] = []
                if self.args.connectivity_weights:
                    global_weights["U_global"] = []
                if self.args.direct_input:
                    global_weights["D_global"] = []
                    global_weights["D_global"].append(self.initializer((self.input_shape, self.shapes[-1][1])).numpy())
                for lid in range(len(self.shapes)):
                    global_weights["W_global"].append(self.initializer(self.shapes[lid]).numpy())
                    if self.args.connectivity_weights:
                        global_weights["U_global"].append(self.initializer(self.shapes[lid]).numpy())

        elif self.args.model in ['fedweit']:
            global_weights = []
            for i in range(len(self.shapes)):
                global_weights.append(self.initializer(self.shapes[i]).numpy())
        else:
            if self.args.base_network == 'lenet':
                body = self.build_lenet_body(decomposed=False)
            global_weights = body.get_weights()
        return global_weights

    def set_weights(self, weights, client_id=None, type = "global"):
        self.models[f"{client_id}"].set_weights(weights, direct_input = self.args.direct_input, type = type)

    def get_weights(self, client_id = None, type = None):
        return self.models[f"{client_id}"].get_weights(type)

    def init_decomposed_variables(self, initial_weights):
        self.decomposed_variables['shared'] = [tf.Variable(initial_weights[i],
                name='layer_{}/sw'.format(i)) for i in range(len(self.shapes))]
        for tid in range(self.args.num_tasks):
            for lid in range(len(self.shapes)):
                var_types = ['adaptive', 'bias', 'mask'] if self.args.model == 'apd' else ['adaptive', 'bias', 'mask', 'atten', 'from_kb']
                for var_type in var_types:
                    self.create_variable(var_type, lid, tid)

    def create_variable(self, var_type, lid, tid=None):
        trainable = True
        if tid not in self.decomposed_variables[var_type]:
            self.decomposed_variables[var_type][tid] = {}
        if var_type == 'adaptive':
            init_value = self.decomposed_variables['shared'][lid].numpy()/self.adaptive_factor
        elif var_type == 'atten':
            shape = (int(round(self.args.num_clients*self.args.frac_clients)),)
            if tid == 0:
                trainable = False
                init_value = np.zeros(shape).astype(np.float32)
            else:
                init_value = self.initializer(shape)
        elif var_type == 'from_kb':
            shape = np.concatenate([self.shapes[lid], [int(round(self.args.num_clients*self.args.frac_clients))]], axis=0)
            trainable = False
            if tid == 0:
                init_value = np.zeros(shape).astype(np.float32)
            else:
                init_value = self.initializer(shape)
        else:
            init_value = self.initializer((self.shapes[lid][-1], ))
        var = tf.Variable(init_value, trainable=trainable, name='layer_{}/task_{}/{}'.format(lid, tid, var_type))
        self.decomposed_variables[var_type][tid][lid] = var

    def get_variable(self, var_type, lid, tid=None):
        if var_type == 'shared':
            return self.decomposed_variables[var_type][lid]
        else:
            return self.decomposed_variables[var_type][tid][lid]

    def generate_mask(self, mask):
        return tf_activations.sigmoid(mask)

    def get_model_by_tid(self, tid = None, client_id = None):
        if self.args.model in ['fedweit']:
            if self.args.base_network == "made":
                return self.models[f"{client_id}"]
            else:
                self.switch_model_params(tid)
                return self.models[tid]

    def get_trainable_variables(self, curr_task, client_id = None, head=True):
        if self.args.model in ['fedweit']:
            if self.args.base_network == 'made':
                return self.models[f"{client_id}"][curr_task].get_weights()
            else:
                return self.get_decomposed_trainaible_variables(curr_task, retroactive=False, head=head)
        else:
            if head:
                return self.models[curr_task].trainable_variables
            else:
                return self.model_body.trainable_variables

    def get_decomposed_trainaible_variables(self, curr_task, retroactive=False, head=True):

        prev_variables = ['mask', 'bias', 'adaptive'] if self.args.model == 'apd' else ['mask', 'bias', 'adaptive', 'atten']
        trainable_variables = [sw for sw in self.decomposed_variables['shared']]
        if retroactive:
            for tid in range(curr_task+1):
                for lid in range(len(self.shapes)):
                    for pvar in prev_variables:
                        if pvar == 'bias' and tid < curr_task:
                            continue
                        if pvar == 'atten' and tid == 0:
                            continue
                        trainable_variables.append(self.get_variable(pvar, lid, tid))
        else:
            for lid in range(len(self.shapes)):
                for pvar in prev_variables:
                    if pvar == 'atten' and curr_task == 0:
                        continue
                    trainable_variables.append(self.get_variable(pvar, lid, curr_task))
        if head:
            head = self.heads[curr_task]
            trainable_variables.append(head.trainable_weights[0])
            trainable_variables.append(head.trainable_weights[1])
        return trainable_variables

    def get_body_weights(self, task_id=None):
        if self.args.model in ['fedweit']:
            prev_weights = {}
            for lid in range(len(self.shapes)):
                prev_weights[lid] = {}
                sw = self.get_variable(var_type='shared', lid=lid).numpy()
                for tid in range(task_id):
                    prev_aw = self.get_variable(var_type='adaptive', lid=lid, tid=tid).numpy()
                    prev_mask = self.get_variable(var_type='mask', lid=lid, tid=tid).numpy()
                    prev_mask_sig = self.generate_mask(prev_mask).numpy()
                    #################################################
                    prev_weights[lid][tid] = sw * prev_mask_sig + prev_aw
                    #################################################
            return prev_weights
        else:
            return self.model_body.get_weights()

    def set_body_weights(self, body_weights):
        if self.args.model in ['fedweit']:
            for lid, wgt in enumerate(body_weights):
                sw = self.get_variable('shared', lid)
                sw.assign(wgt)
        else:
            self.model_body.set_weights(body_weights)

    def switch_model_params(self, tid):
        for lid, dlay in self.decomposed_layers.items():
            dlay.sw = self.get_variable('shared', lid)
            dlay.aw = self.get_variable('adaptive', lid, tid)
            dlay.bias = self.get_variable('bias', lid, tid)
            dlay.mask = self.generate_mask(self.get_variable('mask', lid, tid))
            if self.args.model == 'fedweit':
                dlay.atten = self.get_variable('atten', lid, tid)
                dlay.aw_kb = self.get_variable('from_kb', lid, tid)

    def build_made(self, initial_weights, client_ids, decomposed=False):
        self.models = {}
        self.lock.acquire()
#        temp = MADE(self.input_shape, self.args.hidden_layers, self.input_shape, natural_order = self.args.natural_order, num_masks = self.args.num_masks, order_agn = self.args.order_agn, order_agn_step_size = self.args.order_agn_step_size, conn_agn_step_size = self.args.conn_agn_step_size, connectivity_weights = self.args.connectivity_weights, direct_input = self.args.direct_input, seed = self.args.seed, global_weights=initial_weights)
        for c in range(len(client_ids)):
            self.models[f"{client_ids[c]}"] = []
            seed = self.args.seed if  self.args.same_masks else self.args.seed+c*self.args.num_masks
            input_order_seed = self.args.seed if self.args.only_federated or self.args.same_input_order else None
            units_per_layer = np.concatenate(([self.input_shape], self.args.hidden_layers, [self.input_shape])) #in MADE case the input & output layer have the same amount of units
            temp = MADE(units_per_layer, natural_input_order = self.args.natural_input_order, num_masks = self.args.num_masks, order_agn = self.args.order_agn, connectivity_weights = self.args.connectivity_weights, direct_input = self.args.direct_input, seed = seed, input_order_seed = input_order_seed, global_weights=initial_weights, only_federated = self.args.only_federated, adaptive_factor = self.adaptive_factor, kernel_initializer = self.initializer, num_tasks = self.args.num_tasks)
            #for i in range(self.args.num_tasks): # build a model for each task
            model = temp.build_model()
            #model.add_loss(self.made_fedweit_loss(x, x_decoded_mean))
            self.models[f"{client_ids[c]}"] = model
            model.compile(optimizer=Adagrad(0.01, epsilon = 1e-6),run_eagerly=True)
            #model.compile(optimizer=Adam(self.args.lr),run_eagerly=True)
            if  c == 0:  model.summary()

    def build_lenet(self, initial_weights, decomposed=False):
        self.lock.acquire()
        self.models = []
        self.model_body = self.build_lenet_body(initial_weights, decomposed=decomposed)
        self.set_body_weights(initial_weights)
        self.initial_body_weights = initial_weights
        for i in range(self.args.num_tasks):
            self.models.append(self.add_head(self.model_body))
        self.lock.release()

    def build_lenet_body(self, initial_weights=None, decomposed=False):
        if decomposed:
            self.init_decomposed_variables(initial_weights)
            tid = 0
            model = tf.keras.models.Sequential()
            model.add(tf_keras.Input(shape=self.input_shape))
            for lid in [0, 1]:
                self.decomposed_layers[self.lid] = self.conv_decomposed(lid, tid,
                    filters = self.shapes[lid][-1],
                    kernel_size = (self.shapes[lid][0], self.shapes[lid][1]),
                    strides = (1,1),
                    padding = 'same',
                    acti = 'relu')
                model.add(self.decomposed_layers[self.lid])
                self.lid += 1
                model.add(tf_layers.Lambda(lambda x: tf.nn.lrn(x, 4, bias=1.0, alpha=0.001/9.0, beta=0.75) ))
                model.add(tf_layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same'))
            model.add(tf_layers.Flatten())
            for lid in [2, 3]:
                self.decomposed_layers[self.lid] = self.dense_decomposed(lid, tid,
                    units = self.shapes[lid][-1],
                    acti = 'relu')
                model.add(self.decomposed_layers[self.lid])
                self.lid += 1
        else:
            model = tf_models.Sequential()
            model.add(tf_layers.Conv2D(20, kernel_size=(5,5), use_bias=True, activation='relu', padding='same', kernel_regularizer=tf_regularizers.l2(self.args.wd), input_shape=self.input_shape))
            model.add(tf_layers.Lambda(lambda x: tf.nn.lrn(x, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)))
            model.add(tf_layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))
            model.add(tf_layers.Conv2D(50, kernel_size=(5,5), use_bias=True, activation='relu', padding='same', kernel_regularizer=tf_regularizers.l2(self.args.wd)))
            model.add(tf_layers.Lambda(lambda x: tf.nn.lrn(x, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)))
            model.add(tf_layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))
            model.add(tf_layers.Flatten())
            model.add(tf_layers.Dense(800, activation='relu', kernel_regularizer=tf_regularizers.l2(self.args.wd)))
            model.add(tf_layers.Dense(500, activation='relu', kernel_regularizer=tf_regularizers.l2(self.args.wd)))
        return model

    def add_head(self, body):
        head = tf_layers.Dense(self.args.num_classes, activation='softmax')
        body_out = body.output
        head_out = head(body_out)
        model = tf.keras.Model(inputs=body.input, outputs=head_out)
        self.heads.append(head)
        self.initial_heads_weights.append(head.get_weights())
        return model # multiheaded model

    def conv_decomposed(self, lid, tid, filters, kernel_size, strides, padding, acti):
        return  DecomposedConv(
            name        = 'layer_{}'.format(lid),
            filters     = filters,
            kernel_size = kernel_size,
            strides     = strides,
            padding     = padding,
            activation  = acti,
            lambda_l1   = self.args.lambda_l1,
            lambda_mask = self.args.lambda_mask,
            shared      = self.get_variable('shared', lid),
            adaptive    = self.get_variable('adaptive', lid, tid),
            from_kb     = self.get_variable('from_kb', lid, tid),
            atten       = self.get_variable('atten', lid, tid),
            bias        = self.get_variable('bias', lid, tid), use_bias=True,
            mask        = self.generate_mask(self.get_variable('mask', lid, tid)),
            kernel_regularizer = tf_regularizers.l2(self.args.wd))

    def dense_decomposed(self, lid, tid, units, acti):
        return DecomposedDense(
            name        = 'layer_{}'.format(lid),
            activation  = acti,
            units       = units,
            lambda_l1   = self.args.lambda_l1,
            lambda_mask = self.args.lambda_mask,
            shared      = self.get_variable('shared', lid),
            adaptive    = self.get_variable('adaptive', lid, tid),
            from_kb     = self.get_variable('from_kb', lid, tid),
            atten       = self.get_variable('atten', lid, tid),
            bias        = self.get_variable('bias', lid, tid), use_bias=True,
            mask        = self.generate_mask(self.get_variable('mask', lid, tid)),
            kernel_regularizer = tf_regularizers.l2(self.args.wd))
