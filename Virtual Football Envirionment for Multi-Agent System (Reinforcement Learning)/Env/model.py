import tensorflow as tf
import numpy as np
import os
import globals

LAYER_SIZE1 = 512
LAYER_SIZE2 = 256
ALPHA = 0.01
KEEP_PROB = 1.0


        
def save_running_model_matrices(sess):

    def matrices_to_file(var_dict):
        if not os.path.exists("matrices0/"):
            os.makedirs("matrices0/")
        if not os.path.exists("matrices1/"):
            os.makedirs("matrices1/")
            
        for key, value in var_dict.items():
            file_name_part = key.split("/")
            file_name = "matrices0/" + file_name_part[0] + "_" + file_name_part[1] + "_" + file_name_part[2][0:-2] + ".npy"
            np.save(file_name, value[0].flatten())# , header=key, delimiter=",", newline="},\n{")
            file_name = "matrices1/" + file_name_part[0] + "_" + file_name_part[1] + "_" + file_name_part[2][0:-2] + ".npy"
            np.save(file_name, value[0].flatten())
            
    print("Saving matrices!!!")
    t_vars = tf.trainable_variables()
    g_vars = tf.global_variables()
    var_dict = dict()
    for var in t_vars:
        if (var.name.startswith('team_member_move_shoot_') and (("beta" in var.name) or ("gamma" in var.name) or ("bias" in var.name) or ("kernel" in var.name))):
            np_var = sess.run([var])
            var_dict[var.name] = np_var
        
    for var in g_vars:
        if (var.name.startswith('team_member_move_shoot_') and (("moving_variance" in var.name) or ("moving_mean" in var.name))):
            np_var = sess.run([var])
            var_dict[var.name] = np_var
    
    matrices_to_file(var_dict)
    print("Done!!!")

class Group:
    """Actor (Policy) Model."""

    def __init__(self):
        """Initialize parameters and build model."""
        self.test_layer = [None for _ in range(globals.N_PLAYER)]
        self.team_member_move_shoot_q_nets = [None for _ in range(globals.N_PLAYER)]
        self.team_member_move_shoot_output = [None for _ in range(globals.N_PLAYER)]
        self.team_member_move_shoot_loss = [None for _ in range(globals.N_PLAYER)]
        self.team_member_move_shoot_opt = [None for _ in range(globals.N_PLAYER)]
        
        self.build_model()
        self.build_loss()
        
    def build_model(self):
        self.states = tf.placeholder(tf.float32, (None, globals.STATE_SIZE), name='states')
        # One hot encode the actions to later choose the Q-value for the action
        self.actions_ = tf.placeholder(tf.int32, (None, 1), name='actions')
        
        self.gradient_weights = tf.placeholder(tf.float32, (None, 1), name='gradient_weights')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        # Target Q values for training
        self.targetQs_ = tf.placeholder(tf.float32, (None, 1), name='targetQs')

        # Create move shoot model for each member and bind it to coach model
        for i in range(globals.N_PLAYER):
            self.team_member_move_shoot_output[i], self.team_member_move_shoot_q_nets[i] = self.team_member_move_shoot_model(self.states, False, i)
        
    def build_loss(self):
        # for team_member_move var list and team_member_shoot var list
        for i in range(globals.N_PLAYER):
            self.team_member_move_shoot_loss[i] = tf.reduce_mean(self.gradient_weights * tf.square(self.targetQs_ - self.team_member_move_shoot_q_nets[i]))
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.team_member_move_shoot_opt[i] = tf.train.AdamOptimizer().minimize(self.team_member_move_shoot_loss[i])

        
    def team_member_move_shoot_model(self, states, reuse, id):
        """Build an actor (policy) network that maps states -> actions."""
        one_hot_move_shoot_actions = tf.one_hot(self.actions_, globals.MOVE_ACTION_SIZE + globals.SHOOT_ACTION_SIZE)
        
        with tf.variable_scope('team_member_move_shoot_{}'.format(id), reuse=False):
            team_move_shoot_net = tf.layers.dense(states, LAYER_SIZE1, use_bias=False, activation=None)
            self.test_layer[id] = team_move_shoot_net
            team_move_shoot_net = tf.layers.batch_normalization(team_move_shoot_net, training=self.is_training)
            team_move_shoot_net = tf.maximum(ALPHA * team_move_shoot_net, team_move_shoot_net)
            #team_move_shoot_net = tf.nn.relu(team_move_shoot_net)
            #team_move_shoot_net = tf.nn.dropout(team_move_shoot_net, keep_prob=KEEP_PROB)
            
            
            team_move_shoot_net = tf.layers.dense(team_move_shoot_net, LAYER_SIZE1, use_bias=False, activation=None)
            team_move_shoot_net = tf.layers.batch_normalization(team_move_shoot_net, training=self.is_training)
            team_move_shoot_net = tf.maximum(ALPHA * team_move_shoot_net, team_move_shoot_net)
            #team_move_shoot_net = tf.nn.relu(team_move_shoot_net)
            #team_move_shoot_net = tf.nn.dropout(team_move_shoot_net, keep_prob=KEEP_PROB)
            
            team_move_shoot_net = tf.layers.dense(team_move_shoot_net, LAYER_SIZE2, use_bias=False, activation=None)
            team_move_shoot_net = tf.layers.batch_normalization(team_move_shoot_net, training=self.is_training)
            team_move_shoot_net = tf.maximum(ALPHA * team_move_shoot_net, team_move_shoot_net)
            #team_move_shoot_net = tf.nn.relu(team_move_shoot_net)
            #team_move_shoot_net = tf.nn.dropout(team_move_shoot_net, keep_prob=KEEP_PROB)
            
            team_move_shoot_net = tf.layers.dense(team_move_shoot_net, LAYER_SIZE2, use_bias=False, activation=None)
            team_move_shoot_net = tf.layers.batch_normalization(team_move_shoot_net, training=self.is_training)
            team_move_shoot_net = tf.maximum(ALPHA * team_move_shoot_net, team_move_shoot_net)
            #team_move_shoot_net = tf.nn.relu(team_move_shoot_net)
            #team_move_shoot_net = tf.nn.dropout(team_move_shoot_net, keep_prob=KEEP_PROB)
            
            output = tf.layers.dense(team_move_shoot_net, globals.MOVE_ACTION_SIZE + globals.SHOOT_ACTION_SIZE, activation=None, kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))
            Q = tf.reshape(tf.reduce_sum(tf.multiply(output, tf.squeeze(one_hot_move_shoot_actions)), axis=1), [-1, 1])
        return output, Q
