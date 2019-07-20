import os, re, itertools, random, math
import tensorflow as tf
import numpy as np
from collections import namedtuple, deque
from features import Features, states_to_complex_states
import globals, model

            
class ServerInfo:
    def __init__(self):
        self.oBall = globals.Object()
        self.Player_A = [globals.Object() for _ in range(globals.N_PLAYER)]
        self.Player_B = [globals.Object() for _ in range(globals.N_PLAYER)]
        self.features = None
        self.gameTurn = -1
        self.scoreTeamA = -1
        self.scoreTeamB = -1
        self.stateMath = -1
        
    def parse_server_content(self, server_content):
        item_list = server_content.split(" ")
        match = int(item_list[3])
        if (match == globals.HALF_1):
            self.gameTurn, self.scoreTeamA, self.scoreTeamB, self.stateMath = [int(num) for num in item_list[:4]]
            self.oBall.m_pos.x, self.oBall.m_pos.y, self.oBall.m_moveSpeed.x, self.oBall.m_moveSpeed.y = [int(num) for num in item_list[4:(4+4)]]
            i = 0
            for (a, b, c) in zip(*[iter(item_list[8:(8+(3*globals.N_PLAYER))])]*3):
                self.Player_A[i].ID = int(a)
                self.Player_A[i].m_pos.x = int(b)
                self.Player_A[i].m_pos.y = int(c)
                i = i + 1
            i = 0
            for (a, b, c) in zip(*[iter(item_list[23:])]*3):
                self.Player_B[i].ID = int(a)
                self.Player_B[i].m_pos.x = int(b)
                self.Player_B[i].m_pos.y = int(c)
                i = i + 1
        else:
            self.gameTurn, self.scoreTeamB, self.scoreTeamA, self.stateMath = [int(num) for num in item_list[:4]]
            self.oBall.m_pos.x, self.oBall.m_pos.y, self.oBall.m_moveSpeed.x, self.oBall.m_moveSpeed.y = [int(num) for num in item_list[4:(4+4)]]
            i = 0
            for (a, b, c) in zip(*[iter(item_list[8:(8+(3*globals.N_PLAYER))])]*3):
                self.Player_B[i].ID = int(a)
                self.Player_B[i].m_pos.x = int(b)
                self.Player_B[i].m_pos.y = int(c)
                i = i + 1
            i = 0
            for (a, b, c) in zip(*[iter(item_list[23:])]*3):
                self.Player_A[i].ID = int(a)
                self.Player_A[i].m_pos.x = int(b)
                self.Player_A[i].m_pos.y = int(c)
                i = i + 1
                
        self.features = Features(self.Player_A, self.Player_B, self.oBall)
        return match
            
class BotAInfo:
    def __init__(self):
        self.Player_A = [globals.Object() for _ in range(globals.N_PLAYER)]
        
    def parse_BotA_content(self, BotA_content):
        item_list = BotA_content.split(" ")
        i = 0
        for (a, x, y, f) in zip(*[iter(item_list)]*4):
            self.Player_A[i].m_action = int(a)
            self.Player_A[i].m_targetPos.x = int(x)
            self.Player_A[i].m_targetPos.y = int(y)
            self.Player_A[i].m_force = int(f)
            i = i + 1
            
def get_reward(si, si_f):
    opGOAL_x = globals.MAP_WIDTH
    opGOAL_up = globals.MAP_HEIGHT/2 - globals.GOALWIDTH/2
    opGOAL_low = globals.MAP_HEIGHT/2 + globals.GOALWIDTH/2
    
    global_reward = np.float32(0.0)
    partial_rewards = [np.float32(0.0) for _ in range(globals.N_PLAYER)]
    
    max_dist = math.sqrt(globals.MAP_WIDTH**2 + globals.MAP_HEIGHT**2)
    # 3 * (d(min_B, Ball) - d(min_A, Ball)) / math.sqrt(globals.MAP_WIDTH**2 + globals.MAP_HEIGHT**2)
    for i in range(globals.N_PLAYER):
        partial_rewards[i] = - (globals.distance(si.Player_A[i], si.oBall) / max_dist)
        partial_rewards[i] = - ((globals.distance(si_f.Player_A[i], si_f.oBall) / max_dist) ** 2) - (globals.distance(si_f.Player_A[i], si_f.oBall) / max_dist) - partial_rewards[i]
        # global_reward = global_reward + globals.distance(si_f.Player_B[i], si_f.oBall)

    
    # global_reward = global_reward / (globals.N_PLAYER * max_dist)
    
    # -(d(min_B) < globals.SHOOT_RANGE) + (d(min_A) < globals.SHOOT_RANGE)
    # if (minB <= globals.SHOOT_RANGE):
        # reward = reward - 1
    # else:
        # reward = reward + 1
        
    # if (minA <= globals.SHOOT_RANGE):
        # reward = reward + 1
    # else:
        # reward = reward - 1
    
    # 5 * min(d(Ball, opGOAL)) / math.sqrt(globals.MAP_WIDTH**2 + globals.MAP_HEIGHT**2)
    d_to_goal = np.float32(0.0)

    if ((si_f.oBall.m_pos.y >= opGOAL_up) and (si_f.oBall.m_pos.y <= opGOAL_low)):
        d_to_goal = np.float32(globals.MAP_WIDTH / 2) - si_f.oBall.m_pos.x

    elif (si_f.oBall.m_pos.y < opGOAL_up):
        up_pos = globals.Object()
        up_pos.m_pos.x = opGOAL_x
        up_pos.m_pos.y = opGOAL_up
        d_to_goal = globals.distance(si_f.oBall, up_pos) - np.float32(globals.MAP_WIDTH / 2)
    else:
        low_pos = globals.Object()
        low_pos.m_pos.x = opGOAL_x
        low_pos.m_pos.y = opGOAL_low
        d_to_goal = globals.distance(si_f.oBall, low_pos) - np.float32(globals.MAP_WIDTH / 2)
        
    global_reward = global_reward - 2.0 * d_to_goal / (globals.MAP_WIDTH / 2)
        
    d_to_goal = np.float32(0.0)

    if ((si.oBall.m_pos.y >= opGOAL_up) and (si.oBall.m_pos.y <= opGOAL_low)):
        d_to_goal = np.float32(globals.MAP_WIDTH / 2) - si.oBall.m_pos.x

    elif (si.oBall.m_pos.y < opGOAL_up):
        up_pos = globals.Object()
        up_pos.m_pos.x = opGOAL_x
        up_pos.m_pos.y = opGOAL_up
        d_to_goal = globals.distance(si.oBall, up_pos) - np.float32(globals.MAP_WIDTH / 2)
    else:
        low_pos = globals.Object()
        low_pos.m_pos.x = opGOAL_x
        low_pos.m_pos.y = opGOAL_low
        d_to_goal = globals.distance(si.oBall, low_pos) - np.float32(globals.MAP_WIDTH / 2)
        
    global_reward = global_reward - (-2.0 * d_to_goal / (globals.MAP_WIDTH / 2))
    
    if (si_f.scoreTeamA > si.scoreTeamA):
        global_reward = global_reward + 4
        
    if (si_f.scoreTeamB > si.scoreTeamB):
        global_reward = global_reward - 3
    
    
    global_reward = global_reward 
    return [partial_rewards, global_reward]
    
            
class PEReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size=50000, e = 0.001, a = 0.95):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.probs = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.count = 0
        self.experience = namedtuple("Experience", field_names=["state", "actions", "reward", "next_state", "done"])
        
        self.e = np.float32(e)
        self.a = np.float32(a)
        
    def load_replay(self, file_content_str):
        packages = file_content_str.split("Server send:")[2:-1]
        for i in range(len(packages)-1):
            si = ServerInfo()
            si_f = ServerInfo()
            bai = BotAInfo()
            all_content = "Server send:" + packages[i]
            all_content_f = "Server send:" + packages[i+1]
            s_m = re.search(r"Server send:'([ 0-9-]*)'", all_content)
            error_m = re.search(r"Error", all_content)
            s_m_f = re.search(r"Server send:'([ 0-9-]*)'", all_content_f)
            a_m = re.search(r"Bot A send:'([ 0-9-]*)'", all_content)
            b_m = re.search(r"Bot B send:'([ 0-9-]*)'", all_content)
            server_content = ""
            server_content_f = ""
            if (s_m is not None) and (s_m_f is not None) and (error_m is None): 
                server_content = s_m.group(1)
                server_content_f = s_m_f.group(1)
            else:
                continue
            
            current_match = si.parse_server_content(server_content)
            
            if (current_match != si_f.parse_server_content(server_content_f)):
                rep_temp = self.memory[-1]
                self.memory.pop()
                self.probs.pop()
                self.count -= 1
                self.add(rep_temp.state, rep_temp.actions, rep_temp.reward, rep_temp.next_state, 1)
                continue
                
            if (current_match == globals.HALF_1):
                if ((a_m is not None) and (b_m is not None)): 
                    BotA_content = a_m.group(1)
                else:
                    continue
            elif (current_match == globals.HALF_2):
                if ((a_m is not None) and (b_m is not None)): 
                    BotA_content = b_m.group(1)
                else:
                    continue
            else:
                break
            bai.parse_BotA_content(BotA_content)
            
            
            state = globals.input_to_states(si.gameTurn, si.oBall, si.Player_A, si.Player_B, si.features.Bmap)
            next_state = globals.input_to_states(si_f.gameTurn, si_f.oBall, si_f.Player_A, si_f.Player_B, si_f.features.Bmap)
            # print(si.features.Bmap)
            
            
            actions = [-1 for _ in range(globals.N_PLAYER)]
            for j in range(globals.N_PLAYER):
                if bai.Player_A[j].m_action == globals.MOVE_ACTION:
                    actions[j] = globals.x_y_to_id(bai.Player_A[j].m_targetPos.x, bai.Player_A[j].m_targetPos.y)
                else:
                    actions[j] = globals.MOVE_ACTION_SIZE + globals.x_y_f_to_id(bai.Player_A[j].m_targetPos.x, bai.Player_A[j].m_targetPos.y, bai.Player_A[j].m_force)
            
            reward = get_reward(si, si_f)
            
            # print(reward)
            
            self.add(state, actions, reward, next_state, 0)
            
        rep_temp = self.memory[-1]
        self.memory.pop()
        self.probs.pop()
        self.count -= 1
        self.add(rep_temp.state, rep_temp.actions, rep_temp.reward, rep_temp.next_state, 1)

    def add(self, state, actions, reward, next_state, done):
        """Add a new experience to memory."""
        if self.count < self.buffer_size: 
            
            self.count += 1
        else:
            self.memory.popleft()
            self.probs.popleft()
            
        self.memory.append(self.experience(state, actions, reward, next_state, done))
        partial_rewards = reward[0]
        global_rewards = reward[1]
        self.probs.append(((np.absolute(np.sum(partial_rewards) + global_rewards) + self.e) ** self.a)) # ((abs(reward) + self.e) ** self.a)
        
        
    def normalize_probs(self):
        return np.array(self.probs) / sum(np.array(self.probs))
        
    def get_experience_by_index(self, index):
        return self.memory[index]
        
    def get_probs_by_index(self, index):
        return self.probs[index]

    def sample_indices(self, batch_size, get_last_num=None):
        """Randomly sample a batch of experiences from memory."""
        idx = None
        if (get_last_num is None):
            idx = np.random.choice(range(len(self.memory)), size=batch_size, p=self.normalize_probs())
        else:
            idx = np.random.choice(range(max(0, len(self.memory) - get_last_num * batch_size - 1), len(self.memory) - 1), size=batch_size)
        return idx

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Trainer:
    def __init__(self, team_group, batch_size, epoch_count, sess):
        self.sess = sess
        self.team_group = team_group
        self.batch_size = batch_size
        self.epoch_count = epoch_count
        self.replay_buffer = PEReplayBuffer()
        
    def update_buffer(self, file_path_list):
        for file_path in file_path_list:
            with open(file_path, "r") as rep_file:
                self.replay_buffer.load_replay(rep_file.read())
                print("buffer len: " + str(len(self.replay_buffer)))
                
    def test(self, bot_num):
        for i in range(int(0.2 * self.epoch_count)):
            experiences_indices = self.replay_buffer.sample_indices(batch_size=self.batch_size, get_last_num=int(0.2 * self.epoch_count))
            states = np.vstack([states_to_complex_states(self.replay_buffer.get_experience_by_index(id).state) for id in experiences_indices])
            partial_rewards = np.array([self.replay_buffer.get_experience_by_index(id).reward[0] for id in experiences_indices]).astype(np.float32).reshape(-1, globals.N_PLAYER)
            global_rewards = np.array([self.replay_buffer.get_experience_by_index(id).reward[1] for id in experiences_indices]).astype(np.float32).reshape(-1, 1)
            next_states = np.vstack([states_to_complex_states(self.replay_buffer.get_experience_by_index(id).next_state) for id in experiences_indices])
            dones = np.array([self.replay_buffer.get_experience_by_index(id).done for id in experiences_indices]).astype(np.int32).reshape(-1, 1)
            
            uni_probs = np.array([1.0 for _ in experiences_indices]).astype(np.float32).reshape(-1, 1)
            
            
            max_Qs_next = [None for _ in range(bot_num)]
            max_Qs = [None for _ in range(bot_num)]
            for bot_id in range(bot_num):
                actions = np.array([self.replay_buffer.get_experience_by_index(id).actions[bot_id] for id in experiences_indices]).astype(np.float32).reshape(-1, 1)
                
                target_Qs = self.sess.run(self.team_group.team_member_move_shoot_output[bot_id], 
                            feed_dict={self.team_group.states: next_states, self.team_group.is_training: False})

                max_Qs_next[bot_id] = np.max(target_Qs, axis=1).reshape(-1,1)
                
                max_Qs[bot_id] = self.sess.run(self.team_group.team_member_move_shoot_q_nets[bot_id], 
                            feed_dict={self.team_group.states: states,
                                       self.team_group.actions_: actions, self.team_group.is_training: False})
            
            for bot_id in range(bot_num):
                actions = np.array([self.replay_buffer.get_experience_by_index(id).actions[bot_id] for id in experiences_indices]).astype(np.float32).reshape(-1, 1)
                print("Testing Bot {} ...".format(bot_id))

                bot_reward = np.array(global_rewards) + np.sum(partial_rewards, axis=1).reshape(-1, 1)
                print("Total Reward: {}".format(bot_reward[0]))
                for i1 in range(bot_num):
                    bot_reward = bot_reward - (max_Qs[i1] - globals.GAMMA * max_Qs_next[i1] * dones)
                bot_reward = bot_reward + (max_Qs[bot_id] - globals.GAMMA * max_Qs_next[bot_id] * dones)
                print("Partial Reward: {}".format(bot_reward[0]))
                
                targets = bot_reward + globals.GAMMA * max_Qs_next[bot_id] * dones

                loss = self.sess.run(self.team_group.team_member_move_shoot_loss[bot_id],
                            feed_dict={self.team_group.states: states,
                                       self.team_group.targetQs_: targets,
                                       self.team_group.actions_: actions, self.team_group.gradient_weights: uni_probs, self.team_group.is_training: False})
                                       
                print(loss)
        
    def train(self, bot_num):
        for i in range(self.epoch_count):
            experiences_indices = self.replay_buffer.sample_indices(batch_size=self.batch_size)
            
            probs = np.array([self.replay_buffer.get_probs_by_index(id) for id in experiences_indices]).astype(np.float32).reshape(-1, 1)
            p_b_weights = np.float32(1.0) / probs
            states = np.vstack([states_to_complex_states(self.replay_buffer.get_experience_by_index(id).state) for id in experiences_indices])
            partial_rewards = np.array([self.replay_buffer.get_experience_by_index(id).reward[0] for id in experiences_indices]).astype(np.float32).reshape(-1, globals.N_PLAYER)
            global_rewards = np.array([self.replay_buffer.get_experience_by_index(id).reward[1] for id in experiences_indices]).astype(np.float32).reshape(-1, 1)
            next_states = np.vstack([states_to_complex_states(self.replay_buffer.get_experience_by_index(id).next_state) for id in experiences_indices])
            dones = np.array([self.replay_buffer.get_experience_by_index(id).done for id in experiences_indices]).astype(np.int32).reshape(-1, 1)
            
            
            max_Qs_next = [None for _ in range(bot_num)]
            max_Qs = [None for _ in range(bot_num)]
            for bot_id in range(bot_num):
                actions = np.array([self.replay_buffer.get_experience_by_index(id).actions[bot_id] for id in experiences_indices]).astype(np.float32).reshape(-1, 1)
                
                target_Qs = self.sess.run(self.team_group.team_member_move_shoot_output[bot_id], 
                            feed_dict={self.team_group.states: next_states, self.team_group.is_training: False})

                max_Qs_next[bot_id] = np.max(target_Qs, axis=1).reshape(-1,1)
                
                max_Qs[bot_id] = self.sess.run(self.team_group.team_member_move_shoot_q_nets[bot_id], 
                            feed_dict={self.team_group.states: states,
                                       self.team_group.actions_: actions, self.team_group.is_training: False})
            
            for bot_id in range(bot_num):
                actions = np.array([self.replay_buffer.get_experience_by_index(id).actions[bot_id] for id in experiences_indices]).astype(np.float32).reshape(-1, 1)
                print("Training Bot {} ...".format(bot_id))

                bot_reward = np.array(global_rewards) + np.sum(partial_rewards, axis=1).reshape(-1, 1)
                print("Total Reward: {}".format(bot_reward[0]))
                for i1 in range(bot_num):
                    bot_reward = bot_reward - (max_Qs[i1] - globals.GAMMA * max_Qs_next[i1] * dones)
                bot_reward = bot_reward + (max_Qs[bot_id] - globals.GAMMA * max_Qs_next[bot_id] * dones)
                print("Partial Reward: {}".format(bot_reward[0]))
                
                targets = bot_reward + globals.GAMMA * max_Qs_next[bot_id] * dones

                loss, _ = self.sess.run([self.team_group.team_member_move_shoot_loss[bot_id], self.team_group.team_member_move_shoot_opt[bot_id]],
                            feed_dict={self.team_group.states: states,
                                       self.team_group.targetQs_: targets,
                                       self.team_group.actions_: actions, self.team_group.gradient_weights: p_b_weights, self.team_group.is_training: True})
                                       
                print(loss)
            
        print("Saving After Training...")
        t_vars = tf.trainable_variables()
        vars_list = [var for var in t_vars]
        saver = tf.train.Saver(var_list=vars_list)
        saver.save(self.sess, globals.MODEL_SAVE_PATH)
        print("Done!")
        
    def perform_layer_output_test(self, complex_states, bot_num):
        outputs = [None for _ in range(bot_num)]
        for bot_id in range(bot_num):
            outputs[bot_id] = self.sess.run(self.team_group.test_layer[bot_id], 
                        feed_dict={self.team_group.states: complex_states, self.team_group.is_training: False}).squeeze(0)
                        
        return outputs