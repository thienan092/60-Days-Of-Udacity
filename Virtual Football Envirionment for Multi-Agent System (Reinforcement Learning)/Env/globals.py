import numpy as np
import math

N_PLAYER = 5
MAP_WIDTH = 14400
MAP_HEIGHT = 9600
GOALWIDTH = 3000
W_BINS = 64
H_BINS = 48
FORCE_MAX = 100
FORCE_MIN = 20
FORCE_FD = 4
ROOT_STATE_SIZE = 1 - 1 + (2 + 2) + (2 * N_PLAYER) + (2 * N_PLAYER)
STATE_SIZE = ROOT_STATE_SIZE * (ROOT_STATE_SIZE + 1)
MOVE_ACTION_SIZE = W_BINS * H_BINS
SHOOT_ACTION_SIZE = MOVE_ACTION_SIZE * FORCE_FD
SHOOT_RANGE = 200
MOVE_ACTION = 1
SHOOT_ACTION = 2

SEVER_SIZE = 4 + (2 + 2) + (3 * N_PLAYER) + (3 * N_PLAYER)
BOT_SIZE = 4 * N_PLAYER

GAMMA = 0.9
MODEL_SAVE_PATH = "check_points/best_model.ckpt"
REPLAY_SAVE_PATH = "matches_data/replay"
HALF_1 = 1
HALF_2 = 2


class Position:
    def __init__(self, _x = 0, _y = 0):
        self.x = _x
        self.y = _y
    
class Object:
    def __init__(self):
        self.ID = 0
        self.m_action = 0
        self.m_force = 0
        self.m_pos = Position()
        self.m_targetPos = Position()
        self.m_moveSpeed = Position()
    
class Features:
    def __init__(self, Player_A, Player_B, oBall):
        self.AtoB = [[0 for _ in range(N_PLAYER)] for _ in range(N_PLAYER)]
        self.AtoBall = [0 for _ in range(N_PLAYER)]
        self.Bmap = [id for id in range(N_PLAYER)]
        self.calc_AtoB(Player_A, Player_B)
        self.calc_AtoBall(Player_A, oBall)
        self.calc_Bmap()
    
    def calc_AtoB(self, Player_A, Player_B):
        for i in range(N_PLAYER):
            for j in range(N_PLAYER):
                self.AtoB[i][j] = distance(Player_A[i], Player_B[j])
                
    def calc_AtoBall(self, Player_A, oBall):
        for i in range(N_PLAYER):
            self.AtoBall[i] = distance(Player_A[i], oBall)
            
    def calc_Bmap(self):
        mark = [False for _ in range(N_PLAYER)]
        old_id1 = -1
        old_id2 = -1
        old_id3 = -1
        old_id4 = -1
        old_id5 = -1
        min = None
        for id1 in range(N_PLAYER):
            # ID1 check validate and then, swap marks
            if (mark[id1] == False): 
                if (old_id1 != -1):
                    mark[old_id1] = False
                mark[id1] = True
                old_id1 = id1
            else:
                continue
                
            for id2 in range(N_PLAYER):
            
                # ID2 check validate and then, swap marks
                if (mark[id2] == False): 
                    if (old_id2 != -1):
                        mark[old_id2] = False
                    mark[id2] = True
                    old_id2 = id2
                else:
                    continue
                    
                for id3 in range(N_PLAYER):
                    
                    # ID3 check validate and then, swap marks
                    if (mark[id3] == False): 
                        if (old_id3 != -1):
                            mark[old_id3] = False
                        mark[id3] = True
                        old_id3 = id3
                    else:
                        continue
                    
                    for id4 in range(N_PLAYER):
                    
                        # ID4 check validate and then, swap marks
                        if (mark[id4] == False): 
                            if (old_id4 != -1):
                                mark[old_id4] = False
                            mark[id4] = True
                            old_id4 = id4
                        else:
                            continue

                        id5 = sum(range(N_PLAYER)) - (id1 + id2 + id3 + id4)
                        mark[id5] = False
                        
                        # Update min, map #
                        dist_map_value = 0.0
                        tempt_map = [id1, id2, id3, id4, id5]
                        for id_map in range(N_PLAYER):
                            dist_map_value = dist_map_value + self.AtoB[id_map][tempt_map[id_map]]
                        
                        if (min is None) or (min > dist_map_value):
                            min = dist_map_value
                            for id_map in range(N_PLAYER):
                                self.Bmap[id_map] = tempt_map[id_map]                        
                            
                        # Update min, map #
                        
                    if (old_id4 != -1):
                        mark[old_id4] = False
                    old_id4 = -1
                    
                if (old_id3 != -1):
                    mark[old_id3] = False
                old_id3 = -1
                        
            if (old_id2 != -1):
                mark[old_id2] = False
            old_id2 = -1
        
def distance(A, B):
    return math.sqrt((A.m_pos.x - B.m_pos.x)**2 + (A.m_pos.y - B.m_pos.y)**2)
        
def to_pos_x_y(q_id):
    x = int(q_id / H_BINS)
    y = q_id % H_BINS
    x = int(x * (float(MAP_WIDTH) / (W_BINS - 1)))
    y = int(y * (float(MAP_HEIGHT) / (H_BINS - 1)))
    return x, y
    
def to_pos_x_y_f(q_id):
    x = q_id / (H_BINS * FORCE_FD)
    r = q_id % (H_BINS * FORCE_FD)
    y = r / FORCE_FD
    f = r % FORCE_FD
    x = int(x * (float(MAP_WIDTH) / (W_BINS - 1)))
    y = int(y * (float(MAP_HEIGHT) / (H_BINS - 1)))
    f = int(f * (float(FORCE_MAX - FORCE_MIN) / (FORCE_FD - 1))) + FORCE_MIN
    return x, y, f

def x_y_to_id(x, y):
    r0_q_id = int(1.0 * x * (W_BINS - 1) / MAP_WIDTH)
    r1_q_id = int(1.0 * y * (H_BINS - 1) / MAP_HEIGHT)
    return r0_q_id * H_BINS + r1_q_id
    
def x_y_f_to_id(x, y, f):
    r0_q_id = int(1.0 * x * (W_BINS - 1) / MAP_WIDTH)
    r1_q_id = int(1.0 * y * (H_BINS - 1) / MAP_HEIGHT)
    r2_q_id = int(1.0 * (f - FORCE_MIN) * (FORCE_FD - 1) / (FORCE_MAX - FORCE_MIN))
    return r0_q_id * H_BINS * FORCE_FD + r1_q_id * FORCE_FD + r2_q_id

def input_to_states(gameTurn, oBall, Player_A, Player_B, Bmap):
    state = np.array([0.0 for _ in range(STATE_SIZE+1)])
    state[0] = gameTurn
    state[1] = oBall.m_pos.x
    state[2] = oBall.m_pos.y
    state[3] = oBall.m_moveSpeed.x
    state[4] = oBall.m_moveSpeed.y
    
    for i in range(N_PLAYER):
        state[5+i] = Player_A[i].m_pos.x
        state[10+i] = Player_A[i].m_pos.y
        state[15+i] = Player_B[Bmap[i]].m_pos.x
        state[20+i] = Player_B[Bmap[i]].m_pos.y
    return state[1:]
    
def norm_state(state_var):
    state = np.array(state_var)
    state[0] = (state[0] - MAP_WIDTH/2) / MAP_WIDTH
    state[1] = (state[1] - MAP_HEIGHT/2) / MAP_HEIGHT
    state[2] = state[2] / 1000
    state[3] = state[3] / 1000
    
    for i in range(N_PLAYER):
        state[4+i] = (state[4+i] - MAP_WIDTH/2) / MAP_WIDTH
        state[9+i] = (state[9+i] - MAP_HEIGHT/2) / MAP_HEIGHT
        state[14+i] = (state[14+i] - MAP_WIDTH/2) / MAP_WIDTH
        state[19+i] = (state[19+i] - MAP_HEIGHT/2) / MAP_HEIGHT
        
    for i in range(24):
        for j in range(24):
            state[i * 24 + j + 24] = state[i] * state[j];
    
    return state