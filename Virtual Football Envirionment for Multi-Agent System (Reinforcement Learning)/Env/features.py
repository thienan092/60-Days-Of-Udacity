import globals

class Features:
    def __init__(self, Player_A, Player_B, oBall):
        self.AtoB = [[0 for _ in range(globals.N_PLAYER)] for _ in range(globals.N_PLAYER)]
        self.AtoBall = [0 for _ in range(globals.N_PLAYER)]
        self.Bmap = [id for id in range(globals.N_PLAYER)]
        self.calc_AtoB(Player_A, Player_B)
        self.calc_AtoBall(Player_A, oBall)
        self.calc_Bmap()
    
    def calc_AtoB(self, Player_A, Player_B):
        for i in range(globals.N_PLAYER):
            for j in range(globals.N_PLAYER):
                self.AtoB[i][j] = globals.distance(Player_A[i], Player_B[j])
                
    def calc_AtoBall(self, Player_A, oBall):
        for i in range(globals.N_PLAYER):
            self.AtoBall[i] = globals.distance(Player_A[i], oBall)
            
    def calc_Bmap(self):
        mark = [False for _ in range(globals.N_PLAYER)]
        old_id1 = -1
        old_id2 = -1
        old_id3 = -1
        old_id4 = -1
        old_id5 = -1
        min = None
        for id1 in range(globals.N_PLAYER):
            # ID1 check validate and then, swap marks
            if (mark[id1] == False): 
                if (old_id1 != -1):
                    mark[old_id1] = False
                mark[id1] = True
                old_id1 = id1
            else:
                continue
                
            for id2 in range(globals.N_PLAYER):
            
                # ID2 check validate and then, swap marks
                if (mark[id2] == False): 
                    if (old_id2 != -1):
                        mark[old_id2] = False
                    mark[id2] = True
                    old_id2 = id2
                else:
                    continue
                    
                for id3 in range(globals.N_PLAYER):
                    
                    # ID3 check validate and then, swap marks
                    if (mark[id3] == False): 
                        if (old_id3 != -1):
                            mark[old_id3] = False
                        mark[id3] = True
                        old_id3 = id3
                    else:
                        continue
                    
                    for id4 in range(globals.N_PLAYER):
                    
                        # ID4 check validate and then, swap marks
                        if (mark[id4] == False): 
                            if (old_id4 != -1):
                                mark[old_id4] = False
                            mark[id4] = True
                            old_id4 = id4
                        else:
                            continue

                        id5 = sum(range(globals.N_PLAYER)) - (id1 + id2 + id3 + id4)
                        mark[id5] = False
                        
                        # Update min, map #
                        dist_map_value = 0.0
                        tempt_map = [id1, id2, id3, id4, id5]
                        for id_map in range(globals.N_PLAYER):
                            dist_map_value = dist_map_value + self.AtoB[id_map][tempt_map[id_map]]
                        
                        if (min is None) or (min > dist_map_value):
                            min = dist_map_value
                            for id_map in range(globals.N_PLAYER):
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