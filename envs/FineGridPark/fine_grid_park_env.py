import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
from typing import Optional
from envs.FineGridPark.unit_packer import Unit,UnitPacker


class FineGridPark(gym.Env):
    '''
    精细网格停车场环境,其中每个状态由步长为1采样得到的box组成
    0,0---------------0,ncol-1
    |                      |
    |                      |
    |                      |
    |                      |
    |                      |
    |                      |
    |                      |
    |                      |
nrow-1,0 ---------------ncol-1,nrow-1
    '''
    def __init__(self,config:Optional[dict] = None):
        init_matrix = config.get("matrix",[]) # 停车场矩阵,反应每个网格的占用状态，0或1，0表示无空位
        self.div_ind = config.get("div_ind",2) # 网格分割系数，1表示一个网格宽度等于一个车头，2表示两个网格宽度等于一个车头，3表示三个网格宽度等于一个车头，4表示四个网格宽度等于一个车头
        self.box_size = self.div_ind*2
        self.packer = UnitPacker(self.box_size)
        self.matrix, self.units = self.packer.convert(init_matrix)
        self.nrow = len(self.matrix)
        self.ncol = len(self.matrix[0])


        self.max_step_index = config.get("max_step_index",1.5)
        self.vision_range = config.get("vision_range",7)
        assert self.vision_range % 2 == 1, "vision_range must be odd"
        self.render_mode = config.get("render_mode", "rgb_array")

        self.save = config.get("save", True)
        self.train = config.get("train", True)
        
        self.agent_state = -1 #前进时将占用一排div_ind X 2的网格，该变量表示智能体占用的行进方向上最左侧的网格
        self.agent_dir = -1 #当前行进方向,0:上,1:左,2:下,3:右

        self.action_space = Discrete(4) # 动作空间，0:向前，1:向后，2:向左，3:向右

        
        unit_width = self.div_ind*2
        self.observation_space = Box(low=0,high=1,shape=(self.vision_range*unit_width*2,self.vision_range*unit_width*2,2),dtype=np.float32)
        
        self.step_count = 0
        self.max_step = self.ncol*self.nrow*self.max_step_index
        self.iter_count = 0

        #渲染窗口
        self.window = None
        self.window_size_per_block = 64
        self.clock = None


        #训练统计
        self.model = None
        self.epoch = 0
        self.best_avg_reward = -np.inf
        self.max_park_num = 0

        self.clean_last_train_result = True

                    
                    
                    

        
    def reset(self,seed:Optional[int] = None,options:Optional[dict] = None):
        random.seed(seed)
        self.iter_count += 1

        #set init coord
        if len(self.entrance_states) == 0:
            raise ValueError("no entrance states")
        else:
            init_states_tuple = random.choice(self.entrance_states)
        
        self.agent_dir = init_states_tuple[1]
        self.init_agent_state(init_states_tuple[0],self.agent_dir)

        self.step_count = 0
        self.rewards = []
        self.traj = {}

        obs = self.observe()
        return obs,{}
    
    def step(self,action:int):
        assert action in self.action_space, f'Invalid action: {action}'

        self.step_count += 1
        pre_state = self.agent_state
        pre_dir = self.agent_dir
        self.action = action

        #向前
        if action == 0:
            self.agent_dir = self.agent_dir
        #向后
        elif action == 1:
            self.agent_dir = (self.agent_dir - 2) % 4
        #向左
        elif action == 2:
            self.agent_dir = (self.agent_dir + 1) % 4
        #向右
        elif action == 3:
            self.agent_dir = (self.agent_dir - 1) % 4 
        
    

    def observe(self):
        #TODO: 观察当前状态
        return []
    
  
        