import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
from typing import Optional
import pygame
import matplotlib.pyplot as plt
plt.ion()

class SimplePark(gym.Env):
    '''
    简单停车场环境
    0,ncol------------------nrow,ncol
    |                      |
    |                      |
    |                      |
    |                      |
    |                      |
    |                      |
    |                      |
    |                      |
    0,0 ------------------nrow,0
    '''
    def __init__(self, config:Optional[dict] = None):
        config = config or {}
        self.nrow = config.get("nrow", 10)
        self.ncol = config.get("ncol", 10)
        self.vision_range = config.get("vision_range", 5)
        #vison_range 是否为奇数？
        assert self.vision_range % 2 == 1, "vision_range must be odd"
        self.all_states = list(range(self.nrow*self.ncol))
        self.disabled_states = config.get("disabled_states", [])
        self.entrances_states = []
        entrances_states = config.get("entrances_states", [])
        self.render_mode = config.get("render_mode", "rgb_array")
        
        #出入口是否靠近边缘
        for s in entrances_states:
            if s in self.disabled_states:
                continue
            
            x,y = self.stateToCoord(s)
            if (x-1<0 or x+1>self.ncol-1 or y-1<0 or y+1>self.nrow-1)\
            or (s - self.ncol in self.disabled_states)\
            or(s + self.ncol in self.disabled_states)\
            or (s -1 in self.disabled_states)\
            or (s+1 in self.disabled_states):
                self.entrances_states.append(s)

        #set init state
        self.agent_state = 0 #当前坐标状态,仅用于环境判断位置
        self.agent_dir = -1 #当前行进方向

        self.action_space = Discrete(4) #动作空间，0：前，1：后，2：左，3：右

        self.observation_space = Box(low=0,high=1,shape=(5,self.vision_range,self.vision_range),dtype=np.float32)

        self.park_states = [] #当前是车位的坐标状态
        self.path_states = [] #当前是通道的坐标状态

        self.step_count = 0
        self.max_step = self.ncol*self.nrow*1.5

        #渲染窗口：
        self.window = None
        self.window_size_per_block = 32
        self.clock = None

        #self.reset()


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        random.seed(seed)
        #set init coord
        if len(self.entrances_states) == 0:
            self.agent_state = random.choice(self.all_states)
            
        else:
            #self.agent_state = random.choice(self.entrances_states)
            self.agent_state = self.entrances_states[1]

        self.agent_dir = -1

        accessible = self.__getStateAccess(self.agent_state)
        accessible_dirs = [i for i,value in enumerate(accessible) if value]
        #choose the opposite direction of the inaccessible direction
        if len(accessible_dirs)>0:
            for i,d in enumerate(accessible):
                if d == False:
                    if accessible[(i-2)%4] :
                        self.agent_dir = (i-2)%4
                        break
            if self.agent_dir == -1:
                self.agent_dir = random.choice(accessible_dirs)

        x,y = self.stateToCoord(self.agent_state)
        print(f"------智能体初始坐标:{(x,y)}------")

        dir_string = ""
        if self.agent_dir == -1:
            dir_string = "无效"
        elif self.agent_dir == 0:
            dir_string = "上"
        elif self.agent_dir == 1:
            dir_string = "左"
        elif self.agent_dir == 2:
            dir_string = "下"
        elif self.agent_dir == 3:
            dir_string = "右"

        print(f"------智能体初始方向:{dir_string}------")

        self.step_count = 0
<<<<<<< HEAD
<<<<<<< HEAD
=======
        self.path_states = [self.agent_state]
        self.park_states = []
>>>>>>> afdad09 (n)
=======
        self.path_states = [self.agent_state]
        self.park_states = []
>>>>>>> afdad09 (n)
        obs = self.observe()
        return obs,{}

    def step(self, action:int):
        assert action in self.action_space, f'Invalid action: {action}'

        self.step_count += 1
        pre_state = self.agent_state

        if action == 0:
            self.agent_dir = self.agent_dir
        elif action == 1:
            self.agent_dir = (self.agent_dir - 2) % 4
        elif action == 2:
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == 3:
            self.agent_dir = (self.agent_dir - 1) % 4

        next_state = self.__nextState(self.agent_state,self.agent_dir)
        if next_state == self.agent_state:
            #如碰壁则方向倒转，避免智能体一直接收相同的环境信息，但state不改变
            self.agent_dir = (self.agent_dir - 2) % 4

        self.agent_state = next_state

        pre_num_park = len(self.park_states)
        #更新车道、车位
        #path
        if self.agent_state in self.park_states:
            self.park_states.remove(self.agent_state)
        if self.agent_state not in self.path_states:
            self.path_states.append(self.agent_state)
        
        #park
        
        parks = []
        left_state = self.__nextState(self.agent_state,(self.agent_dir+1)%4)
        if left_state != self.agent_state:
            parks.append(left_state)
        right_state = self.__nextState(self.agent_state,(self.agent_dir-1)%4)
        if right_state != self.agent_state:
           parks.append(right_state)
        if pre_state != self.agent_state:
            left_state = self.__nextState(pre_state,(self.agent_dir+1)%4)
            if left_state != self.agent_state:
                parks.append(left_state)
            right_state = self.__nextState(pre_state,(self.agent_dir-1)%4)
            if right_state != self.agent_state:
                parks.append(right_state)
        
        for s in parks:
             if s not in self.path_states and s not in self.park_states:
                self.park_states.append(s)
            
        if self.render_mode == "human":
            self.__render_frame()

        park_delta = len(self.park_states) - pre_num_park

        #calculate reward
        reward = park_delta*random.uniform(0.5,1.5) - 0.05 + (next_state in self.entrances_states)*5

        truncated = self.step_count >= self.max_step
        terminated = truncated or self.__allEntrancesReached()
        obs = self.observe()
        return obs,reward,terminated,truncated,{"env_state":"step"}

    def render(self):
        if self.render_mode == "human":
            self.__render_frame()
        else:
            return self.__render_frame()
        

    
    def observe(self) -> np.ndarray:
        '''获取当前环境信息,返回给智能体'''
        #由当前智能体方向决定三维矩阵
        #矩阵大小为5*vision_range*vision_range(未定义、车位、通道、障碍物(边界）、出入口)
        #矩阵中心为当前智能体位置
        #每个纬度代表一个状态的集合

        #根据当前环境状态获得全范围的矩阵
        obs_total = np.zeros((5,self.nrow,self.ncol))

        #遍历所有状态
        for s in self.all_states:
            #park
            coord = self.stateToCoord(s)
            if s in self.park_states:
                obs_total[1,coord[0],coord[1]] = 1
            #path
            elif s in self.path_states:
                obs_total[2,coord[0],coord[1]] = 1
            #障碍物
            elif s in self.disabled_states:
                obs_total[3,coord[0],coord[1]] = 1
            #出入口
            elif s in self.entrances_states:
                obs_total[4,coord[0],coord[1]] = 1
            else:
                obs_total[0,coord[0],coord[1]] = 1

        self.obs_total = obs_total

        #根据智能体当前行进方向旋转矩阵
        # for i in range(len(obs_total)):
        #     cut = obs_total[i,:,:]
        #     if self.agent_dir == 1:
        #         cut = np.rot90(cut)
        #     elif self.agent_dir == 2:
        #         cut = np.rot90(cut,k=2)
        #     elif self.agent_dir == 3:
        #         cut = np.rot90(cut,k=3)
        #     obs_total[i,:,:] = cut
            
        #获取当前智能体感知范围
        center = self.stateToCoord(self.agent_state)
        #create empty numpy array
        self.obs = np.zeros((5,self.vision_range,self.vision_range))
        #extract submatrix
        for i in range(5):
            submatrix = self.__extract_submatrix(obs_total[i,:,:],center,self.vision_range)
            if i == 3:
                #障碍物视域，替换所有-1为1
                submatrix[submatrix == -1] = 1
            else:
                #其他视域，替换所有-1为0
                submatrix[submatrix == -1] = 0
            self.obs[i,:,:] = submatrix
        #上下翻转self.obs
        for i in range(5):
            self.obs[i,:,:] = np.flip(self.obs[i,:,:],axis=0)
        return self.obs
            

        
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> afdad09 (n)
    def show_observation(self):
        plt.close()
        fig,axes = plt.subplots(1,5,figsize=(15,3))
        for i in range(5):
            if i == 0:
                axes[i].set_title("undefined")
            elif i == 1:
                axes[i].set_title("park")
            elif i == 2:
                axes[i].set_title("path")
            elif i == 3:
                axes[i].set_title("disabled")
            elif i == 4:
                axes[i].set_title("entrance")
            array = self.obs[i,:,:]
            array[len(array)//2,len(array[0])//2] = 2
            axes[i].imshow(array)
            axes[i].axis('off')
        plt.tight_layout()
        
        plt.show()
        plt.pause(0.01)
<<<<<<< HEAD
>>>>>>> afdad09 (n)
=======
>>>>>>> afdad09 (n)
        


            


    ############################ helper method ############################
    def coordToState(self,coord):
        x,y = coord
        return int(y*self.ncol+x)
    
    def stateToCoord(self,state):
        
        x = state%self.nrow
        y = state//self.nrow
        return (x,y)
    
    def isStateActive(self, state):
        return state not in self.disabled_states
    
    def __getStateAccess(self,state):
        '''获取当前状态的可达方向'''
        x,y = self.stateToCoord(state)

        accessible = [True,True,True,True] #上、左、下、右
        if x-1 < 0 or state-1 in self.disabled_states:
            accessible[1] = False
        if x + 1 > self.ncol-1 or state + 1 in self.disabled_states:
            accessible[3] = False
        if y-1<0 or state - self.ncol in self.disabled_states:
            accessible[2] = False
        if y+1>self.nrow-1 or state + self.ncol in self.disabled_states:
            accessible[0] = False

        return accessible
    
    def __isLegalMove(self,state,dir) -> bool:
        '''判断当前行进方向是否合理'''
        x,y = self.stateToCoord(state)

        if dir == 0 and (y + 1 > self.nrow-1 or state + self.ncol in self.disabled_states):
            return False
        elif dir == 1 and (x - 1 < 0 or state - 1 in self.disabled_states):
            return False
        elif dir == 2 and ( y - 1 < 0 or state - self.ncol in self.disabled_states):
            return False
        elif dir == 3 and (x + 1 > self.ncol-1 or state + 1 in self.disabled_states):
            return False
        
        return True
    

    def __nextState(self,state,dir):
        '''根据当前状态和行进方向，计算下一个状态'''
        if self.__isLegalMove(state,dir) == False:
            return state
        if dir == 0:
            return state + self.ncol
        elif dir == 1:
            return state - 1
        elif dir == 2:
            return state - self.ncol
        elif dir == 3:
            return state + 1

    
    def __allEntrancesReached(self):
        '''判断所有出入口是否都被访问过'''
        reached = True
        for e in self.entrances_states:
            if e not in self.path_states:
                reached = False
                break
        return reached
    
    def __extract_submatrix(self,matrix:np.ndarray,center:tuple,vision_range:int):
        '''提取当前智能体感知范围的子矩阵,超出范围的为-1'''
        center_x,center_y = center
        # 计算子矩阵的起始和结束索引
        half_size = vision_range // 2
        start_x = center_x - half_size
        end_x = center_x + half_size + 1
        start_y = center_y - half_size
        end_y = center_y + half_size + 1
        
        # 创建一个5x5的矩阵，初始化为-1
        result = np.full((vision_range,vision_range),-1)
        
        # 计算实际可用的矩阵范围
        matrix_rows, matrix_cols = matrix.shape
        src_x_start = max(0, start_x)
        src_x_end = min(matrix_rows, end_x)
        src_y_start = max(0, start_y)
        src_y_end = min(matrix_cols, end_y)
        
        # 计算结果矩阵中需要填充的位置
        dst_x_start = max(0, -start_x)
        dst_x_end = vision_range - max(0, end_x - matrix_rows)
        dst_y_start = max(0, -start_y)
        dst_y_end = vision_range - max(0, end_y - matrix_cols)
        
        # 复制矩阵中有效的部分到结果矩阵
        result[dst_x_start:dst_x_end, dst_y_start:dst_y_end] = \
            matrix[src_x_start:src_x_end, src_y_start:src_y_end]
        
        return result

    def __render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size_per_block*self.ncol,self.window_size_per_block*self.nrow))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        #创建画布
        canvas = pygame.Surface((self.window_size_per_block*self.ncol,self.window_size_per_block*self.nrow))
        canvas.fill((255,255,255))
        csMangr = ColorScaleManager()

        
        
        #绘制车位
        park_color = csMangr.getColor("park_color") 
        for s in self.park_states:
            coord = self.stateToCoord(s)
            self.__draw_rec(canvas,coord,park_color)

        #绘制通道
        path_color = csMangr.getColor("path_color")
        for s in self.path_states:
            coord = self.stateToCoord(s)
            self.__draw_rec(canvas,coord,path_color)

        #绘制出入口
        entrance_color = csMangr.getColor("entrance_color")
        for s in self.entrances_states:
            coord = self.stateToCoord(s)
            self.__draw_rec(canvas,coord,entrance_color)

        #绘制障碍物
        disabled_color = csMangr.getColor("disabled_color")
        for s in self.disabled_states:
            coord = self.stateToCoord(s)
            self.__draw_rec(canvas,coord,disabled_color)

        #绘制智能体
        robot_coord = self.stateToCoord(self.agent_state)
        agent_color = csMangr.getColor("agent_color")
        canvas_x = robot_coord[0]*self.window_size_per_block+self.window_size_per_block//2
        canvas_y = robot_coord[1]*self.window_size_per_block+self.window_size_per_block//2
        pygame.draw.circle(canvas,
                           agent_color,
                           (canvas_x,canvas_y),
                           self.window_size_per_block//3)
        if self.agent_dir == 0:#up
            pygame.draw.line(canvas,
                             agent_color,
                             (canvas_x,
                              canvas_y),
                             (canvas_x,
                              canvas_y+self.window_size_per_block//1.5),
                             self.window_size_per_block//6)
        elif self.agent_dir == 1:#left
            pygame.draw.line(canvas,
                             agent_color,
                             (canvas_x,
                              canvas_y),
                             (canvas_x-self.window_size_per_block//1.5,
                              canvas_y),
                             self.window_size_per_block//6)
        elif self.agent_dir == 2:#down
            pygame.draw.line(canvas,
                             agent_color,
                             (canvas_x,
                              canvas_y),
                             (canvas_x,
                              canvas_y-self.window_size_per_block//1.5),
                             self.window_size_per_block//6)
        elif self.agent_dir == 3:#right
            pygame.draw.line(canvas,
                             agent_color,
                             (canvas_x,
                              canvas_y),
                             (canvas_x+self.window_size_per_block//1.5,
                              canvas_y),
                             self.window_size_per_block//6)
        
        #绘制当前智能体感知范围
        vision_color = (0,0,0)
        coord = self.stateToCoord(self.agent_state)
        rec_origin_x = (coord[0]-self.vision_range/2 +0.5)*self.window_size_per_block
        rec_origin_y = (coord[1]-self.vision_range/2 +0.5)*self.window_size_per_block
        pygame.draw.rect(canvas,
                         vision_color,
                         (rec_origin_x,
                          rec_origin_y,
                          self.window_size_per_block*self.vision_range,
                          self.window_size_per_block*self.vision_range),
                         2)
        
        #flip the canvas
        canvas = pygame.transform.flip(canvas,False,True)
        if self.render_mode == "human":
            self.window.blit(canvas,canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(60)
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)),axes=(1,0,2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        
        

    def __draw_rec(self,canvas,coord,color):
        x,y = coord
        pygame.draw.rect(canvas,
                         color,
                         (x*self.window_size_per_block,
                          y*self.window_size_per_block,
                          self.window_size_per_block,
                          self.window_size_per_block))
        
   
        
class ColorScaleManager:

    _instance = None

    def __new__(cls,*args,**kwargs):
        if cls._instance is None:
            cls._instance = super(ColorScaleManager, cls).__new__(cls,*args,**kwargs)
        return cls._instance

    def __init__(self) -> None:
        self.color_dict = {'agent_color':"rgb(178,19,9)",#智能体当前所处位置颜色
                           'undefined_color': "rgb(246,248,234)",
                           'entrance_color' : "rgb(156,155,151)",
                           'path_color' : "rgb(241,199,135)",
                           'park_color' : "rgb(86,145,170)"}
        self.key_list = list(self.color_dict.keys())

    def getColor(self,color_name):
        '''根据颜色名称返回颜色值'''
        if color_name not in self.key_list:
            return (0,0,0)
        
        value =  self.color_dict[color_name]
        value = value.lstrip("rgb(").rstrip(")")
        value = value.split(",")
        return tuple(int(v) for v in value)
    
    def getColorValue(self,color_name):
        '''根据颜色名称返回0-1范围内对应颜色的起始值、中间值和结束值'''
        
        if color_name not in self.key_list:
            return (0,0,0)
        
        index = self.key_list.index(color_name)
        
        return (index/len(self.color_dict),
                (index + 0.5)/len(self.color_dict),
                (index + 1)/len(self.color_dict))

    def getColorScale(self):
        '''返回plotly读取的离散色卡'''
        scale = []
        for i,n in enumerate(self.key_list):
            value = self.getColorValue(n)
            scale.append([value[0],self.color_dict[n]])
            scale.append([value[-1],self.color_dict[n]])
        
        return scale