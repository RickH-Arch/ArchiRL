import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
from typing import Optional
import pygame
import matplotlib.pyplot as plt
import platform
import torch

import os
from PIL import Image

import math


SYSTEM = platform.system()
#if SYSTEM == "Windows":
#    plt.ion()
#     fig_mgr = plt.get_current_fig_manager()
#     fig_mgr.window.geometry("+0+0")
import datetime
TIME_NOW = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

class AdvancePark(gym.Env):
    '''
    以边为单位的停车场环境
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
        '''
        config: 环境配置
        '''
        self.config = config
        self.units_pack = config.get("units_pack",[])
        self.max_step_index = config.get("max_step_index",1.5)

        self.vision_range = config.get("vision_range",7)
        assert self.vision_range % 2 == 1, "vision_range must be odd"
        self.render_mode = config.get("render_mode", "rgb_array")
        self.save = config.get("save", True)

        self.all_states = list(range(self.units_pack.units_arr.size))
        
        self.entrance_states = []
        self.entrance_units = []    
        for unit in self.units_pack.get_flatten_units():
            if unit.is_entrance:
                self.entrance_states.append(unit.state)
                self.entrance_units.append(unit)
        
        self.agent_state = 0 #当前坐标状态,仅用于环境判断位置
        self.agent_dir = -1 #当前行进方向,0:上,1:左,2:下,3:右
        
        self.action_space = Discrete(4) #动作空间，0：向前，1：向后，2：向左，3：向右

        #461
        # self.observation_space = Box(low=0,high=4,shape=(self.vision_range*self.vision_range *4 +  #周边的停车数量信息，4 -> 4条边  
        #                                                   self.vision_range*self.vision_range*4 +  #周边的停车状态信息（是否已是停车位）
        #                                                   self.vision_range*self.vision_range*1 +  #周边单元信息， -1:车道， 0:未定义， 1: 出入口
        #                                                  1*10 +#当前智能体动作
        #                                                  1*10 #当前智能体方向  
        #                                                  ,),dtype=np.float32)
        
        #265
        self.observation_space = Box(low=-4,high=4,shape=(self.vision_range*self.vision_range *4 +  #周边的停车数量信息，4 -> 4条边 ,已是停车位为负，未定义停车位为正
                                                          self.vision_range*self.vision_range*1 +  #周边单元信息， -1:车道， 0:未定义， 1: 出入口
                                                          1*10 +#当前智能体动作
                                                         1*10 #当前智能体方向  
                                                         ,),dtype=np.float32)
        
        self.ncol = self.units_pack.units_arr.shape[1]
        self.nrow = self.units_pack.units_arr.shape[0]
        

        self.step_count = 0
        self.max_step = self.ncol*self.nrow*self.max_step_index
        self.iter_count = 0

        #渲染窗口
        self.window = None
        self.window_size_per_block = 64
        self.clock = None

        #提前计算停车数量的全范围矩阵，避免在每个step重复计算
        self.park_num_matrix = np.zeros((4,self.nrow,self.ncol))
        
        for unit in self.units_pack.get_flatten_units():
            coord = unit.coord
            for i in range(4):
                self.park_num_matrix[i,coord[1],coord[0]] = unit.edge_carNum[i]

        self.model = None
        self.epoch_count = 0

        self.best_avg_reward = -np.inf
        self.max_park_num = 0
                



    def reset(self,seed:Optional[int] = None,options:Optional[dict] = None):
        random.seed(seed)
        self.iter_count += 1
        #set init coord
        if len(self.entrance_states) == 0:
            self.agent_state = random.choice(self.all_states)
        else:
            #self.agent_state = random.choice(self.entrance_states)
            self.agent_state = self.entrance_states[0]

        
        self.start_state = self.agent_state
        self.agent_dir = -1

        self.units_pack.reset()

        self.step_count = 0
        self.rewards = []
        self.traj = {}

        #确定智能体的初始方向
        accessible_edges = self.units_pack.get_unit_byState(self.agent_state).get_accessible()
        if len(accessible_edges) == 0:
            raise ValueError("no accessible edges")
        if len(accessible_edges) == 3:
            un_accessible_edge = [i for i in range(4) if i not in accessible_edges][0]
            self.agent_dir = (un_accessible_edge + 2) % 4
        else:
            self.agent_dir = random.choice(accessible_edges)

        #print(f"智能体初始方向：{self.agent_dir}")

        self.action = 0

        obs =  self.observe()
        return obs,{}
    
    def step(self, action):
        assert action in self.action_space, f'Invalid action: {action}'

        self.step_count += 1
        pre_state = self.agent_state
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

        next_state,legal = self.__nextState(self.agent_state,self.agent_dir)
        #record trajectory
        traj = f"{self.agent_state}->{self.agent_dir}"
        if traj in self.traj:
            self.traj[traj] += 1
        else:
            self.traj[traj] = 1
        #print(f"next_state:{next_state},legal:{legal}")

        reach_entry = False
        if legal:
            self.agent_state = next_state

            unit = self.units_pack.get_unit_byState(self.agent_state)
            reach_entry = unit.is_entrance and not unit.is_lane

        pre_park_num = self.units_pack.get_total_park_num()

        unit = self.units_pack.get_unit_byState( self.agent_state)
        if not unit.is_lane:
            unit.turn_to_lane()

        post_park_num = self.units_pack.get_total_park_num()
        self.park_num = post_park_num

        punishment = (1-legal) * -50
        #左右转、断头路惩罚
        if action == 1 or action == 3:
            punishment += 5
        elif action == 2:
            punishment += 10

        reward = (post_park_num - pre_park_num)*3 - 0.5 + reach_entry*50 + punishment
        
        self.rewards.append(reward)

        truncated = self.step_count >= self.max_step
        terminated = truncated or self.__allEntrancesReached()

        obs = self.observe()

        if terminated:
            try:
                self.epoch_count += 1
                sum_reward = sum(self.rewards)
                
                avg_reward = sum(self.rewards)/len(self.rewards)

                avg_reward = round(avg_reward,2)
                total_reward = sum(self.rewards)
                if not truncated:
                    print(f"------正常结束ep:{self.epoch_count}|step:{self.step_count}------parking count:{post_park_num}------avg reward:{avg_reward}-------total reward:{total_reward}")
                else:
                    print(f"------环境截断ep:{self.epoch_count}|step:{self.step_count}------parking count:{post_park_num}------avg reward:{avg_reward}-------total reward:{total_reward}")

                if self.save:
                    
                            
                    if avg_reward > self.best_avg_reward :
                        self.best_avg_reward = avg_reward
                        #self.__save_model()
                        self.__save_img()
                    elif post_park_num >= self.max_park_num:
                        self.max_park_num = post_park_num
                        #self.__save_model()
                        self.__save_img()
                    elif self.epoch_count % 100 == 0:
                        self.__save_img()
            except Exception as e:
                pass
                

        return obs,reward,terminated,truncated,{"env_state":"step"}
    
    def render(self):
        if self.render_mode == "human":
            self.__render_frame()
            self.__show_observation()
        else:
            return self.__render_frame()
        

    def observe(self):
        '''
        观察当前状态
        '''
        #获得全范围矩阵
        obs_state = np.zeros((4,self.nrow,self.ncol))
        obs_lane = np.zeros((1,self.nrow,self.ncol))

        #遍历所有状态
        for unit in self.units_pack.get_flatten_units():
            coord = unit.coord
            if unit.is_lane:
                obs_lane[0,coord[1],coord[0]] = 1
            for i in range(4):
                if unit.edge_state[i] == 1:
                    obs_state[i,coord[1],coord[0]] = 1

        

        #如obs_state == 1，则park_num_matrix取负
        p_now = np.copy(self.park_num_matrix)
        for i in range(4):
            for j in range(self.nrow):
                for k in range(self.ncol):
                    if obs_state[i,j,k] == 1:
                        p_now[i,j,k] = -p_now[i,j,k]

        #根据智能体方向切换p_now顺序
        if self.agent_dir == 1:
            p_now = np.roll(p_now,-1,axis=0)
        elif self.agent_dir == 2:
            p_now = np.roll(p_now,-2,axis=0)
        elif self.agent_dir == 3:
            p_now = np.roll(p_now,-3,axis=0)
                
        obs_total = np.concatenate((p_now,obs_lane),axis=0)

        center = self.units_pack.get_unit_byState(self.agent_state).coord
        self.obs = np.zeros((len(obs_total),self.vision_range,self.vision_range))
        for i in range(len(obs_total)):
            sub_mat = self.__extract_submatrix(obs_total[i],center,self.vision_range)
            sub_mat[sub_mat == -1] = 0
            self.obs[i] = sub_mat

        for i in range(len(self.obs)):
            cut = self.obs[i,:,:]
            if self.agent_dir == 1:
                cut = np.rot90(cut,k=3)
            elif self.agent_dir == 2:
                cut = np.rot90(cut,k=2)
            elif self.agent_dir == 3:
                cut = np.rot90(cut,k=1)
            self.obs[i,:,:] = cut

        result = self.obs
        result = np.transpose(result,(1,2,0))
        result = result.reshape(-1)

        #add action and direction
        action = np.array([self.action/4]*10)
        direction = np.array([self.agent_dir]*10)

        result = np.concatenate((result,action,direction),axis=0)
        result = result.astype(np.float32)
        return result
    
    def __show_observation(self):
        '''
        显示当前观察
        '''
        plt.close()
        
        #lane
        lane_mat = self.obs[-1,:,:]
        
        plt.imshow(lane_mat,cmap="gray")

        car_num_up = self.obs[0,:,:]
        car_num_left = self.obs[1,:,:]
        car_num_down = self.obs[2,:,:]
        car_num_right = self.obs[3,:,:]

        
        for i in range(car_num_up.shape[0]):
            for j in range(car_num_up.shape[1]):
                if car_num_up[i,j] < 0:
                    plt.text(j,i-0.35,f"{car_num_up[i,j]}",ha="center",va="center",color="green",fontsize=6)
                else:
                    plt.text(j,i-0.35,f"{car_num_up[i,j]}",ha="center",va="center",color="red",fontsize=6)
                if car_num_down[i,j] < 0:
                    plt.text(j,i+0.35,f"{car_num_down[i,j]}",ha="center",va="center",color="green",fontsize=6)
                else:
                    plt.text(j,i+0.35,f"{car_num_down[i,j]}",ha="center",va="center",color="red",fontsize=6)
                if car_num_left[i,j] < 0:
                    plt.text(j-0.35,i,f"{car_num_left[i,j]}",ha="center",va="center",color="green",fontsize=6,rotation=90)
                else:
                    plt.text(j-0.35,i,f"{car_num_left[i,j]}",ha="center",va="center",color="red",fontsize=6,rotation=90)
                if car_num_right[i,j] < 0:
                    plt.text(j+0.35,i,f"{car_num_right[i,j]}",ha="center",va="center",color="green",fontsize=6,rotation=-90)
                else:
                    plt.text(j+0.35,i,f"{car_num_right[i,j]}",ha="center",va="center",color="red",fontsize=6,rotation=-90)

        
        plt.show()

    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        

    def __nextState(self,state,dir):
        '''
        根据当前状态和行进方向，计算下一个状态
        '''
        unit = self.units_pack.get_unit_byState(state)
        accessible = unit.get_accessible()
        
        if dir not in accessible:
            if unit.neighbor_unit[dir] is not None:
                return unit.neighbor_unit[dir].state,False
            else:
                return state,False
        else:
            return unit.neighbor_unit[dir].state,True
        
    def __allEntrancesReached(self):
        '''
        判断所有入口是否都被访问过
        '''
        for unit in self.entrance_units:
            if not unit.is_lane:
                return False
        return True

    def __render_frame(self):
        '''
        渲染当前环境
        '''
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size_per_block*self.ncol,self.window_size_per_block*self.nrow))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        #创建画布
        canvas = pygame.Surface((self.window_size_per_block*self.ncol,self.window_size_per_block*self.nrow))
        canvas.fill((255,255,255))
        cMgr = ColorManager()

        #绘制车位
        vacant_park_color = cMgr.getColor("vacant_park_color")
        park_color = cMgr.getColor("park_color")
        boundary_color = cMgr.getColor("boundary_color")
        for unit in self.units_pack.get_flatten_units():
            if unit.is_lane:
                continue
            coord = unit.coord
            for i in range(4):
                scale,offset = self.__get_park_scale_and_offset(i,unit.edge_carNum[i])
                if unit.edge_carNum[i] > 0:
                    if unit.edge_state[i] == 1:
                        self.__draw_rec(canvas,coord,park_color,scale,offset)
                    elif unit.edge_state[i] == 0:
                        self.__draw_rec(canvas,coord,vacant_park_color,scale,offset)
                else:
                    self.__draw_rec(canvas,coord,boundary_color,scale,offset)

        #绘制车道
        park_color = cMgr.getColor("path_color")
        for unit in self.units_pack.get_flatten_units():
            if unit.is_lane:
                coord = unit.coord
                self.__draw_rec(canvas,coord,park_color)

        #绘制出入口
        for unit in self.entrance_units:
            coord = unit.coord
            if unit.is_lane:
                self.__draw_rec(canvas,coord,cMgr.getColor("reached_entrance_color"))
            else:
                self.__draw_rec(canvas,coord,cMgr.getColor("entrance_color"))

        #绘制轨迹
        if len(self.traj) > 0:
            traj_color = cMgr.getColor("traj_color")
            max_count = max(self.traj.values())
            min_count = min(self.traj.values())
            if max_count == min_count:
                for t in self.traj:
                    self.traj[t] = 1
            else:
                for t in self.traj:
                    self.traj[t] = ((self.traj[t] - min_count) / (max_count - min_count))*5+1

            #self.traj = {list(self.traj.keys())[0]:self.traj[list(self.traj.keys())[0]]}
            for key in self.traj.keys():
                start_coord = self.stateToCoord(int(key.split("->")[0]))
                direction = int(key.split("->")[1])
                end_coord = start_coord
                #tuple to list
                end_coord = list(end_coord)

                if direction == 0:
                    end_coord[1]-=0.3
                elif direction == 1:
                    end_coord[0]-=0.3
                elif direction == 2:
                    end_coord[1]+=0.3
                elif direction == 3:
                    end_coord[0]+=0.3
                pygame.draw.line(canvas,traj_color,(start_coord[0]*self.window_size_per_block+self.window_size_per_block//2,
                                                    start_coord[1]*self.window_size_per_block+self.window_size_per_block//2),
                                                    (end_coord[0]*self.window_size_per_block+self.window_size_per_block//2,
                                                    end_coord[1]*self.window_size_per_block+self.window_size_per_block//2),
                                                    int(self.traj[key]))

        #绘制智能体
        unit = self.units_pack.get_unit_byState(self.agent_state)
        robot_coord = unit.coord
        agent_color = cMgr.getColor("agent_color")
        canvas_x = robot_coord[0]*self.window_size_per_block+self.window_size_per_block//2
        canvas_y = robot_coord[1]*self.window_size_per_block+self.window_size_per_block//2
        pygame.draw.circle(canvas,
                           agent_color,
                           (canvas_x,canvas_y),
                           self.window_size_per_block//3)
        if self.agent_dir == 2:#down
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
        elif self.agent_dir == 0:#up
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
        vision_color = (246,248,234)
        coord = robot_coord
        rec_origin_x = (coord[0]-self.vision_range/2 +0.5)*self.window_size_per_block
        rec_origin_y = (coord[1]-self.vision_range/2 +0.5)*self.window_size_per_block
        pygame.draw.rect(canvas,
                         vision_color,
                         (rec_origin_x,
                          rec_origin_y,
                          self.window_size_per_block*self.vision_range,
                          self.window_size_per_block*self.vision_range),
                         2)
        
        #绘制网格
        mesh_color = cMgr.getColor("mesh_color")
        for i in range(self.ncol):
            pygame.draw.line(canvas,
                             mesh_color,
                             (i*self.window_size_per_block,0),
                             (i*self.window_size_per_block,self.window_size_per_block*self.nrow))
        for i in range(self.nrow):
            pygame.draw.line(canvas,
                             mesh_color,
                             (0,i*self.window_size_per_block),
                             (self.window_size_per_block*self.ncol,i*self.window_size_per_block))
        
        
        if self.render_mode == "human":
            self.window.blit(canvas,canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(6000)
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)),axes=(1,0,2))
    
    

    def __draw_rec(self,canvas,coord,color,scale = (1,1),offset = (0,0)):
        x,y = coord
        pygame.draw.rect(canvas,
                         color,
                         (x*self.window_size_per_block + 1 + (self.window_size_per_block-self.window_size_per_block*scale[0])//2 + offset[0]*self.window_size_per_block,
                          y*self.window_size_per_block + 1 + (self.window_size_per_block-self.window_size_per_block*scale[1])//2 + offset[1]*self.window_size_per_block,
                          self.window_size_per_block*scale[0],
                          self.window_size_per_block*scale[1]))
        
    def __get_park_scale_and_offset(self,dir:int,park_count:int):
        sc = 0.3*park_count
        if park_count == 0:
            sc = 1
        if dir == 0:
            scale = (sc,0.1)
            offset = (0,-0.4)
        elif dir == 1:
            scale = (0.1,sc)
            offset = (-0.4,0)
        elif dir == 2:
            scale = (sc,0.1)
            offset = (0,0.4)
        elif dir == 3:
            scale = (0.1,sc)
            offset = (0.4,0)
        return scale,offset
    
    def __extract_submatrix(self,matrix:np.ndarray,center:tuple,vision_range:int):
        '''提取当前智能体感知范围的子矩阵,超出范围的为-1'''
        center_x,center_y = center
        # 计算子矩阵的起始和结束索引
        half_size = vision_range // 2
        start_x = center_x - half_size
        end_x = center_x + half_size + 1
        start_y = center_y - half_size
        end_y = center_y + half_size + 1
        
        # 创建矩阵，初始化为-1
        result = np.full((vision_range,vision_range),-1)
        
        # 计算实际可用的矩阵范围
        matrix_rows, matrix_cols = matrix.shape
        src_x_start = max(0, start_x)
        src_x_end = min(matrix_cols, end_x)
        src_y_start = max(0, start_y)
        src_y_end = min(matrix_rows, end_y)
        
        # 计算结果矩阵中需要填充的位置
        dst_x_start = max(0, -start_x)
        dst_x_end = vision_range - max(0, end_x - matrix_cols)
        dst_y_start = max(0, -start_y)
        dst_y_end = vision_range - max(0, end_y - matrix_rows)
        
        # 复制矩阵中有效的部分到结果矩阵
        result[dst_y_start:dst_y_end,dst_x_start:dst_x_end] = \
            matrix[src_y_start:src_y_end,src_x_start:src_x_end]
        
        return result
    
    def __save_model(self):
        if self.model is not None:
            folder = f"./parking_best_model"
            if not os.path.exists(folder):
                os.makedirs(folder)

            #delete all files in the folder
            for file in os.listdir(folder):
                os.remove(os.path.join(folder,file))

            path = os.path.join(folder,f"model_epoch{self.epoch_count}_{TIME_NOW}.pth")
            
            torch.save(self.model.policy.state_dict(),path)
            print(f"模型已保存到{path}")

    def __save_img(self):
        img = self.__render_frame()
        img = Image.fromarray(img)
        folder = f"./parking_img/parking_img_{datetime.datetime.now().strftime('%Y%m%d')}"
        path = os.path.join(folder,f"epoch:{self.epoch_count}_num:{self.park_num}_avgrwd:{self.best_avg_reward}_{datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')}.png")
        #是否存在路径
        if not os.path.exists(folder):
            os.makedirs(folder)
        img.save(path)
        

    def coordToState(self,coord):
        x,y = coord
        return int(y*self.ncol+x)
    
    def stateToCoord(self,state):
        
        x = state%self.ncol
        y = state//self.ncol
        return (x,y)
    

class ColorManager:

    _instance = None

    def __new__(cls,*args,**kwargs):
        if cls._instance is None:
            cls._instance = super(ColorManager, cls).__new__(cls,*args,**kwargs)
        return cls._instance

    def __init__(self) -> None:
        self.color_dict = {'agent_color':"rgb(178,19,9)",#智能体当前所处位置颜色
                           'undefined_color': "rgb(246,248,234)",
                           'boundary_color' : "rgb(50,50,50)",
                           'entrance_color' : "rgb(156,155,151)",
                           'reached_entrance_color' : "rgb(98,184,115)",
                           'path_color' : "rgb(241,199,135)",
                           'vacant_park_color' : "rgb(190,190,190)",
                           'park_color' : "rgb(86,145,170)",
                           'traj_color' : "rgb(250,124,69)",
                           'mesh_color' : "rgb(200,200,200)"}
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
    
    
    
    