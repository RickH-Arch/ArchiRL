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
from park_unit import ParkUnit

SYSTEM = platform.system()
if SYSTEM == "Windows":
    plt.ion()
    fig_mgr = plt.get_current_fig_manager()
    fig_mgr.window.geometry("+0+0")
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

        self.observation_space = Box(low=-1,high=1,shape=(self.vision_range*self.vision_range *4 +  
                                                          self.vision_range*self.vision_range*2 +
                                                         1 +#当前智能体动作
                                                         1 #当前智能体方向  
                                                         ,),dtype=np.float32)
        
        self.ncol = self.units_pack.units_arr.shape[1]
        self.nrow = self.units_pack.units_arr.shape[0]
        

        self.step_count = 0
        self.max_step = self.ncol*self.nrow*self.max_step_index

        #渲染窗口
        self.window = None
        self.window_size_per_block = 36
        self.clock = None
        
    def reset(self,seed:Optional[int] = None,options:Optional[dict] = None):
        random.seed(seed)
        self.count += 1
        #set init coord
        if len(self.entrances_states) == 0:
            self.agent_state = random.choice(self.all_states)
        else:
            self.agent_state = random.choice(self.entrances_states)

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

        self.agent_state = next_state
        reach_entry = next_state in self.entrances_states and next_state not in self.path_states

        pre_park_num = self.units_pack.get_total_park_num()

        unit = self.units_pack.get_unit_byState(next_state)
        if not unit.is_lane:
            unit.turn_to_lane()

        post_park_num = self.units_pack.get_total_park_num()

        punishment = legal * -50
        reward = (post_park_num - pre_park_num)*3 - 0.5 + reach_entry*50 + punishment
        self.rewards.append(reward)

        truncated = self.step_count >= self.max_step
        terminated = truncated or self.__allEntrancesReached()

        obs = self.observe()

        if terminated:
            avg_reward = sum(self.rewards)/len(self.rewards)
            avg_reward = round(avg_reward,3)
            total_reward = sum(self.rewards)
            if not truncated:
                print(f"------所有出入口均被访问过------parking count:{post_park_num}------avg reward:{avg_reward}-------total reward:{total_reward}")
            else:
                print(f"------环境截断:{truncated}------parking count:{post_park_num}------avg reward:{avg_reward}-------total reward:{total_reward}")

            if self.save:
                #TODO:保存当前环境和模型
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
        return []
        

    def __nextState(self,state,dir):
        '''
        根据当前状态和行进方向，计算下一个状态
        '''
        unit = self.units_pack.get_unit_byState(state)
        accessible = unit.get_accessible()
        
        if dir not in accessible:
            return unit.neighbor_unit[dir].state,False
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
                scale,offset = self.__get_park_scale_and_offset(i,unit.edge_count[i])
                if unit.edge_count[i] > 0:
                    if unit.edge_state[i] == 1:
                        self.__draw_rec(canvas,coord,park_color,scale,offset)
                    elif unit.edge_state[i] == 0:
                        self.__draw_rec(canvas,coord,vacant_park_color,scale,offset)
                else:
                    self.__draw_rec(canvas,coord,boundary_color,scale,offset)

        #绘制车道
        park_color = cMgr.getColor("park_color")
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
                end_coord[1]+=0.3
            elif direction == 1:
                end_coord[0]-=0.3
            elif direction == 2:
                end_coord[1]-=0.3
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
    
    def __show_observation(self):
        '''
        显示当前观察
        '''
        pass

    def __draw_rec(self,canvas,coord,color,scale = (1,1),offset = (0,0)):
        x,y = coord
        pygame.draw.rect(canvas,
                         color,
                         (x*self.window_size_per_block + (self.window_size_per_block-self.window_size_per_block*scale[0])//2 + offset[0]*self.window_size_per_block,
                          y*self.window_size_per_block + (self.window_size_per_block-self.window_size_per_block*scale[1])//2 + offset[1]*self.window_size_per_block,
                          self.window_size_per_block*scale[0],
                          self.window_size_per_block*scale[1]))
        
    def __get_park_scale_and_offset(self,dir:int,park_count:int):
        sc = 0.3*park_count
        if park_count == 0:
            sc = 1
        if dir == 0:
            scale = (sc,0.1)
            offset = (0,-0.9)
        elif dir == 1:
            scale = (0.1,sc)
            offset = (-0.9,0)
        elif dir == 2:
            scale = (sc,0.1)
            offset = (0,0.9)
        elif dir == 3:
            scale = (0.1,sc)
            offset = (0.9,0)
        return scale,offset

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
                           'traj_color' : "rgb(250,124,69)"}
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
    
    
    
    