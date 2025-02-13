import numpy as np
import plot
from plot import ColorScaleManager
import random

csMangr = ColorScaleManager()
class SimplePark:
    '''正交矩形网格空间'''
    def __init__(self,nrow,ncol,disabled_states,entrances_states) -> None:
        '''
        disabled_states: 一维int列表(state),对应网格为非激活网格\n
        entrances_states: 一维int列表(state),对应网格需位于边界且激活
        '''
        self.nrow = nrow
        self.ncol = ncol
        self.all_states = list(range(nrow*ncol))
        self.disabled_states = disabled_states

        #出入口需靠近边缘
        self.entrances_states = []
        for s in entrances_states:
            if s in disabled_states:
                continue
            
            x,y = self.stateToCoord(s)
            if (x-1<0 or x+1>ncol-1 or y-1<0 or y+1>nrow-1)\
            or (s - ncol in disabled_states)\
            or(s + ncol in disabled_states)\
            or (s -1 in disabled_states)\
            or (s+1 in disabled_states):
                self.entrances_states.append(s)
                

        #set init state
        self.agent_state = 0 #当前坐标状态,仅用于环境判断位置
        

        self.dir_space = [-1,0,1,2,3] #智能体当前行进的方向，-1：无效， 0：上，1：左，2：下，3：右
        self.agent_dir = -1 

        self.action_space = [0,1,2,3] #动作空间，0：前，1：后，2：左，3：右

        self.park_states = [] #当前是车位的坐标状态
        self.path_states = [] #当前是通道的坐标状态

        self.traj_states = [] #记录五步以内智能体经过的状态

        self.step_count = 0
        self.max_step = ncol*nrow*2

        self.reset()


    def reset(self):
        #set init coord
        if len(self.entrances_states) == 0:
            self.agent_state = random.choice(self.all_states)
        else:
            self.agent_state = random.choice(self.entrances_states)

        #set init dir
        self.agent_dir = -1
        
        accessible = self._getStateAccess(self.agent_state)
        
        accessible_dirs = [i for i,value in enumerate(accessible) if value]
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
        

    def step(self,action):

        if action not in self.action_space:
            raise Exception("非法动作")
        if self.agent_dir == -1:
            raise Exception("智能体当前行动方向异常")
        
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
        
        done = self.step_count == self.max_step

        next_state = self._nextState(self.agent_state,self.agent_dir)
        if next_state == self.agent_state:
            #如碰壁则方向倒转，避免智能体一直接收相同的环境信息，但state不改变
            self._pushTraj(self.agent_state)
            self.agent_dir = (self.agent_dir - 2) % 4
            return self._detect(), done
            
        self.agent_state = next_state
            
        
        #更新车道、车位
        #path
        if self.agent_state in self.park_states:
            self.park_states.remove(self.agent_state)
        if self.agent_state not in self.path_states:
            self.path_states.append(self.agent_state)
        
        #park
        parks = []
        left_state = self._nextState(self.agent_state,(self.agent_dir+1)%4)
        if left_state != self.agent_state:
            parks.append(left_state)
        right_state = self._nextState(self.agent_state,(self.agent_dir-1)%4)
        if right_state != self.agent_state:
           parks.append(right_state)
        if pre_state != self.agent_state:
            left_state = self._nextState(pre_state,(self.agent_dir+1)%4)
            if left_state != self.agent_state:
                parks.append(left_state)
            right_state = self._nextState(pre_state,(self.agent_dir-1)%4)
            if right_state != self.agent_state:
                parks.append(right_state)
        
        for s in parks:
             if s not in self.path_states and s not in self.park_states:
                self.park_states.append(s)

        return self._detect(),done
               
        

    def _detect(self):
        '''
        读取环境，返回智能体可感知的环境状态,该状态应与智能体行进方向相关
        '''
        return 1


    ############################ helper method ############################
    def coordToState(self,coord):
        x,y = coord
        return int(y*self.ncol+x)
    
    def stateToCoord(self,state):
        x = state%self.ncol
        y = state//self.ncol
        return (x,y)
    
    def isStateActive(self, state):
        return state not in self.disabled_states
    
    def _getStateAccess(self,state):
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
    
    def _isLegalMove(self,state,dir) -> bool:
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
    
    def _pushTraj(self,state):
        self.traj_states = self.traj_states[1:]
        self.traj_states.append(state)

    def _nextState(self,state,dir):
        if self._isLegalMove(state,dir) == False:
            return state
        if dir == 0:
            return state + self.ncol
        elif dir == 1:
            return state - 1
        elif dir == 2:
            return state - self.ncol
        elif dir == 3:
            return state + 1

    

    ############################ display method ############################
    def getGridStateNow(self):
        '''获取当前网格停车场状态'''
        grid = np.zeros((self.nrow,self.ncol))
        for s in self.all_states:
            coord = self.stateToCoord(s)

            value = 0
            if s == self.agent_state:
                value =  csMangr.getColorValue("agent_color")[1]
            else:
                if s in self.disabled_states:
                    #value = csMangr.GetColorValue("disabled_color",True)[1]
                    value = np.nan
                elif s in self.entrances_states:
                    value = csMangr.getColorValue("entrance_color")[1]
                elif s in self.path_states:
                    value = csMangr.getColorValue("path_color")[1]
                elif s in self.park_states:
                    value = csMangr.getColorValue("park_color")[1]
                else:
                    value = csMangr.getColorValue("undefined_color")[1]

            grid[coord[1],coord[0]] = value
        return grid

    
    def display(self):
        grid = self.getGridStateNow()
        plot.showPark(grid,self.agent_dir)