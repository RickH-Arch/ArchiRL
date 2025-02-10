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
        self.entrances_states = entrances_states

        #set init state
        self.coord_state = 0 #当前坐标状态,仅用于环境判断位置
        self.reset()

        self.dir_space = [-1,0,1,2,3] #智能体当前行进的方向，-1：无效， 0：上，1：下，2：左，3：右
        self.dir = -1 

        self.action_space = [0,1,2,3] #动作空间，0：前，1：后，2：左，3：右

        self.park_states = [] #当前是车位的坐标状态
        self.path_states = [] #当前是通道的坐标状态




    def reset(self):
        if len(self.entrances_states) == 0:
            self.coord_state = random.choice(self.all_states)
        else:
            self.coord_state = random.choice(self.entrances_states)

    def step(self,action):
        pass

    def detect(self):
        '''
        读取环境，返回智能体可感知的环境状态
        '''
        pass


    ############################ helper method ############################
    def CoordToState(self,coord):
        x,y = coord
        return int(y*self.width+x)
    
    def StateToCoord(self,state):
        x = state%self.width
        y = state//self.width
        return (x,y)
    
    def IsStateActive(self, state):
        return state not in self.disabled_states
    

    ############################ display method ############################
    def GeyGridStateNow(self):
        '''获取当前网格停车场状态'''
        grid = np.zeros((self.nrow,self.ncol))
        for s in self.all_states:
            coord = self.StateToCoord(s)
            value = 0
            if s in self.disabled_states:
                value = csMangr.GetColorValue("disabled_color")[1]
            #TODO: get color value

    
    def Display(self):
        plot.ShowGrid()