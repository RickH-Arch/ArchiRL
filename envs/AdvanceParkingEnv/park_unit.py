import numpy as np
import math

class ParkUnitMatPack:
    '''
    停车场单元矩阵
    '''
    def __init__(self,units:list[list]):
        '''
        units: 停车场单元矩阵
        '''
        self.units_arr = np.array(units)
        # 赋值单元状态
        for i in range(self.units_arr.shape[0]):
            for j in range(self.units_arr.shape[1]):
                self.units_arr[i][j].state = i * self.units_arr.shape[1] + j


    def get_unit_byCoord(self,coord:tuple[int,int]):
        '''
        获取单元,coord:(x,y)
        '''
        if coord[0] < 0 or coord[0] >= self.units_arr.shape[1] or coord[1] < 0 or coord[1] >= self.units_arr.shape[0]:
            raise ValueError(f"coord out of range: {coord}")
        return self.units_arr[coord[1]][coord[0]]
    
    def get_unit_byState(self,state:int):
        '''
        根据智能体状态索引获取网格单元
        '''
        if state < 0 or state >= self.units_arr.size:
            raise ValueError(f"state out of range: {state}")
        coord = np.unravel_index(state,self.units_arr.shape)
        return self.units_arr[coord[0]][coord[1]]
    
    def get_flatten_units(self):
        '''
        获取单元矩阵
        '''
        return self.units_arr.flatten()
    
    
    def reset(self):
        '''
        重置单元矩阵
        '''
        for unit in self.get_flatten_units():
            unit.reset()
    
    def get_total_park_num(self):
        '''
        获取总停车位数
        '''
        num = 0
        for unit in self.get_flatten_units():
            if unit.is_lane:
                continue
            num += unit.get_car_num()
        return num
    
    def connect_neighbor(self):
        '''
        连接相邻单元
        '''
        for unit in self.get_flatten_units():
            unit.find_neighbor(self.get_flatten_units())

class ParkUnit:
    '''
    用于车位排布的网格单元，以边为单位记录状态
    边索引: 0123 -> 上左下右
             0
        +---------+
        |         |
     1  |         |  3
        |         |
        +---------+
             2
    '''
    def __init__(self, edge_carNum:list[int],coord:tuple[int,int]):
        '''
        edge_carNum: 边可通行车辆数
        coord: 单元坐标
        '''
        self.edge_carNum = edge_carNum
        self.edge_state = [0] * 4 # 0: 空闲，1: 占用
        self.is_lane = False # 是否为车道
        self.is_entrance = False # 是否为入口

        self.edge_width = [-1] * 4 # 边可通行宽度
        self.neighbor_unit = [None] * 4 # 相邻单元索引

        self.coord = coord #（ x,y ）(col,row) 对应平面索引：[y][x]
        self.state = -1 # 单元对应的智能体状态
        
    def find_neighbor(self,units:list):
        '''
        找到相邻单元
        '''
        for unit in units:
            if unit.coord == self.coord:
                continue
            #上
            if unit.coord[0] == self.coord[0] and unit.coord[1] == self.coord[1] - 1:
                self.neighbor_unit[0] = unit
            #下
            if unit.coord[0] == self.coord[0] and unit.coord[1] == self.coord[1] + 1:
                self.neighbor_unit[2] = unit
            #左
            if unit.coord[0] == self.coord[0] - 1 and unit.coord[1] == self.coord[1]:
                self.neighbor_unit[1] = unit
            #右
            if unit.coord[0] == self.coord[0] + 1 and unit.coord[1] == self.coord[1]:
                self.neighbor_unit[3] = unit

            if None not in self.neighbor_unit:
                break

    def reset(self):
        '''
        重置单元
        '''
        self.is_lane = False
        self.edge_state = [0] * 4
        
    def get_accessible(self):
        '''
        获取可通行边
        '''
        accessible = []
        for i,car_num in enumerate(self.edge_carNum):
            if car_num>=2:
                neib = self.neighbor_unit[i]
                if neib is None:
                    continue
                if i == 0:
                    if neib.edge_carNum[2] < 2:
                        continue
                elif i == 1:
                    if neib.edge_carNum[3] < 2:
                        continue
                elif i == 2:
                    if neib.edge_carNum[0] < 2:
                        continue
                elif i == 3:
                    if neib.edge_carNum[1] < 2:
                        continue
                accessible.append(i)
        return accessible
    
    def turn_to_lane(self):
        '''
        转换为车道
        '''
        self.is_lane = True
        self.edge_state = [0] * 4
        for i in range(4):
            if self.neighbor_unit[i] is None or self.neighbor_unit[i].is_lane or self.neighbor_unit[i].is_entrance:
                continue
            if i == 0:
                self.neighbor_unit[i].edge_state[2] = 1
            elif i == 1:
                self.neighbor_unit[i].edge_state[3] = 1
            elif i == 2:
                self.neighbor_unit[i].edge_state[0] = 1
            elif i == 3:
                self.neighbor_unit[i].edge_state[1] = 1

    def get_car_num(self):
        '''
        获取单元内车辆数
        '''


        total = 0
        real = [0]*4
        n = 0
        ind = []
        for i in range(4):
            if self.edge_state[i] == 1:
                real[i] = self.edge_carNum[i]
                n+=1
                ind.append(i)
        
        
        if n == 0:
            total =  0
        elif n == 1:
            total = sum(real)
        elif n == 2:
            #对向停车？
            if abs(ind[0] - ind[1]) == 2:
                total = real[ind[0]] + real[ind[1]]
            #相邻两边停车？
            else:
                total = max(real[ind[0]],real[ind[1]])+1
        elif n == 3:
            #find the missing edge
            me = -1
            for i in range(4):
                if self.edge_state[i] != 1:
                    me = i
                    break
            if me == -1:
                total = 0
            else:
                total = self.edge_carNum[(me-1)%4] + self.edge_carNum[(me+1)%4]
        else:
            n1 = self.edge_carNum[0] + self.edge_carNum[2]
            n2 = self.edge_carNum[1] + self.edge_carNum[3]
            total = max(n1,n2)
        return total * 0.5
        
        