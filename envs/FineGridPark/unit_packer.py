import numpy as np
import pandas as pd

class UnitPacker:
    '''
    将精细网格矩阵采样为特定大小步长为1的子矩阵,得到状态空间
    '''
    def __init__(self,box_size:int):
        assert box_size%2==0, "box_size cannot be odd"
        self.box_size = box_size
        self.stride = 1


    def convert(self,init_matrix:np.ndarray):
        '''
        0:空地 1:障碍物 2:出入口(解析后为该数值,原始数值为e0,e1,e2,e3,标明出入口位置以及方向）
        '''
        #添加外围轮廓
        self.matrix = np.pad(init_matrix,((self.box_size-1,self.box_size-1),(self.box_size-1,self.box_size-1)),mode='constant',constant_values=1)

        max_start_row = len(self.matrix)-self.box_size+1
        max_start_col = len(self.matrix[0])-self.box_size+1
        self.units = []
        for i in range(0,max_start_row,self.stride):
            for j in range(0,max_start_col,self.stride):
                state = i*max_start_col+j
                unit = Unit(self.box_size)
                unit.read_data(self.matrix[i:i+self.box_size,j:j+self.box_size],(i,j),state)
                self.units.append(unit)

        #find neibors
        for unit in self.units:
            unit.find_neibors(self.units)
        return self.matrix, self.units
    


class Unit:

    def __init__(self,box_size:int):
        
        self.box_size = box_size
        self.neibors = [None,None,None,None] #0:up, 1:left, 2:down, 3:right
        self.is_entrance = False
        self.entrance_dir = -1
        self.is_lane = False
        self.is_narrow_lane = False
        self.is_park = False #park会被其他为lane的单元覆盖 

    def read_data(self,matrix:np.ndarray,location:tuple,state:int):
        '''
        读取单元状态,matrix为精细网格局部矩阵,location为单元在精细网格中的位置
        '''
        assert matrix.shape == (self.box_size,self.box_size), "matrix shape must be equal to box_size"
        self.condition = matrix
        self.state = state
        self.location = location

        coords = []
        for i in range(self.box_size):
            for j in range(self.box_size):
                value = self.condition[i,j]
                if type(value) == str:  
                    if 'e' in value:
                        coords.append((i,j))

        if len(coords) == self.box_size:
            direction = int(self.condition[coords[0][0],coords[0][1]][-1])
            if direction == 0 and coords[0][1] == 0:
                self.is_entrance = True
                self.entrance_dir = 0
            elif direction == 1 and coords[0][0] == 0:
                self.is_entrance = True
                self.entrance_dir = 1
            elif direction == 2 and coords[0][1] == self.box_size-1:
                self.is_entrance = True
                self.entrance_dir = 2
            elif direction == 3 and coords[0][0] == self.box_size-1:
                self.is_entrance = True
                self.entrance_dir = 3

        if self.is_entrance:
            self.condition[:] = 2

    def move(self,direction:int) -> 'Unit':
        neibor = self.neibors[direction]
        next_unit = None
        if neibor is None:
            return None
        if neibor.is_entrance:
            if (neibor.entrance_dir + 2) % 4 == direction:
                next_unit = neibor
            else:
                return None
        else:
            condition = neibor.__get_condition(direction)
            if 2 in condition:
                #寻找出入口unit
                en_unit = None
                next_unit = self.neibors[direction]
                for i in range(self.box_size):
                    if next_unit.is_entrance:
                        en_unit = next_unit
                        break
                    else:
                        next_unit = next_unit.neibors[direction]
                        if next_unit is None:
                            break
                if en_unit is not None:
                    #只能相向入口方向行驶
                    if en_unit.entrance_dir == (direction+2)%4:
                        next_unit = neibor
                    else:
                        return None
                else:
                    return None

            #宽度不小于3/2个车头也可通行，需标记为窄车道，当网格划分不精确时，可能会出现网格被一小截障碍物占用也会被标为障碍物的情况
            if 1 in condition:
                indexs = []
                for i in range(self.box_size):
                    if condition[i] == 1:
                        indexs.append(i)
                if len(indexs) <= self.box_size/4:
                    #at side?
                    atSide = True
                    for i in range(len(indexs)):
                        if indexs[i] >self.box_size/4 or indexs[i] < self.box_size-self.box_size/4:
                            atSide = False
                            break
                    if atSide:
                        self.is_narrow_lane = True
                        next_unit = neibor
                        next_unit.is_narrow_lane = True
                    
                else:
                    return self.neibors[direction]
            else:
                next_unit = neibor
            self.make_park()
            if next_unit is not None:
                self.is_lane = True
                next_unit.is_lane = True
                next_unit.make_park()

            return next_unit
        
    def make_park(self):
        for unit in self.neibors:
            if unit is not None:
                unit.is_park = True

    def find_neibors(self,units:list):
        for unit in units:
            if unit.location == (self.location[0],self.location[1]-1):
                self.neibors[0] = unit
            elif unit.location == (self.location[0]-1,self.location[1]):
                self.neibors[1] = unit
            elif unit.location == (self.location[0],self.location[1]+1):
                self.neibors[2] = unit
            elif unit.location == (self.location[0]+1,self.location[1]):
                self.neibors[3] = unit

    def __get_condition(self,direction:int):
        if direction == 1:
            #返回最左侧一列数值
            return self.condition[:,0]
        elif direction == 0:
            #返回最上侧一列数值
            return self.condition[0,:]
        elif direction == 3:
            #返回最右侧一列数值
            return self.condition[:,-1]
        elif direction == 2:
            #返回最下侧一列数值
            return self.condition[-1,:]
        else:
            raise ValueError("direction must be 0,1,2,3")
            