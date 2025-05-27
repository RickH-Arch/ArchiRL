import pandas as pd
import numpy as np
from envs.AdvanceParkingEnv.park_unit import ParkUnit,ParkUnitMatPack

class ParkReader:
    def read(self,file_path:str):
        entrance_coord = []
        oc_coord = []
        df = pd.read_csv(file_path,header=None)
        rows,cols = df.shape
        blocks = []
        for i in range(0,rows,3):
            b = []
            for j in range(0,cols,3):
                unit = df.iloc[i:i+3,j:j+3]

                #创建与unit同样大小的np矩阵
                car_num_unit = np.zeros((3,3))
                nums = [unit.iloc[0,1],unit.iloc[1,0],unit.iloc[1,2],unit.iloc[2,1]]
                
                # 将nums中字符串形式的数字转换为int形式
                nums = [int(num) if isinstance(num, str) and num.isdigit() else num for num in nums]
                #如果nums中全是None
                if not all(np.isnan(num) for num in nums):
                    if np.nan in nums:
                        raise ValueError(f"{j//3},{i//3}unit中存在None")
                    else:
                        car_num_unit[0,1] = unit.iloc[0,1]
                        car_num_unit[1,0] = unit.iloc[1,0]
                        car_num_unit[1,2] = unit.iloc[1,2]
                        car_num_unit[2,1] = unit.iloc[2,1]
                mid = unit.iloc[1,1]
                if type(mid) == str:
                    if mid == "entrance" or "entrance" in mid or mid == "en" or "en" in mid:
                        entrance_coord.append((j//3,i//3))
                    #非停车空地类型,不影响训练结果，但会在结果生成后自动剔除该区域停车位
                    if mid == "oc":
                        oc_coord.append((j//3,i//3))
                b.append(car_num_unit)
            blocks.append(b)

        blocks = np.array(blocks)

        
        units_mat = []
        for r in range(blocks.shape[0]):
            units_row = []
            for c in range(blocks.shape[1]):
                block = blocks[r][c]
                up = block[0][1]
                down = block[2][1]
                left = block[1][0]
                right = block[1][2]
                unit = ParkUnit(edge_carNum=[up,left,down,right],coord=(c,r))
                
                units_row.append(unit)
            units_mat.append(units_row)
        units_pack = ParkUnitMatPack(units_mat)

        for coord in entrance_coord:
            units_pack.get_unit_byCoord(coord).is_entrance = True
        for coord in oc_coord:
            units_pack.get_unit_byCoord(coord).is_occupied = True

        units_pack.connect_neighbor()

        print("read pack with shape:",units_pack.units_arr.shape)

        return units_pack