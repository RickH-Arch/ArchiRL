import pandas as pd
import numpy as np
from envs.AdvanceParkingEnv.park_unit import ParkUnit,ParkUnitMatPack

class ManualParkReader:
    def read(self,file_path:str):
        df = pd.read_csv(file_path,header=None)
        rows,cols = df.shape
        blocks = []
        for i in range(0,rows,3):
            b = []
            for j in range(0,cols,3):
                unit = df.iloc[i:i+3,j:j+3]
                b.append(unit)
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

        units_pack.get_unit_byCoord((7,12)).is_entrance = True
        units_pack.get_unit_byCoord((14,12)).is_entrance = True
        units_pack.connect_neighbor()

        print("read pack with shape:",units_pack.units_arr.shape)

        return units_pack