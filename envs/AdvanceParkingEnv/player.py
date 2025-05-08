from manual_park_reader import ManualParkReader
from advance_park import AdvancePark

file_path = 'envs/AdvanceParkingEnv/manual_park_data.csv'

reader = ManualParkReader()
units_pack = reader.read(file_path)

config = {
    
}

env = AdvancePark(units_pack)
env.reset()
env.render()
