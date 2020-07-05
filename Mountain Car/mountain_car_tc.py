import tiles3 as tc
import numpy as np

class MountainCarTileCoder:
    def __init__(self,iht_size = 4096, num_tilings = 8, num_tiles = 8):
        self.iht = tc.IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles

    def get_tiles(self,position,velocity):
        POSITION_MIN = -1.2
        POSITION_MAX = 0.5
        VELOCITY_MIN = -0.07
        VELOCITY_MAX = 0.07

        position_scale = self.num_tiles / (POSITION_MAX-POSITION_MIN)
        velocity_scale = self.num_tiles / (VELOCITY_MAX-VELOCITY_MIN)

        tiles = tc.tiles(self.iht,self.num_tilings,[position * position_scale,
                                                    velocity * velocity_scale])
        
        return np.array(tiles)
