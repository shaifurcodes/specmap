from clustering import SpecMapClustering
import numpy as  np
import matplotlib.pyplot as plt
class InputBasedDataGenerator(object):
    '''
    tasks:
    1. read input pathloss files for individual maps
    2. generate individual maps for given tx powers
    3. generate maps for each config
    4. generate training matrix for given %-dim and sample_count
    '''
    def __init__(self, pathloss_files, tx_powers, configs, dim_ratio, sample_per_config, mapDirectory = '.'):
        '''

        :param pathloss_files:
        :param tx_powers:
        :param configs:
        :param dim_ratio:
        :param sample_per_config:
        '''
        self.pathloss_files = pathloss_files
        self.tx_powers = tx_powers
        self.configs = configs
        self.dim_ratio = dim_ratio
        self.sample_per_config = sample_per_config


        self.grid_x = self.grid_y = -1
        self.single_tx_pathloss = []
        self.maps = []
        self.noise_floor_dBm = -400.0

        self.training_data_vals = []
        self.training_data_x_indx = []
        self.training_data_y_indx = []
        self.training_data_labels = [] #index to self.all_maps

        self.mapDirectory = mapDirectory

    def loadPathlossMaps(self):
        '''
        :return:
        '''
        for fname in sorted(self.pathloss_files):
            with open(fname, 'r') as f:
                self.single_tx_pathloss.append( np.loadtxt(fname, dtype=np.float) )
        self.grid_x, self.grid_y = self.single_tx_pathloss[0].shape

    def saveMap(self, cur_map, title):
        '''

        :return:
        '''
        xi = np.arange(0, self.grid_x, 1.0)
        yi = np.arange(0, self.grid_y, 1.0)
        CS = plt.contourf(xi, yi, cur_map, 10, cmap='viridis')
        plt.colorbar()
        plt.savefig(self.mapDirectory+"/"+title + ".png")
        plt.close()

    def saveSampleMap(self, cur_map, title):
        '''

        :param cur_map:
        :return:
        '''
        xr = np.arange( 0, self.grid_x, 1.0 )
        yr = np.arange( 0, self.grid_y, 1.0 )
        x, y = np.meshgrid(xr, yr)
        plt.scatter(x, y, s=1.0, marker= "D", c=cur_map)
        plt.colorbar()
        plt.savefig(self.mapDirectory+"/"+title+'.png')
        plt.close()

    def generateMapPerConfig(self, saveMap = False):
        '''
        :return:
        '''
        for cur_config in self.configs:
            output_map_mW = np.zeros(  shape=(self.grid_x, self.grid_y) , dtype = np.float )
            for map_indx, pwr_level in enumerate(cur_config):
                pwr_indx = int(pwr_level) - 1
                if pwr_indx < 0:
                    continue
                cur_map_dBm = self.tx_powers[map_indx][pwr_indx] - self.single_tx_pathloss[map_indx]
                output_map_mW +=  np.power(10, cur_map_dBm/10.0)

            output_map_dB = 10*np.log10(output_map_mW, where = output_map_mW>0.0)
            output_map_dB[output_map_mW <=0.0] = self.noise_floor_dBm
            self.maps.append(output_map_dB)
            if saveMap:
                self.saveMap(output_map_dB, cur_config)

    def generateTrainingData(self, saveMapFig = False, mapDirectory = '.'):
        '''
        :return:
        '''
        self.training_data_vals = []
        self.training_data_x_indx = []
        self.training_data_y_indx = []
        self.training_data_labels = [] #index to self.all_maps

        self.debug_training_maps = []
        for i in self.maps:
            cur_training_map = np.empty_like(self.maps[0])
            cur_training_map[:] = np.nan
            self.debug_training_maps.append(cur_training_map)

        total_maps = len(self.maps)
        all_loc_indx = list( np.ndindex( self.grid_x, self.grid_y) )
        total_indices = len(all_loc_indx)
        indices_to_be_chosen = max(1, int(  np.round(self.dim_ratio*self.grid_x*self.grid_y)  )  )
        n_sample = self.sample_per_config * total_maps

        for i in np.arange(n_sample):
            map_indx = np.random.choice(total_maps, 1)[0]
            indx_to_locs = np.random.choice(total_indices, indices_to_be_chosen, replace = False)
            chosen_indices = sorted([all_loc_indx[cindx] for cindx in indx_to_locs])
            x_indx, y_indx = zip(*chosen_indices)

            chosen_signal_vals = self.maps[map_indx][x_indx , y_indx]
            self.debug_training_maps[map_indx][x_indx , y_indx] = self.maps[map_indx][x_indx , y_indx]

            self.training_data_labels.append(map_indx)
            self.training_data_vals.append(chosen_signal_vals)
            self.training_data_x_indx.append(x_indx)
            self.training_data_y_indx.append(y_indx)
        if saveMapFig:
            for idx, config in enumerate(self.configs):
                self.saveSampleMap( self.debug_training_maps[idx] , "sample_"+config)

    def getMaps(self):
        '''

        :return:
        '''
        return self.maps

    def getTrainingData(self, saveMap = False):
        '''

        :return:
        '''
        self.loadPathlossMaps()
        self.generateMapPerConfig(saveMap=saveMap)
        self.generateTrainingData(saveMapFig=saveMap)
        return self.training_data_x_indx, self.training_data_y_indx, self.training_data_vals, self.training_data_labels, self.grid_x, self.grid_y

if __name__ == '__main__':
    pathloss_files = ['../splat_data/pathloss_1.txt', '../splat_data/pathloss_2.txt']
    tx_powers = [
         [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0 ]
        ,[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0 ]
    ]
    configs = ['60', '06', '66']
    dim_ratio = 0.2/100.0
    sample_per_config = 100
    #----------------------------------------#
    ibdg = InputBasedDataGenerator(pathloss_files = pathloss_files,
                                   tx_powers=tx_powers,
                                   configs=configs,
                                   dim_ratio=dim_ratio,
                                   sample_per_config=sample_per_config)
    _, _, _, _ , _, _ = ibdg.getTrainingData(saveMap=False)
    # ibdg.loadPathlossMaps()
    # ibdg.generateMapPerConfig(saveMap=True)
    # print "generateing training data.."
    # ibdg.generateTrainingData(saveMapFig=True)