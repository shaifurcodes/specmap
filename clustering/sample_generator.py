import numpy as np
import itertools


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits import mplot3d

class GenerateSpectrumMap:
    '''
    Given a set of Tx-locations and Tx-powers, and params related to wireless propagation model,
    generate:
        i) all permutations(ON/OFF of transmitters) of maps
        ii) training data for clustering algorithms
    '''
    def __init__(self,
                    max_x_meter,
                    max_y_meter,
                    tx_power_dBm,
                    tx_loc,
                    configs,
                    n = 2.0,
                    lmda_meter = 0.3,
                    d_0_meter = 1.0,
                    sigma_sq_db = 5.0,
                    noise_floor_dB = -60.0
                 ):
        '''
        set the params
        '''
        self.max_x_meter = max_x_meter
        self.max_y_meter = max_y_meter
        self.tx_power_dBm = tx_power_dBm
        self.tx_loc = tx_loc
        self.n = n
        self.lmbda_meter = lmda_meter
        self.d_0_meter =  d_0_meter
        self.sigma_db = np.sqrt(sigma_sq_db)
        self.configs = configs
        self.noise_floor_dB = noise_floor_dB

        self.k_dB = 20.0 * np.log10(4.0 * np.pi * self.d_0_meter / self.lmbda_meter)
        self.total_tx = self.tx_loc.shape[0]
        #np.random.seed(1009993)

    def generateIndividualMap(self):
        '''
        generates a map for each Tx
        :return:
        '''
        self.ind_map = [[] for i in range(self.total_tx) ]

        self.spectrum_maps = []
        for i in range(self.total_tx): #TODO: change iterator to handle mulitple power levels
            tx_x, tx_y = self.tx_loc[i][0], self.tx_loc[i][1]
            x_vals = np.arange(0, self.max_x_meter+1, self.d_0_meter)
            y_vals = np.arange(0, self.max_y_meter+1, self.d_0_meter)
            x_grid, y_grid = np.meshgrid(x_vals, y_vals, sparse=False, indexing='ij')
            dist_sq_map = (  (x_grid - tx_x)**2.0 + (y_grid - tx_y)**2.0 )
            path_loss =  self.k_dB  + 5.0 *self.n * np.log10( dist_sq_map/self.d_0_meter**2.0, where = dist_sq_map > 0.0)
            path_loss[dist_sq_map <= 0.0] = 0.0
            for j, tx_pwr in enumerate(self.tx_power_dBm[i]):
                cur_map = tx_pwr - path_loss
                self.ind_map[i].append(cur_map)
                #self.displayMaps(map_list= self.ind_map[i], figFilename= './heatmaps/'+str(i)+'.png')

    def get_map_grid_x(self):
        return self.ind_map[0][0].shape[0]

    def get_map_grid_y(self):
        return self.ind_map[0][0].shape[1]

    def combineMap(self, cur_config):
        '''

        :param cur_config:
        :return:
        '''
        output_map = np.zeros_like( self.ind_map[0][0] )
        for map_indx, pwr_indx in enumerate(cur_config):
            pwr_indx = int(pwr_indx) - 1
            if pwr_indx <0:
                continue
            cur_map = self.ind_map[map_indx][pwr_indx]
            output_map +=  np.power(10, cur_map/10.0)

        output_map_dB = 10*np.log10(output_map, where = output_map>0.0)
        output_map_dB[output_map<=0.0] = self.noise_floor_dB
        self.spectrum_maps.append(output_map_dB)

    def generateAllCombinationMap(self):
        '''

        :return:
        '''
        for cur_config in self.configs:
            self.combineMap(cur_config)
        # for cur_config, cur_map in zip(self.configs, self.spectrum_maps):
        #     self.displayMaps( [cur_map],'./heatmaps/'+cur_config+'.png')

    def displayMaps(self, map_list, figFilename, n_rows = 1):
        fig = plt.figure()
        fig.suptitle("Maximum %-diff in input map-pairs", fontsize=16)
        nrows_ncols = (n_rows, int(  np.round( len(map_list)/n_rows) ))
        grid = AxesGrid(fig, 111,
                        nrows_ncols= nrows_ncols,
                        axes_pad=0.1,
                        share_all=True,
                        label_mode="L",
                        cbar_location="right",
                        cbar_mode="single",
                        )
        vmin =  float('inf')
        vmax = -float('inf')
        for cur_map in map_list:
            cur_map_min = np.min(cur_map)
            cur_map_max = np.max(cur_map)
            vmin = min(vmin, cur_map_min)
            vmax = max(vmax, cur_map_max)
        if (vmax == vmin):
            vmax += 1
        for cur_map, ax in zip(map_list, grid):
            im = ax.imshow(cur_map, vmin=vmin, vmax=vmax)
        grid.cbar_axes[0].colorbar(im)
        #plt.show(block = False)
        plt.savefig(figFilename)

    def generateTrainingData(self, n_sample, dim_ratio = 0.05, add_noise = True):
        '''
        randomly generate sample_count samples each with a dim_ratio of dimensions
        return the sample array
        :param dim_ratio:
        :return:
        '''
        self.training_data_vals = []
        self.training_data_x_indx = []
        self.training_data_y_indx = []
        self.training_data_labels = [] #index to self.all_maps

        total_maps = len(self.spectrum_maps)
        map_max_x, map_max_y = self.spectrum_maps[0].shape
        all_loc_indx = list(np.ndindex(map_max_x, map_max_y))
        total_indices = len(all_loc_indx)
        indices_to_be_chosen = max(1, int(np.round(dim_ratio*map_max_x*map_max_y)))

        for i in np.arange(n_sample):
            map_indx = np.random.choice(total_maps, 1)[0]
            indx_to_locs = np.random.choice(total_indices, indices_to_be_chosen, replace = False)
            chosen_indices = sorted([all_loc_indx[cindx] for cindx in indx_to_locs])
            x_indx, y_indx = zip(*chosen_indices)
            chosen_signal_vals = self.spectrum_maps[map_indx][x_indx , y_indx]
            if add_noise:
                shadowing_vals = np.random.normal(loc = 0.0,
                                                  scale = self.sigma_db,
                                                  size = len(chosen_signal_vals)
                                                 )
                chosen_signal_vals +=  shadowing_vals
            self.training_data_labels.append(map_indx)
            self.training_data_vals.append(chosen_signal_vals)
            self.training_data_x_indx.append(x_indx)
            self.training_data_y_indx.append(y_indx)

    def prettyPrintSamples(self):
        '''
        mostly for debug
        :return:
        '''
        counter = 1
        for x_indices, y_indices, vals, label in zip(self.training_data_x_indx,
                                               self.training_data_y_indx,
                                               self.training_data_vals,
                                                self.training_data_labels
                                               ):
            print "Training Sample# ",counter, " map indx: ",label
            counter += 1
            for x_indx, y_indx, val in zip(x_indices, y_indices, vals):
                print "\t",x_indx,",",y_indx," : ",val

    def displayTrainingDataMap(self, map_indx):
        '''
        displays the heatmap for map_index training data
        :param map_indx:
        :return:
        '''
        cur_map =  np.empty_like(self.spectrum_maps[0])
        cur_map[:] = np.nan
        for x_indx, y_indx, val in zip(self.training_data_x_indx[map_indx],
                                       self.training_data_y_indx[map_indx],
                                       self.training_data_vals[map_indx]
                                       ):
            cur_map[ x_indx,y_indx ] = val
        self.displayMaps([self.spectrum_maps[ self.training_data_labels[map_indx]], cur_map])

    def plotMap3D(self, curMap):
        dim_x, dim_y = curMap.shape
        x_vals = np.arange(0.0, dim_x, 1.0)
        y_vals = np.arange(0.0, dim_y, 1.0)
        X, Y = np.meshgrid(x_vals, y_vals)

        # Plot the surface.
        ax = plt.axes(projection='3d')
        ax.set_zlim(0, 100)
        ax.plot_surface(X, Y, curMap, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title('surface')
        plt.show()

    def generateDiffMaps(self):
        '''

        :return:
        '''
        total_maps = len(self.spectrum_maps)
        diff_map = np.zeros_like(self.spectrum_maps[0])
        for i in range(total_maps - 1):
            for j in range(i+1,total_maps):
                cur_diff_map = 100.0*np.abs(  (self.spectrum_maps[i] - self.spectrum_maps[j]) / self.spectrum_maps[i] )
                diff_map = np.maximum(cur_diff_map, diff_map)
                #self.displayMaps(map_list=[cur_diff_map], figFilename="./plots/diff_" + str(i) + '_vs_' + str(j) + '.png')
        self.displayMaps(map_list=[diff_map], figFilename='./plots/diff_map.png')
        self.plotMap3D(diff_map)

if __name__ == '__main__':
    max_x = 1000.0
    max_y = 1000.0
    grid_size = 10.0
    noise_db = 10
    tx_power_levels = [   [ 10.0, 20.0 ,30.0, 40.0, 50.0, 60.0 ]
                         ,[ 10.0, 20,0, 30.0, 40.0, 50.0, 60.0 ]
                         ,[ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 ]
                         ,[ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 ]
                         ,[ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 ]
                         ,[ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 ]
                        , [ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 ]
                        , [ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 ]
                        , [ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 ]
                        , [ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 ]
                      ]
    tx_locs =  np.array([
                             [  100.0,  250.0  ]
                            ,[ 100.0,  750.0  ]
                            ,[  300.0,  500.0  ]
                            ,[ 300.0,  950.0  ]
                            ,[ 500.0,  150.0  ]
                            ,[ 500.0,  850.0  ]
                            ,[ 700.0,  450.0 ]
                            ,[ 700.0,  700.0 ]
                            ,[ 900.0,  50.0 ]
                            ,[ 900.0,  500.0 ]
                        ])

    #configs = ['22222', '23232', '32423', '42024', '14241']
    configs = ['1000000000', '0000000001']

    gsm = GenerateSpectrumMap(max_x_meter = max_x,
                              max_y_meter = max_y,
                              tx_power_dBm = tx_power_levels,
                              tx_loc = tx_locs,
                              configs = configs,
                              sigma_sq_db = noise_db,
                              d_0_meter=10.0 )
    gsm.generateIndividualMap()
    gsm.generateAllCombinationMap()
    gsm.generateDiffMaps()
    #gsm.generateTrainingData(n_sample=1000, dim_ratio=0.4, add_noise=True)

