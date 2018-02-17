import numpy as np
import itertools
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

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

        self.k_dB = 20.0 * np.log10(4.0 * np.pi * self.d_0_meter / self.lmbda_meter)
        self.noise_floor_dB = noise_floor_dB
        #np.random.seed(1009993)

    def generateIndividualMap(self):
        '''
        generates a map for each Tx
        :return:
        '''

        self.ind_map = []
        self.all_maps = []
        for i in range(self.tx_power_dBm.shape[0]):
            tx_x, tx_y = self.tx_loc[i][0], self.tx_loc[i][1]

            x_vals = np.arange(0, self.max_x_meter+1, self.d_0_meter)
            y_vals = np.arange(0, self.max_y_meter+1, self.d_0_meter)

            x_grid, y_grid = np.meshgrid(x_vals, y_vals, sparse=False, indexing='ij')
            dist_sq_map = (  (x_grid - tx_x)**2.0 + (y_grid - tx_y)**2.0 )

            path_loss =  self.k_dB  + 5.0 *self.n * np.log10( dist_sq_map/self.d_0_meter**2.0, where = dist_sq_map > 0.0)
            path_loss[dist_sq_map <= 0.0] = 0.0
            cur_map = self.tx_power_dBm[i] - path_loss
            self.ind_map.append(cur_map)
        #self.displayMaps(self.ind_map)

    def get_map_grid_x(self):
        return self.ind_map[0].shape[0]

    def get_map_grid_y(self):
        return self.ind_map[0].shape[1]

    def combineMap(self, indexList):
        '''

        :param indexList:
        :return:
        '''
        cur_map_mW = np.zeros_like( self.ind_map[0] )
        for indx in indexList:
            cur_map_mW +=  np.power(10, self.ind_map[indx]/10.0)

        cur_map_dB = 10*np.log10(cur_map_mW, where = cur_map_mW>0.0)
        cur_map_dB[cur_map_mW<=0.0] = self.noise_floor_dB
        #print "DEBUG: locs w NF: ",zip( cur_map_dB[cur_map_mW<=0.0] )

        self.all_maps.append(cur_map_dB)

    def generateAllCombinationMap(self):
        '''

        :return:
        '''
        all_indx = range(0, len( self.ind_map  )  )
        for l in range(1, len(all_indx) + 1):
            for comb in itertools.combinations(all_indx, l):
                self.combineMap(list(comb))
        #self.displayMaps(self.all_maps, 2)

    def displayMaps(self, map_list, figFilename, n_rows = 1):
        fig = plt.figure()
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(n_rows, int(  np.round( len(map_list)/n_rows) )),
                        axes_pad=0.01,
                        share_all=True,
                        label_mode="L",
                        cbar_location="right",
                        cbar_mode="single",
                        )

        for cur_map, ax in zip(map_list, grid):
            im = ax.imshow(cur_map, vmin=self.noise_floor_dB, vmax=0.0)
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

        total_maps = len(self.all_maps)
        map_max_x, map_max_y = self.all_maps[0].shape
        all_loc_indx = list(np.ndindex(map_max_x, map_max_y))
        total_indices = len(all_loc_indx)
        indices_to_be_chosen = max(1, int(np.round(dim_ratio*map_max_x*map_max_y)))

        for i in np.arange(n_sample):
            map_indx = np.random.choice(total_maps, 1)[0]
            indx_to_locs = np.random.choice(total_indices, indices_to_be_chosen, replace = False)
            chosen_indices = sorted([all_loc_indx[cindx] for cindx in indx_to_locs])
            x_indx, y_indx = zip(*chosen_indices)
            chosen_signal_vals = self.all_maps[map_indx][x_indx , y_indx]
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
            # print "map_indx: ", map_indx
            # print "locations: ", chosen_indices
            # print "x_indx:", x_indx
            # print "y_indx:", y_indx
            # print "chosen_vals: ",chosen_signal_vals

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
        cur_map =  np.empty_like(self.all_maps[0])
        cur_map[:] = np.nan
        for x_indx, y_indx, val in zip(self.training_data_x_indx[map_indx],
                                       self.training_data_y_indx[map_indx],
                                       self.training_data_vals[map_indx]
                                       ):
            cur_map[ x_indx,y_indx ] = val
        self.displayMaps([self.all_maps[ self.training_data_labels[map_indx]], cur_map])

if __name__ == '__main__':
    gsm = GenerateSpectrumMap(max_x_meter = 1000.0,
                              max_y_meter = 1000.0,
                              tx_power_dBm = np.array([30.0, 22.0, 15.0]),
                              tx_loc = np.array([
                                    [-100, -100],
                                    [500,  500],
                                    [750.0, 750.0]
                              ]),
                              d_0_meter=10.0
                              )
    gsm.generateIndividualMap()
    gsm.generateAllCombinationMap()
    gsm.generateTrainingData(n_sample=100, dim_ratio=0.8, add_noise=False)
