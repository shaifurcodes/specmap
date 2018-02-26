from __future__ import division

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

from scipy.sparse import coo_matrix
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA as myPCA
from sklearn.cluster import KMeans

from idw import  idw



class SpecMapClustering:
    '''
    tasks:
    i) runs IEM
    ii) runs EMII
    iii_ runs KMeans
    '''
    def __init__(self, x_indices, y_indices, vals, labels, x_dim, y_dim, pca_var_ratio = 1.0, gmm_cov_type = 'full'):
        '''
        :param x_indices:
        :param y_indices:
        :param vals:
        :param labels:
        :param x_dim:
        :param y_dim:
        :param pca_var_ratio:
        '''
        self.labels = labels
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.pca_var_ratio = pca_var_ratio
        self.gmm_cov_type = gmm_cov_type
        #-------------init training data as sparse matrix---------------------#
        self.training_data = []
        for cur_x_indices, cur_y_indices, cur_vals in zip(x_indices, y_indices, vals):
            cur_coo_mat = coo_matrix((cur_vals, (cur_x_indices, cur_y_indices)), shape=(self.x_dim, self.y_dim))
            self.training_data.append(cur_coo_mat)
        #-----------------------------------------------------------------------#
        self.cached_training_matrix = None
        self.kmeans_training_matrix = None

    def displayMaps(self, map_list, figFilename = None, n_rows = 1, title=''):
        fig = plt.figure()
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(n_rows, int(  np.round( len(map_list)/n_rows) )),
                        axes_pad=0.01,
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
            #print 'DEBUG:vmax, vmin: ',vmax, vmin

        for cur_map, ax in zip(map_list, grid):
            im = ax.imshow(cur_map, vmin= vmin, vmax=vmax)
        grid.cbar_axes[0].colorbar(im)

        if figFilename is not None:
            plt.savefig(figFilename)
        else:
            plt.show(block = False)


    def matPrint(self, a):
        '''
        :param a:
        :return:
        '''
        x,y = a.shape
        print "<Begin Matrix>"
        for i in range(x):
            for j in range(y):
                print "%0.2f" % a[i, j],"\t",
            print "\n"
        print "<End Matrix>"

    def setPCAVarRatio(self, pca_var_ratio):
        '''
        :param pca_var_ratio:
        :return:
        '''
        self.pca_var_ratio = pca_var_ratio

    def setGMMCovType(self, gmm_cov_type):
        '''
        :param gmm_cov_type:
        :return:
        '''
        self.gmm_cov_type = gmm_cov_type

    def generateAggregateMapPerCluster(self, prev_predicted_labels, prev_posterior_prob):
        '''
        :param prev_predicted_labels:
        :param prev_post_prob:
        :return:
        '''
        no_of_clusters = max(prev_predicted_labels) + 1
        aggregate_map = []
        aggregate_count = []
        for i in range(no_of_clusters):
            aggregate_map.append(np.zeros(shape=(self.x_dim, self.y_dim), dtype=np.float) )
            aggregate_count.append(np.zeros(shape=(self.x_dim, self.y_dim), dtype=np.float) )
        for cur_map, cur_label, cur_prob in zip(self.training_data, prev_predicted_labels, prev_posterior_prob):
            aggregate_map[cur_label] += cur_prob*cur_map.toarray()
            aggregate_count[cur_label][cur_map.row, cur_map.col] += cur_prob
        for i in range(no_of_clusters):
            aggregate_map[i] = np.divide(aggregate_map[i],  aggregate_count[i], where=aggregate_count[i] !=0)
            aggregate_map[i] = np.where(aggregate_count[i] == 0, np.nan, aggregate_map[i])
        return aggregate_map

    def interpolateMissingValues(self, cur_map, qnear = 10):
        '''
        :param cur_map:
        :param qnear:
        :return:
        '''
        grid_x, grid_y = np.where( ~np.isnan(cur_map) )
        grid_xy = np.array(  zip(grid_x, grid_y), dtype=np.float64  )
        vals = cur_map[grid_x, grid_y]

        q_grid_x, q_grid_y = np.where( np.isnan( cur_map) )
        if len(q_grid_x) == 0:
            return  cur_map
        q_grid_xy = np.array(  zip(q_grid_x, q_grid_y), dtype=np.float64  )
        #cur_idw = IDW(grid_xy, vals)
        cur_map[q_grid_x, q_grid_y] = idw(  X=grid_xy, z=vals, q= q_grid_xy, nnear= min(qnear, len(vals))   )
        return cur_map


    def generateTrainingMatrix(self, prev_predicted_labels = None, prev_post_prob = None, cache_training_matrix = False):
        '''
        :param prev_predicted_labels:
        :param prev_post_prob:
        :param cache_training_matrix:
        :return:
        '''
        cur_training_matrix = None

        aggregate_map = None
        if prev_predicted_labels is not None:
            aggregate_map = self.generateAggregateMapPerCluster(prev_predicted_labels, prev_post_prob)

        indx = 0
        # ----interpolate the missing values----------#
        for cur_sample in self.training_data :
            cur_map = np.empty( shape = (self.x_dim, self.y_dim), dtype=np.float )
            cur_map[:] = np.nan
            cur_map[cur_sample.row, cur_sample.col] =   cur_sample.toarray()[ cur_sample.row, cur_sample.col]
            #self.matPrint(cur_map)
            if prev_predicted_labels is not None:
                prev_label = prev_predicted_labels[indx]
                cur_map = np.where(np.isnan(cur_map), aggregate_map[prev_label], cur_map)
            indx += 1
            #print("DEBUG: processing ",indx," sample")
            cur_map = self.interpolateMissingValues(cur_map)
            #self.plotMap3D(cur_map)

            if indx > 1:
                cur_training_matrix  =  np.vstack( (cur_training_matrix, cur_map.flatten()) )
            else:
                cur_training_matrix = cur_map.flatten()
        if cache_training_matrix:
            if self.kmeans_training_matrix is None:
                self.kmeans_training_matrix = np.copy(cur_training_matrix)
        if self.pca_var_ratio < 1.0:
            pca = myPCA(self.pca_var_ratio)
            print "\tRunning PCA.."
            prev_dim = cur_training_matrix.shape[1]
            cur_training_matrix =  pca.fit_transform(cur_training_matrix)
            print "\t\t dim reduced ",prev_dim," --> ",cur_training_matrix.shape[1]
        if cache_training_matrix:
            if self.cached_training_matrix is None:
                self.cached_training_matrix = np.copy( cur_training_matrix )
        return cur_training_matrix

    def runIIGMMClusteringForKnownComponents(self, n_components,
                                             max_iteration=100
                                             ):
        '''
        :param n_components:
        :param max_iteration:
        :param coverience_type:
        :return:
        '''
        cur_training_matrix = None
        cur_gmm = GaussianMixture( n_components = n_components,
                                        covariance_type = self.gmm_cov_type
                                        #,warm_start=True
                                        #,max_iter=1
                                        )
        prev_model_score = None
        prev_predicted_labels = None
        prev_posterior_probability = None

        new_model_score = None
        new_pred_labels = None
        new_posterior_probability = None

        new_ari = None
        new_bic = None

        for i in range(max_iteration):
            print "DEBUG: EMII: iteration#", i
            if i==0 and (self.cached_training_matrix is not None): #use cached training matrix, if available, for the first iteration
                cur_training_matrix = self.cached_training_matrix
            else:
                cur_training_matrix = self.generateTrainingMatrix(prev_predicted_labels = prev_predicted_labels,
                                                              prev_post_prob = prev_posterior_probability)
            cur_gmm.fit(cur_training_matrix)
            new_model_score = cur_gmm.score( cur_training_matrix )
            new_pred_labels = cur_gmm.predict( cur_training_matrix )
            new_posterior_probability = np.amax( cur_gmm.predict_proba( cur_training_matrix), axis = 1)
            new_ari = adjusted_rand_score(self.labels, new_pred_labels)
            new_bic = cur_gmm.bic(cur_training_matrix)
            print "\t model-score:",new_model_score, " ARI: ",new_ari
            if i>0:
                percent_change = 100.0*np.abs(prev_model_score - new_model_score)/prev_model_score
                if percent_change <0.01: #<----if change less than 0.5%---
                    break
            prev_model_score = new_model_score
            prev_predicted_labels = new_pred_labels
            prev_posterior_probability = new_posterior_probability
        return new_bic, new_ari, new_pred_labels, new_posterior_probability

    def runKMeansClusteringForKnownComponents(self, n_components):
        '''

        :param n_components:
        :return:
        '''
        if self.kmeans_training_matrix is None:
            _ = self.generateTrainingMatrix(cache_training_matrix=True)
        self.kms_predicted_labels = KMeans(n_clusters= n_components).fit_predict(self.kmeans_training_matrix)
        self.kms_ari = adjusted_rand_score( self.labels, self.kms_predicted_labels)
        return self.kms_ari, self.kms_predicted_labels

    def runGMMClusteringForKnownComponents(self, n_components):
        '''
        :param n_components:
        :return:
        '''
        if self.cached_training_matrix is None:
            _ = self.generateTrainingMatrix(cache_training_matrix=True)
        cur_gmm = GaussianMixture( n_components = n_components,
                               covariance_type = self.gmm_cov_type).fit(self.cached_training_matrix)
        predicted_labels = cur_gmm.predict(self.cached_training_matrix)
        post_prob = np.amax( cur_gmm.predict_proba( self.cached_training_matrix), axis = 1)
        cur_ari = adjusted_rand_score(self.labels, predicted_labels)
        #cur_aic = cur_gmm.aic(self.cached_training_matrix)
        cur_bic = cur_gmm.bic(self.cached_training_matrix)

        return cur_bic, cur_ari, predicted_labels, post_prob

    def runGMMClustering(self, n_component_list):
        '''
        :param n_component_list:
        :return:
        '''
        min_comp = float('inf')
        min_bic = float('inf')
        min_aic = float('inf')
        min_ari = float('inf')
        min_predicted_labels = None
        min_post_prob = None
        for c in n_component_list:
            print "DEBUG: IEM: cluster# : ", c
            cur_bic, cur_ari, predicted_labels, post_prob = self.runGMMClusteringForKnownComponents(n_components =c)
            print "\t BIC: ",cur_bic," ARI: ",cur_ari
            if cur_bic < min_bic:
                min_comp, min_bic, min_ari, min_predicted_labels, min_post_prob = c, cur_bic, cur_ari, predicted_labels, post_prob

        return min_comp, min_bic, min_ari, min_predicted_labels, min_post_prob

    def generateDerivedMaps(self, predicted_labels, pred_post_prob = None):
        '''
        :param predicted_labels:
        :return:
        '''
        post_prob = pred_post_prob
        if post_prob is None:
            post_prob = np.ones(shape = predicted_labels.shape, dtype = np.int)
        derived_maps = self.generateAggregateMapPerCluster(prev_predicted_labels=predicted_labels,
                                                          prev_posterior_prob=post_prob)
        for indx, cur_map in enumerate(derived_maps):
            derived_maps[indx] = self.interpolateMissingValues(cur_map=cur_map)
        return derived_maps

    def computeMapAccuracy(self, mapA_dBm, mapB_dBm):
        '''
        convert dBm to miliwatt and then take pairwise percent diff, here, mapA_dBm is the original map to base %-error
        :return:
        '''
        mapA_mW = np.power(10.0, mapA_dBm / 10.0)
        mapB_mW = np.power(10.0, mapB_dBm / 10.0)

        mapA_mW[mapA_dBm == 0.0] = 0.000001 #to avoid div-by-zero

        error_A_B = (np.abs(mapA_mW - mapB_mW)) / mapA_mW
        avg_error = np.average(error_A_B)
        return avg_error

    def computeAbsMapAccuracy(self, mapA_dBm, mapB_dBm):
        '''
        :param mapA_dBm:
        :param mapB_dBm:
        :return:
        '''
        return np.sum(np.abs(mapA_dBm - mapB_dBm))

    def avgPairwiseMapError(self,maps_A_dBm, maps_B_dBm):
        '''
        compute bipartite graph between maps_A_dBm, and maps_B_dBm based on minimum error
        and generate avg error
        :param maps_A_dBm:
        :param maps_B_dBm:
        :return: mapA indices, corresponding mapB indices, avg_map_error
        '''
        error_matrix = np.zeros( shape = ( len(maps_A_dBm), len(maps_B_dBm) ) )
        for i,cur_map_A in enumerate(maps_A_dBm):
            for j, cur_map_B in enumerate(maps_B_dBm):
                error_matrix[i][j] = self.computeMapAccuracy(cur_map_A, cur_map_B)
        #if dimensions are equal, just associate row-wise mins
        min_err_row_indx, min_err_col_indx = linear_sum_assignment(error_matrix)

        if len(maps_A_dBm) > len(maps_B_dBm): #unassociated rows there
            unassociated_rows = np.array([x for x in range(len(maps_A_dBm)) if x not in min_err_row_indx])
            while unassociated_rows.size >0:
                new_error_matrix = error_matrix[ unassociated_rows , :]
                new_min_error_row_indx,  new_min_error_col_indx = linear_sum_assignment(new_error_matrix)
                new_min_error_row_indx = unassociated_rows[ new_min_error_row_indx ]
                min_err_row_indx = np.concatenate( (min_err_row_indx, new_min_error_row_indx ) )
                min_err_col_indx = np.concatenate( ( min_err_col_indx, new_min_error_col_indx ) )
                unassociated_rows = np.array( [x for x in unassociated_rows if x not in new_min_error_row_indx ] )
        elif len(maps_A_dBm) < len(maps_B_dBm): #unassociated cols there
            unassociated_cols = np.array([x for x in range(len(maps_B_dBm)) if x not in min_err_col_indx])
            while unassociated_cols.size>0:
                new_error_matrix = error_matrix[ : , unassociated_cols]
                new_min_error_row_indx,  new_min_error_col_indx = linear_sum_assignment(new_error_matrix)
                new_min_error_col_indx = unassociated_cols[ new_min_error_col_indx ]
                min_err_row_indx = np.concatenate( (min_err_row_indx, new_min_error_row_indx ) )
                min_err_col_indx = np.concatenate( (min_err_col_indx, new_min_error_col_indx ) )
                unassociated_cols = np.array( [x for x in unassociated_cols if x not in new_min_error_col_indx ] )

        avg_map_error =  np.average(  error_matrix[min_err_row_indx, min_err_col_indx] )
        return zip(min_err_row_indx, min_err_col_indx) , avg_map_error

    def plotMap3D(self, curMap):
        dim_x, dim_y = curMap.shape
        x_vals = np.arange(0.0, dim_x, 1.0)
        y_vals = np.arange(0.0, dim_y, 1.0)
        X, Y = np.meshgrid(x_vals, y_vals)

        # Plot the surface.
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, curMap, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title('surface')
        plt.show()

    # def generateDiffMaps(self):
    #     '''
    #
    #     :return:
    #     '''
    #     total_maps = len(self.spectrum_maps)
    #     diff_map = np.zeros_like(self.spectrum_maps[0])
    #     for i in range(total_maps - 1):
    #         for j in range(i+1,total_maps):
    #             cur_diff_map = 100.0*np.abs(  (self.spectrum_maps[i] - self.spectrum_maps[j]) / self.spectrum_maps[i] )
    #             diff_map = np.maximum(cur_diff_map, diff_map)
    #             #self.displayMaps(map_list=[cur_diff_map], figFilename="./plots/diff_" + str(i) + '_vs_' + str(j) + '.png')
    #     self.displayMaps(map_list=[diff_map], figFilename='./plots/diff_map.png')
    #     self.plotMap3D(diff_map)

#----------------------------------------------------------------------------------------------------------#