from data_generator import GenerateSpectrumMap
from clustering import SpecMapClustering
from timeit import default_timer as mytimer
import cProfile
import numpy as np

def createTrainingDataAndClusteringObject():
    tx_power_dBm = np.array( [40.0, 40.0, 40.0, 40.0] )

    tx_loc = np.array([  [  100,  100 ]
                        ,[  900,  900 ]
                        ,[  500,  500 ]
                        ,[   300, 600 ]
                      ])
    n_sample_per_config = 100
    dim_ratio =  0.3/100.0
    max_x_meter = 1000.0
    max_y_meter = 1000.0
    gsm = GenerateSpectrumMap(max_x_meter = max_x_meter,
                              max_y_meter = max_y_meter,
                              tx_power_dBm = tx_power_dBm,
                              tx_loc = tx_loc,
                              d_0_meter=   10.0,
                              sigma_sq_db = 5.0
                              )
    gsm.generateIndividualMap()
    gsm.generateAllCombinationMap()

    total_distinct_config = int(np.power(2, tx_power_dBm.shape[0])) - 1

    n_sample =  n_sample_per_config * total_distinct_config
    print "#-of generated samples: ",n_sample
    gsm.generateTrainingData(n_sample=n_sample, dim_ratio = dim_ratio, add_noise=False)
    smc = SpecMapClustering(x_indices = gsm.training_data_x_indx,
                            y_indices = gsm.training_data_y_indx,
                            vals = gsm.training_data_vals,
                            labels = gsm.training_data_labels,
                            x_dim= gsm.get_map_grid_x(),
                            y_dim= gsm.get_map_grid_y()
                            )

    return  gsm, smc

def runExperiment():
    np.random.seed(1009993)

    gsm, smc = createTrainingDataAndClusteringObject()

    pca_var_ratio = 0.9
    gmm_cov_type = 'full'
    n_component_list = [15]

    smc.setPCAVarRatio(pca_var_ratio)
    smc.setGMMCovType(gmm_cov_type)

    # ---------------------IEM Experiment--------------------------------------------#
    iem_comp, iem_bic, iem_ari, iem_predicted_labels, iem_post_prob = smc.runGMMClustering(n_component_list=n_component_list)
    iem_derived_maps = smc.generateDerivedMaps(predicted_labels=iem_predicted_labels, pred_post_prob=iem_post_prob)
    iem_map_assoc, iem_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps, iem_derived_maps)
    # ---------------------EMII Experiment---------------------------------------------#
    emii_bic, emii_ari, emii_predicted_labels, emii_post_prob = smc.runIIGMMClusteringForKnownComponents(
        n_components=iem_comp)
    emii_derived_maps = smc.generateDerivedMaps(predicted_labels=emii_predicted_labels, pred_post_prob=emii_post_prob)
    emii_map_assoc, emii_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps, emii_derived_maps)
    # -----------------------K-Means Experiment------------------------------------------------------#
    kmeans_ari, kmeans_predicted_labels = smc.runKMeansClusteringForKnownComponents(n_components=iem_comp)
    kmeans_derived_maps = smc.generateDerivedMaps(predicted_labels=kmeans_predicted_labels)
    kmeans_map_assoc, kmeans_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps, kmeans_derived_maps)
    # ------------------------Summary-------------------------------------------------#
    print "Summary:------------------"
    print "No of chosen components: ", iem_comp, "min BIC: ",iem_bic
    print "KMeans-ARI: ", kmeans_ari, " Avg-Error:", kmeans_avg_map_error
    print "IEM-ARI:", iem_ari, " Avg-Error:", iem_avg_map_error
    print "EMII-ARI:", emii_ari, " Avg-Error:", emii_avg_map_error

    #-------------------------------map-display---------------------------------------#
    # input_maps = gsm.all_maps
    # plot_filenames = ['IEM', 'EMII','KMeans']
    # derived_map_list = [iem_derived_maps, emii_derived_maps , kmeans_derived_maps]
    # map_assoc_list = [iem_map_assoc, emii_map_assoc, kmeans_map_assoc]
    # for derived_maps, map_assoc, filename in zip(derived_map_list, map_assoc_list, plot_filenames):
    #     row_1_maps = []
    #     row_2_maps = []
    #     for i, j in map_assoc:
    #         row_1_maps.append(input_maps[i])
    #         row_2_maps.append(derived_maps[j])
    #     display_mapList = row_1_maps + row_2_maps
    #     smc.displayMaps(map_list=display_mapList, figFilename='./plots/' + filename + '.png', n_rows=2)
    #

if __name__ == '__main__':
    start_t = mytimer()

    profileCode = True
    if profileCode:
        cProfile.run('runExperiment()', 'expProfile.cprof')
    else:
        runExperiment()
    print 'Execution time:',(mytimer() - start_t)