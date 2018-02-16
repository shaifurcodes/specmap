
from sample_generator import GenerateSpectrumMap
from clustering import SpecMapClustering


from timeit import default_timer as mytimer
import cProfile
import numpy as np

def createTrainingDataAndClusteringObject():
    max_x = 1000.0
    max_y = 1000.0
    tx_power_levels = [   [ 20.0,  35.0,   40.0,  50.0 ]
                         ,[ 20.0,  35.0,   40.0,  50.0 ]
                         ,[ 20.0,  35.0,   40.0,  50.0 ]
                         ,[ 20.0,  35.0,   40.0,  50.0 ]
                         ,[ 20.0,  35.0,   40.0,  50.0 ]
                      ]
    tx_locs =  np.array([
                             [ 250.0,  250.0  ]
                            ,[ 250.0,  750.0  ]
                            ,[ 500.0,  500.0  ]
                            ,[ 750.0,  250.0  ]
                            ,[ 750.0,  750.0  ]
                        ])

    configs = ['11411', '12121', '12131', '13121', '13431']
    sample_per_config = 100
    dim_ratio = 0.1

    total_sample = sample_per_config*len(configs)
    gsm = GenerateSpectrumMap(max_x_meter = 1000.0,
                              max_y_meter = 1000.0,
                              tx_power_dBm = tx_power_levels,
                              tx_loc = tx_locs,
                              configs = configs,
                              d_0_meter=10.0 )

    gsm.generateIndividualMap()

    gsm.generateAllCombinationMap()

    gsm.generateTrainingData(n_sample=total_sample, dim_ratio=dim_ratio, add_noise=True)

    smc = SpecMapClustering(x_indices = gsm.training_data_x_indx,
                            y_indices = gsm.training_data_y_indx,
                            vals = gsm.training_data_vals,
                            labels = gsm.training_data_labels,
                            x_dim= gsm.get_map_grid_x(),
                            y_dim= gsm.get_map_grid_y()
                            )

    return  gsm, smc, len(configs)

def runExperiment():
    np.random.seed(1009993)

    gsm, smc, n_component = createTrainingDataAndClusteringObject()

    n_component_list = [n_component]

    pca_var_ratio = 0.95
    gmm_cov_type = 'full'

    smc.setPCAVarRatio(pca_var_ratio)
    smc.setGMMCovType(gmm_cov_type)

    # ---------------------IEM Experiment--------------------------------------------#
    iem_comp, iem_bic, iem_ari, iem_predicted_labels, iem_post_prob = smc.runGMMClustering(n_component_list=n_component_list)
    iem_derived_maps = smc.generateDerivedMaps(predicted_labels=iem_predicted_labels, pred_post_prob=iem_post_prob)
    iem_map_assoc, iem_avg_map_error = smc.avgPairwiseMapError(gsm.spectrum_maps, iem_derived_maps)
    # ---------------------EMII Experiment---------------------------------------------#
    emii_bic, emii_ari, emii_predicted_labels, emii_post_prob = smc.runIIGMMClusteringForKnownComponents(
        n_components=iem_comp)
    emii_derived_maps = smc.generateDerivedMaps(predicted_labels=emii_predicted_labels, pred_post_prob=emii_post_prob)
    emii_map_assoc, emii_avg_map_error = smc.avgPairwiseMapError(gsm.spectrum_maps, emii_derived_maps)
    # -----------------------K-Means Experiment------------------------------------------------------#
    kmeans_ari, kmeans_predicted_labels = smc.runKMeansClusteringForKnownComponents(n_components=iem_comp)
    kmeans_derived_maps = smc.generateDerivedMaps(predicted_labels=kmeans_predicted_labels)
    kmeans_map_assoc, kmeans_avg_map_error = smc.avgPairwiseMapError(gsm.spectrum_maps, kmeans_derived_maps)
    # ------------------------Summary-------------------------------------------------#
    print "Summary:------------------"
    print "No of chosen components: ", iem_comp
    print "(KM, IEM, EMII):"
    print "ARI: ", np.round( kmeans_ari, 2), " , ", np.round( iem_ari, 2), " , " , np.round( emii_ari, 2)
    print "AER: ", np.round( kmeans_avg_map_error, 2), " , ",  np.round( iem_avg_map_error,   2)," , ",  np.round( emii_avg_map_error,  2)

    #-------------------------------map-display---------------------------------------#
    input_maps = gsm.spectrum_maps
    plot_filenames = ['IEM', 'EMII','KMeans']
    derived_map_list = [iem_derived_maps, emii_derived_maps , kmeans_derived_maps]
    map_assoc_list = [iem_map_assoc, emii_map_assoc, kmeans_map_assoc]
    for derived_maps, map_assoc, filename in zip(derived_map_list, map_assoc_list, plot_filenames):
        row_1_maps = []
        row_2_maps = []
        for i, j in map_assoc:
            row_1_maps.append(input_maps[i])
            row_2_maps.append(derived_maps[j])
        display_mapList = row_1_maps + row_2_maps
        smc.displayMaps(map_list=display_mapList, figFilename='./plots/' + filename + '.png', n_rows=2)


if __name__ == '__main__':
    start_t = mytimer()

    profileCode = False
    if profileCode:
        cProfile.run('runExperiment()', 'expProfile.cprof')
    else:
        runExperiment()
    print 'Execution time:',(mytimer() - start_t)