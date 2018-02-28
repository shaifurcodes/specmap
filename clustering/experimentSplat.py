from clustering import SpecMapClustering
from input_based_data_generator import InputBasedDataGenerator
import numpy as np


if __name__ == '__main__':
    np.random.seed(1009993)
    pathloss_files = ['../splat_data/tx_1_pathloss.txt', '../splat_data/tx_2_pathloss.txt', '../splat_data/tx_3_pathloss.txt',
                      '../splat_data/tx_4_pathloss.txt']
    tx_powers = [
         [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0 ]
        ,[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0 ]
        ,[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0 ]
        ,[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0 ]

    ]
    configs = ['0444', '5067', '1306','3470']

    dim_ratio = 0.3/100.0
    #commiting
    sample_per_config = 500
    pca_var_ratio = 0.9

    gmm_cov_type = 'full'

    #----------------------------------------#
    ibdg = InputBasedDataGenerator(pathloss_files = pathloss_files,
                                   tx_powers=tx_powers,
                                   configs=configs,
                                   dim_ratio=dim_ratio,
                                   sample_per_config=sample_per_config)
    x_indices, y_indices, vals, labels, x_grid, y_grid = ibdg.getTrainingData(saveMap=True)
    maps = ibdg.getMaps()

    smc = SpecMapClustering(x_indices = x_indices,
                            y_indices = y_indices,
                            vals = vals,
                            labels = labels,
                            x_dim= x_grid,
                            y_dim= y_grid )
    smc.setPCAVarRatio(pca_var_ratio)
    smc.setGMMCovType(gmm_cov_type)
    print "================================="
    n_component_list = [len(configs)]
    # ---------------------IEM Experiment--------------------------------------------#
    iem_comp, iem_bic, iem_ari, iem_predicted_labels, iem_post_prob = smc.runGMMClustering(n_component_list=n_component_list)
    iem_derived_maps = smc.generateDerivedMaps(predicted_labels=iem_predicted_labels, pred_post_prob=iem_post_prob)
    iem_map_assoc, iem_avg_map_error = smc.avgPairwiseMapError( maps, iem_derived_maps)

    # ---------------------EMII Experiment---------------------------------------------#
    emii_bic, emii_ari, emii_predicted_labels, emii_post_prob = smc.runIIGMMClusteringForKnownComponents(n_components=iem_comp)
    emii_derived_maps = smc.generateDerivedMaps(predicted_labels=emii_predicted_labels, pred_post_prob=emii_post_prob)
    emii_map_assoc, emii_avg_map_error = smc.avgPairwiseMapError(maps, emii_derived_maps)

    # -----------------------K-Means Experiment------------------------------------------------------#
    kmeans_ari, kmeans_predicted_labels = smc.runKMeansClusteringForKnownComponents(n_components=iem_comp)
    kmeans_derived_maps = smc.generateDerivedMaps(predicted_labels=kmeans_predicted_labels)
    kmeans_map_assoc, kmeans_avg_map_error = smc.avgPairwiseMapError( maps, kmeans_derived_maps)

    print "IEM ARI, AER: ", iem_ari, iem_avg_map_error
    print "EMII ARI, AER: ", emii_ari, emii_avg_map_error
    print "KMeans ARI, AER: ",  kmeans_ari, kmeans_avg_map_error

