from clustering import SpecMapClustering
from input_based_data_generator import InputBasedDataGenerator
import numpy as np
import glob
import distutils.dir_util
from timeit import default_timer as mytimer
import cProfile

def  runExperiment():
    np.random.seed(1079993)
    map_directory = './maps' #the directory where all the resulting maps are saved for debugging
    distutils.dir_util.mkpath(map_directory)

    splat_pathloss_dir = '../splat_data/pathloss_maps_tx_5'
    tx_levels = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0 ]

    configs = ['10061', '66001','55555','43011' ]
    dim_ratio = 0.2 / 100
    sample_per_config = 100


    pca_var_ratio = 0.999
    gmm_cov_type = 'full'

    pathloss_files = []
    tx_powers = []
    for pfile in glob.glob(splat_pathloss_dir+"/*.txt"):
        pathloss_files.append(pfile)
        tx_powers.append(tx_levels)

    #----------------------------------------#
    ibdg = InputBasedDataGenerator(pathloss_files = pathloss_files,
                                   tx_powers=tx_powers,
                                   configs=configs,
                                   dim_ratio=dim_ratio,
                                   sample_per_config=sample_per_config,
                                   mapDirectory=map_directory)
    x_indices, y_indices, vals, labels, x_grid, y_grid = ibdg.getTrainingData(saveMap=False)
    input_maps = ibdg.getMaps()

    smc = SpecMapClustering(x_indices = x_indices,
                            y_indices = y_indices,
                            vals = vals,
                            labels = labels,
                            x_dim= x_grid,
                            y_dim= y_grid )
    smc.setPCAVarRatio(pca_var_ratio)
    smc.setGMMCovType(gmm_cov_type)
    iem_comp = 0
    print "================================="
    n_component_list = [len(configs)]
    # ---------------------IEM Experiment--------------------------------------------#
    iem_comp, iem_bic, iem_ari, iem_predicted_labels, iem_post_prob = smc.runGMMClustering(n_component_list=n_component_list)
    iem_derived_maps = smc.generateDerivedMaps(predicted_labels=iem_predicted_labels, pred_post_prob=iem_post_prob)
    iem_map_assoc, iem_avg_map_error = smc.avgPairwiseMapError( input_maps, iem_derived_maps)
    #
    # # ---------------------EMII Experiment---------------------------------------------#
    if iem_comp==0:
        iem_comp = n_component_list[0]
    emii_bic, emii_ari, emii_predicted_labels, emii_post_prob = smc.runIIGMMClusteringForKnownComponents(n_components=iem_comp)
    emii_derived_maps = smc.generateDerivedMaps(predicted_labels=emii_predicted_labels, pred_post_prob=emii_post_prob)
    emii_map_assoc, emii_avg_map_error = smc.avgPairwiseMapError(input_maps, emii_derived_maps)

    # -----------------------K-Means Experiment------------------------------------------------------#
    if iem_comp==0:
        iem_comp = n_component_list[0]
    kmeans_ari, kmeans_predicted_labels = smc.runKMeansClusteringForKnownComponents(n_components=iem_comp)
    kmeans_derived_maps = smc.generateDerivedMaps(predicted_labels=kmeans_predicted_labels)
    kmeans_map_assoc, kmeans_avg_map_error = smc.avgPairwiseMapError( input_maps, kmeans_derived_maps)

    print "IEM ARI, AER: ", iem_ari, iem_avg_map_error
    print "EMII ARI, AER: ", emii_ari, emii_avg_map_error
    print "KMeans ARI, AER: ",  kmeans_ari, kmeans_avg_map_error

    # print "Result Summary:"
    # print "================================="
    # print "No of chosen components: ", iem_comp
    # print "(KM, IEM, EMII):"
    # print "ARI: ", np.round( kmeans_ari, 2), " , ", np.round( iem_ari, 2), " , " , np.round( emii_ari, 2)
    # print "AER: ", np.round( kmeans_avg_map_error, 2), " , ",  np.round( iem_avg_map_error,   2)," , ",  np.round( emii_avg_map_error,  2)

    #-------------------------------map-display---------------------------------------#

    plot_filenames = ['IEM', 'EMII','KMeans']
    derived_map_list = [iem_derived_maps, emii_derived_maps , kmeans_derived_maps]
    map_assoc_list =  [iem_map_assoc, emii_map_assoc, kmeans_map_assoc]
    for derived_maps, map_assoc, filename in zip(derived_map_list, map_assoc_list, plot_filenames):
        row_1_maps = []
        row_2_maps = []
        for i, j in map_assoc:
            row_1_maps.append(input_maps[i])
            row_2_maps.append(derived_maps[j])
        display_mapList = row_1_maps + row_2_maps
        smc.displayMaps(map_list=display_mapList, figFilename='./maps/' + filename + '.png', n_rows=2)



if __name__ == '__main__':
    start_t = mytimer()

    profileCode = False
    if profileCode:
        cProfile.run('runExperiment()', 'expProfile.cprof')
    else:
        runExperiment()
    print 'Execution time:', np.round( (mytimer() - start_t),3),"seconds"

