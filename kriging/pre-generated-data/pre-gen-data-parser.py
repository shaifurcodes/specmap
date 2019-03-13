import numpy as np
import cProfile
from timeit import default_timer as mytimer
from guppy import hpy
from functools import reduce

import matplotlib.pyplot as plt

class SUAllocation(object):
    '''
    for every SU:
        1. Find the closest PU+PRs AND closest 5 SS
        2. Run IDW
    '''
    def __init__(self):
        '''

        '''
        self.total_SS = 0
        self.total_PU = 0
        self.total_SU = 0
        self.pr_per_pu = 5
        self.SS = None
        self.SU = None
        self.PU = None
        self.PR = None
        self.loss_PU_SS = None
        self.loss_PR_SU = None

    def allocateArrays(self):
        '''
        allocate PU, SU, PR, SS arrays, must set total_* values
        :return:
        '''
        self.SS = np.zeros((self.total_SS, 3), dtype=float)
        self.PU = np.zeros((self.total_PU, 4), dtype=float)
        self.SU = np.zeros((self.total_SU, 3), dtype=float)
        self.PR = np.zeros((self.total_PU, self.pr_per_pu, 4), dtype=float)
        self.loss_PR_SU = np.zeros((self.total_PU, self.pr_per_pu, self.total_SU ), dtype = float)
        self.loss_PU_SS = np.zeros( (self.total_PU, self.total_SS) , dtype = float)
        return

    def parsePreGeneratedPathlossData(self, ifname = 'data_80k.txt',
                                      total_PU = 400,
                                      total_SU = 500,
                                      total_SS = 80000,
                                      pr_per_pu = 5
                                      ):
        '''
        parse the file as which is structued as follows:
            PU PU_ID X Y Z Transmit_power
            PR PU_ID PR_ID X Y Z Threshold
            SS SS_ID X Y Z
            SU SU_ID X Y Z
            A PU_ID SS_ID Path_loss
            B PU_ID PR_ID SU_ID Path_loss
        :return:
        '''
        self.total_PU = total_PU
        self.total_SU = total_SU
        self.total_SS = total_SS
        self.pr_per_pu = pr_per_pu

        self.allocateArrays()

        with open(ifname, 'r') as f:
            for line in f:
                words = line.split()
                tx_type = words[0]
                if tx_type=='COUNT':
                    self.total_PU, self.total_SU, self.total_SS = int( words[1] ), int( words[2] ), int( words[3] )
                    #--reallocatep---COUNT Should be the first line---#
                    self.allocateArrays()
                elif tx_type in ['PU', 'SS', 'SU']:
                    indx = int(words[1])
                    x = float(words[2])
                    y = float(words[3])
                    z = float(words[4])
                    if tx_type == 'PU':
                        t = float(words[5])
                        self.PU[indx] = [x, y, z, t]
                    elif tx_type == 'SU':
                        self.SU[indx] = [x, y, z]
                    elif tx_type == 'SS':
                        self.SS[indx] = [x, y, z]

                elif tx_type == 'PR':
                    pu_indx = int(words[1])
                    pr_indx = int(words[2])
                    x = float(words[3])
                    y = float(words[4])
                    z = float(words[5])
                    t = float(words[6])
                    self.PR[pu_indx][pr_indx] = [x, y, z, t]

                elif tx_type == 'A':
                    pu_indx = int(words[1])
                    ss_indx = int(words[2])
                    t =  float(words[3])
                    self.loss_PU_SS[pu_indx, ss_indx] = t

                elif tx_type == 'B':
                    pu_indx = int(words[1])
                    pr_indx = int(words[2])
                    su_indx = int(words[3])
                    t =  float(words[4])
                    self.loss_PR_SU[pu_indx, pr_indx, su_indx] = t
        return

    def findObjectsInBoundedRegion(self, val_array ,max_x = float('inf'),
                                   max_y = float('inf'),
                                   min_x = 0.,
                                   min_y = 0.,
                                   x_indx = 0,
                                   y_index = 1):
        '''
        :param max_x:
        :param max_y:
        :param min_x:
        :param min_y:
        :return:
        '''
        indx_val_array = reduce(np.intersect1d, (np.where(val_array[:, x_indx] >= min_x),
                                       np.where(val_array[:, x_indx] <= max_x),
                                       np.where(val_array[:, y_index] >= min_y),
                                       np.where(val_array[:, y_index] <= max_y)))
        return indx_val_array

    def plotLocations(self):
        '''

        :return:
        '''
        print "DEBUG: PU:", self.PU.shape
        print self.PU

        print "DEBUG: SU:", self.SU.shape
        print self.SU

        print "DEBUG: SS:",self.SS.shape
        print self.SS


        plt.scatter( self.PU[:, 0], self.PU[:, 1], marker = 's', s=50, c='b' )
        plt.scatter( self.SS[:, 0], self.SS[:, 1], marker = 'o', s=10,  c='r' )
        plt.scatter( self.SU[:, 0], self.SU[:, 1], marker = '^', s=50, c='g' )
        plt.show()
        return

    def extractSmallerScenario(self, max_x = float('inf'),
                               max_y = float('inf'),
                               min_x = -float('inf'),
                               min_y = -float('inf'),
                               ofname = 'data_small.txt'):
        '''
        must call self.parseGeneratedPathlssData(..) first
        extracts smaller scenario from 80k datafile, must load
            -> find all the PU, SS and SU in the range and list them
            -> find the PU within the range and along its self.PR and populate self.PU, self.PR
            -> find all the SS within the range and populate self.SS and self.loss_PU_SS
            -> find all the SU within the range and populate self.SU and self.loss_PR_SU for each PR
            -> finally save all the above in the output file mentioned
        :return:
        '''
        start_t = mytimer()
        print "debug: extracting"
        if self.PU is None:
            self.parsePreGeneratedPathlossData() #default for 80k data
        n_pu = None
        n_su = None
        n_ss = None

        x_indx, y_indx = 0, 1
        n_pu = self.findObjectsInBoundedRegion(self.PU, max_x= max_x , max_y = max_y,
                                                        min_x= min_x, min_y = min_y,
                                                        x_indx = x_indx, y_index = y_indx )
        print "Total PU:", n_pu.shape[0]

        x_indx, y_indx = 0, 1
        n_su = self.findObjectsInBoundedRegion(self.SU, max_x= max_x , max_y = max_y,
                                                        min_x= min_x, min_y = min_y,
                                                        x_indx = x_indx, y_index = y_indx )
        print "Total SU:", n_su.shape[0]


        x_indx, y_indx = 0, 1
        n_ss = self.findObjectsInBoundedRegion(self.SS, max_x= max_x , max_y = max_y,
                                                        min_x= min_x, min_y = min_y,
                                                        x_indx = x_indx, y_index = y_indx )
        print "Total SS: ", n_ss.shape[0]

        # plt.scatter( self.PU[n_pu, 0], self.PU[n_pu, 1], marker = 's', s=25, c='b' )
        # plt.scatter( self.SS[n_ss, 0], self.SS[n_ss, 1], marker = 'o', s=4,  c='r' )
        # plt.scatter( self.SU[n_su, 0], self.SU[n_su, 1], marker = '^', s=25, c='g' )
        # plt.show()

        #--RETHINK USING LOOP TO AVOID ERROR--!!
        with open(ofname, "w") as f:

            #---write the count-----#
            f_line = "COUNT"+" "+str(  n_pu.shape[0] )+" "+str( n_su.shape[0] )+" "+str( n_ss.shape[0] )
            f.write(f_line)
            # ------write the PU's----#
            #-------PU PU_ID X Y Z Transmit_power
            pu_counter = 0
            for i in n_pu:
                f_line = "\n"+"PU"+" "+str(pu_counter)+\
                                   " "+str( self.PU[i, 0] )+" "+str( self.PU[i, 1] )+\
                                   " "+str( self.PU[i, 2] )+" "+str( self.PU[i, 3] )
                f.write(f_line)
                pu_counter += 1

            #------write the SU's----#
            #-----SU SU_ID X Y Z
            su_counter = 0
            for i in n_su:
                f_line = "\n"+"SU"+" "+str(su_counter)+\
                                   " "+str( self.SU[i, 0] )+" "+str( self.SU[i, 1] )+\
                                   " "+str( self.SU[i, 2] )
                f.write(f_line)
                su_counter += 1

            #-------write the SS's----#
            #-----SS SS_ID X Y Z
            ss_counter = 0
            for i in n_ss:
                f_line = "\n"+"SS"+" "+str(ss_counter)+\
                                   " "+str( self.SS[i, 0] )+" "+str( self.SS[i, 1] )+\
                                   " "+str( self.SS[i, 2] )
                f.write(f_line)
                ss_counter += 1

            #---write the PR's-----#
            #---PR PU_ID PR_ID X Y Z Threshold
            pu_counter = 0
            for i in n_pu:
                for j in np.arange(0, self.pr_per_pu):
                    f_line = "\n"+"PR"+" "+str( pu_counter )+" "+str( j )+\
                                   " "+str( self.PR[i, j, 0] )+" "+str( self.PR[i, j, 1] )+\
                                   " "+str( self.PR[i, j, 2] )
                pu_counter += 1

            #-----write the A: PU-SS path-loss
            #-----A PU_ID SS_ID Path_loss---
            pu_counter = 0
            for i in n_pu:
                ss_counter = 0
                for j in n_ss:
                    f_line = "\n"+"A"+" "+str(pu_counter)+" "+str(ss_counter)+\
                                       " "+str( self.loss_PU_SS[i, j] )
                    f.write(f_line)
                    ss_counter += 1
                pu_counter += 1


            #-----write the B: PR-SU path-loss
            #-----B PU_ID PR_ID SU_ID Path_loss
            pu_counter = 0
            su_counter = 0
            for i in n_pu:
                su_counter = 0
                for j in n_su:
                    for k in np.arange(0, self.pr_per_pu):
                        f_line = "\n"+"B"+" "+str(pu_counter)+" "+str( k )+" "+str( su_counter )+\
                                           " "+str( self.loss_PR_SU[i, k, j] )
                        f.write(f_line)
                    su_counter += 1
                pu_counter += 1

        print 'Extraction time:', np.round((mytimer() - start_t), 3), "seconds"
        return


    def processSURequest(self, suID = 0):
        '''
        allocate power to suID so that
        :param suID:
        :return:
        '''
        return

def runExperiment():
    '''

    :return:
    '''
    check_memory = False #<---debug

    if check_memory:
        h = hpy()

    sua = SUAllocation()
    ifname =  'data_small.txt'
    sua.parsePreGeneratedPathlossData(ifname=ifname)
    #sua.extractSmallerScenario(max_x= 8500., max_y = 8500., min_x = 7500., min_y =  7500.)
    sua.plotLocations()

    if check_memory:
        import pdb
        pdb.set_trace()
        print h.heap()
    return

if __name__ == '__main__':
    start_t = mytimer()

    profileCode = False #<--debug
    if profileCode:
        cProfile.run('runExperiment()', 'expProfile.cprof')
    else:
        runExperiment()
    print 'Execution time:', np.round( (mytimer() - start_t),3),"seconds"
