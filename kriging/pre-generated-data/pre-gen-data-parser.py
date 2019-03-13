import numpy as np
import cProfile
from timeit import default_timer as mytimer
from guppy import hpy

class SUAllocation(object):
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

        with open('data_80k.txt', 'r') as f:
            for line in f:
                words = line.split()
                tx_type = words[0]
                if tx_type in ['PU', 'SS', 'SU']:
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


def runExperiment():
    '''

    :return:
    '''
    check_memory = False #<---debug

    if check_memory:
        h = hpy()

    sua = SUAllocation()
    sua.parsePreGeneratedPathlossData()
    print sua.PU.shape, " data>>", sua.PU[0:3, :]
    print sua.SS.shape, " data>>", sua.SS[0:3, :] #
    print sua.SU.shape, " data>>", sua.SU[0:3, :] #
    print sua.PR.shape, " data>>", sua.PR[ 0 ]
    print sua.loss_PR_SU.shape, " data>>", sua.loss_PR_SU[0:2, 0:, 0] #
    print sua.loss_PU_SS.shape, " data>>", sua.loss_PU_SS[0:2, 0:2]  #
    if check_memory:
        import pdb
        pdb.set_trace()
        print h.heap()

if __name__ == '__main__':
    start_t = mytimer()

    profileCode = False #<--debug
    if profileCode:
        cProfile.run('runExperiment()', 'expProfile.cprof')
    else:
        runExperiment()
    print 'Execution time:', np.round( (mytimer() - start_t),3),"seconds"
