import numpy as np


def test_data():
    total_SS = 80000
    total_PU = 400
    total_SU = 500
    pr_per_pu = 5

    SS = np.zeros( ( total_SS, 3), dtype = float )
    PU = np.zeros( ( total_PU, 4), dtype=float )
    SU = np.zeros( ( total_SU, 3), dtype = float )
    PR = np.zeros( ( total_PU, pr_per_pu, 4 ), dtype = float )
    max_x = max_y = max_z = - float('inf')
    min_x = min_y = min_z =   float('inf')
    pr_count = 0
    x = y = z = 0

    with open('data_80k.txt', 'r') as f:
        for line in f:
            words = line.split()
            tx_type = words[0]
            if tx_type in ['PU','SS','SU']:
                indx = int(words[1])
                x = float(words[2])
                y = float(words[3])
                z = float(words[4])
                if tx_type == 'PU':
                    t = float(words[5])
                    PU[indx] = [x, y, z, t]
                elif tx_type == 'SU':
                    SU[indx] == [x, y, z]
                elif tx_type == 'SS':
                    SS[indx] == [x, y, z]
                # check
                if x < 0 or y < 0 or z < 0:
                    print line
            elif tx_type == 'PR':
                pu_indx = int(words[1])
                pr_indx = int(words[2])
                x = float(words[3])
                y = float(words[4])
                z = float(words[5])
                t = float(words[6])
                PR[pu_indx][pr_indx] = [x, y, z, t]
                #check
                if x <0 or y<0 or z<0: print line

            max_x, max_y, max_z = max(max_x, x), max(max_y, y), max(max_z, z)
            min_x, min_y, min_z = min(min_x, x), min(min_y, y), min(min_z, z)

        print PU.shape," data>>", PU[0, :]
        print SS.shape," data>>", SS[0, :]
        print SU.shape," data>>", SU[0, :]
        print PR.shape, " data>>", PR[0]

        print "max (x,y,z): ",max_x,", ",max_y,", ",max_z
        print "min (x,y,z): ",min_x,", ",min_y,", ",min_z
        print "pr_count: ", pr_count



if __name__ == '__main__':
    test_data()