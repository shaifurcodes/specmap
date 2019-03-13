import pstats

if __name__ == '__main__':
    cprof_fname = 'expProfile.cprof'
    p = pstats.Stats( cprof_fname )
    p.strip_dirs().sort_stats(-1).print_stats()