import pstats
if __name__ == '__main__':
    profile_filename = 'expProfile.cprof'
    p = pstats.Stats(profile_filename)
    p.sort_stats('time').print_stats()
