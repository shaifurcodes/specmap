clear
reset
unset key

set output 'tx_1_rdb.png'
set terminal pngcairo size 800, 600 enhanced font "Arial-Bold, 24" lw 4

ifname='tx_1_rdb.txt'
set size ratio -1
set palette defined (0 0 0 0.5, 1 0 0 1, 2 0 0.5 1, 3 0 1 1, 4 0.5 1 0.5, 5 1 1 0, 6 1 0.5 0, 7 1 0 0, 8 0.5 0 0)
set view map
set dgrid3d
set pm3d interpolate 0,0

set xtics ("0" 0, "1" 25, "2" 50, "3" 75, "4" 99 ) offset 0,1
set xlabel "km" offset 0, 1.8

set ytics ("0" 0, "1" 25, "2" 50, "3" 75, "4" 99 ) offset 1,0
set ylabel "km" offset 0, 0
set cblabel "Received Power (dBm)"
set contour base
set cntrparam bspline
set cntrparam points 700
set cntrparam order 10
set cntrparam levels discrete -90.0
splot ifname matrix with pm3d

