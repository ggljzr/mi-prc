set terminal pngcairo font ',8' enhanced

set title "Doby běhů jednotlivých verzí algoritmu"

set key box bottom right
set autoscale
set xtics (10000,20000,50000,80000,100000,130000,170000,200000,230000,260000,300000)
set ytics auto
set logscale y

set ylabel "Doba běhu (milisekundy)"
set xlabel "Velikost instance"

set output "times.png"

plot "times" using 1:2 title "Sekvenční triviální algoritmus" with linespoints lt rgb "#7A8A5D" pointtype 7, \
"times" using 1:3 title "Paralelní triviální algoritmus" with linespoints lt rgb "#61878B" pointtype 7,\
"times" using 1:4 title "Paralelní Karatsubův algoritmus" with linespoints lt rgb "#7C5971" pointtype 7

set title "MFLOPS jednotlivých verzí algoritmu"

set key box bottom right
set autoscale
set xtics (10000,20000,50000,80000,100000,130000,170000,200000,230000,260000,300000)
set ytics auto

set ylabel "MFLOPS"
set xlabel "Velikost instance"

set output "flops.png"

plot "flops" using 1:2 title "Sekvenční triviální algoritmus" with linespoints lt rgb "#7A8A5D" pointtype 7, \
"flops" using 1:3 title "Paralelní triviální algoritmus" with linespoints lt rgb "#61878B" pointtype 7,\
"flops" using 1:4 title "Paralelní Karatsubův algoritmus" with linespoints lt rgb "#7C5971" pointtype 7