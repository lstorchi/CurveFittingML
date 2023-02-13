set key autotitle columnhead
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
#set xlabel "v Set"
set ylabel "Test Set MSE"
set grid
set key left top
set terminal pdf color enhanced 
#set term tikz standalone size 20cm, 14cm font ",14"
#set term cairolatex standalone size 10cm, 7cm
set output "dv3vibsetplot.pdf"
plot "Test2D dv=3 - Removing set of VIBs.tsv" u 8:xtic(1) lw 2 ti "GP", \
  "Test2D dv=3 - Removing set of VIBs.tsv" u 12:xtic(1) lw 2 ti "NN1", \
  "Test2D dv=3 - Removing set of VIBs.tsv" u 16:xtic(1) lw 2 ti "NN2", \
  "Test2D dv=3 - Removing set of VIBs.tsv" u 20:xtic(1) lw 2 ti "NN3"
#unset output
#system("pdflatex dv3rndplot")
