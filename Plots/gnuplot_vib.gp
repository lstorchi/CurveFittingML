set key autotitle columnhead
#set key noenhanced
#set style data linespoints
set style data lines
set xlabel "v"
set ylabel "Test Set MSE"
set grid
set terminal pdf color enhanced 
#set output "plot.pdf"
#set term tikz standalone size 20cm, 14cm font ",14"
set output "dvNvibplot.pdf"
plot "Test2D dv=N - Remove VIB.tsv" u 1:8 lw 2 ti "GP", \
  "Test2D dv=N - Remove VIB.tsv" u 1:12 lw 2 ti "NN1", \
  "Test2D dv=N - Remove VIB.tsv" u 1:16 lw 2 ti "NN2", \
  "Test2D dv=N - Remove VIB.tsv" u 1:20 lw 2 ti "NN3"
unset output
#system("pdflatex dvNvibplot")
