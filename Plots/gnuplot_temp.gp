set key autotitle columnhead
#set key noenhanced
#set style data linespoints
set style data lines
#set style data histogram
#set style histogram cluster gap 1
#set style fill solid border -1
#set boxwidth 0.9
set xtic rotate by -45 scale 0
set xlabel "T (K)"
set ylabel "Test Set MSE"
set grid
set terminal pdf color enhanced 
#set term tikz standalone size 20cm, 14cm font ",14"
set output "dvNtempplot.pdf"
#plot "Test2D dv=N - Remove TEMP.tsv" u 8:xtic(1) lw 2 ti "GP", \
#  "Test2D dv=N - Remove TEMP.tsv" u 12:xtic(1) lw 2 ti "NN1", \
#  "Test2D dv=N - Remove TEMP.tsv" u 16:xtic(1) lw 2 ti "NN2", \
#  "Test2D dv=N - Remove TEMP.tsv" u 20:xtic(1) lw 2 ti "NN3"
plot "Test2D dv=1 - Remove TEMP.tsv" u 1:8 lw 2 ti "GP", \
  "Test2D dv=N - Remove TEMP.tsv" u 1:12 lw 2 ti "NN1", \
  "Test2D dv=N - Remove TEMP.tsv" u 1:20 lw 2 ti "NN2"
unset output
#system("pdflatex dvNtempplot")
