set key autotitle columnhead
set xlabel "T (K)"
set ylabel "log_{10}(k(34,0|33,0))"
set grid
set key left top
set terminal pdf color enhanced 
#set term tikz standalone size 20cm, 14cm font ",14"
set output "dvNv34plot.pdf"
plot "Test2D dv=N - new vib data V=34.tsv" u 1:5 w p pt 6 lw 1 ti "QM", \
  "Test2D dv=N - new vib data V=34.tsv" u 1:6 w l lw 2 ti "GP", \
  "Test2D dv=N - new vib data V=34.tsv" u 1:7 w l lw 2 ti "NN1"
unset output
#system("pdflatex dvNv34plot")
