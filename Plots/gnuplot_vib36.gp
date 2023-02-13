set key autotitle columnhead
set xlabel "T"
set ylabel "Rate Coefficient"
set grid
set key left top
set terminal pdf color enhanced 
#set term tikz standalone size 20cm, 14cm font ",14"
set output "dvNv36plot.pdf"
plot "Test2D dv=N - new vib data V=36.tsv" u 1:5 w p pt 6 lw 1 ti "QM", \
  "Test2D dv=N - new vib data V=36.tsv" u 1:6 w l lw 2 ti "GP", \
  "Test2D dv=N - new vib data V=36.tsv" u 1:7 w l lw 2 ti "NN1"
unset output
#system("pdflatex dvNv36plot")
