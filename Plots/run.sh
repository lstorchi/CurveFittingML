

for name in gnuplot_random.gp gnuplot_temp.gp gnuplot_vib34.gp gnuplot_vib36.gp gnuplot_vib.gp gnuplot_vibsets.gp 
do 
  sed s/dvN/dv1/ $name > /tmp/scritta
  sed s/dv=N/dv=1/ /tmp/scritta > torun.gp
  gnuplot < torun.gp
  rm -f torun.gp

  sed s/dvN/dv2/ $name > /tmp/scritta
  sed s/dv=N/dv=2/ /tmp/scritta > /tmp/torun1.gp
  sed s/\|33\,0/\|32\,0/ /tmp/torun1.gp > /tmp/torun2.gp
  sed s/\|35\,0/\|34\,0/ /tmp/torun2.gp > torun.gp
  gnuplot < torun.gp
  rm -f torun.gp

  sed s/dvN/dv3/ $name > /tmp/scritta
  sed s/dv=N/dv=3/ /tmp/scritta > /tmp/torun1.gp
  sed s/\|33\,0/\|31\,0/ /tmp/torun1.gp > /tmp/torun2.gp
  sed s/\|35\,0/\|33\,0/ /tmp/torun2.gp > torun.gp
 
  gnuplot < torun.gp
  rm -f torun.gp
done
