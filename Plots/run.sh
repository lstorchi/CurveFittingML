for name in gnuplodv1random.gp gnuplodv1temp.gp gnuplodv1vib34.gp gnuplodv1vib36.gp gnuplodv1vib.gp gnuplodv1vibsets.gp \
	gnuplodv2random.gp gnuplodv2temp.gp gnuplodv2vib34.gp gnuplodv2vib36.gp gnuplodv2vib.gp gnuplodv2vibsets.gp \
	gnuplodv3random.gp gnuplodv3temp.gp gnuplodv3vib34.gp gnuplodv3vib36.gp gnuplodv3vib.gp gnuplodv3vibsets.gp
do 
   gnuplot < $name 
done
