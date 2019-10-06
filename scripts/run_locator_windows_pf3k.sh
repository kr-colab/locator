#!/bin/bash
#run locator in windows across the genome

cd /data0/cbattey2/locator/

step=200000
for chr in {01,02,03,04,05,06,07,08,09,10,11,12,13,14}
do
	echo "starting chromosome $chr"
	#get chromosome length
	header=`tabix -H /data0/cbattey2/locator/data/pf3k/SNP_INDEL_Pf3D7_$chr\_v3.combined.filtered.vcf.gz_FIELD.vcf.gz | grep "##contig=<ID=Pf3D7_$chr\_v3"`
	length=`echo $header | awk '{sub(/.*=/,"");sub(/>/,"");print}'` 
	
	#subset vcf by region and run locator
	endwindow=$step
	for startwindow in `seq 1 $step $length`
	do 
		echo "processing $startwindow to $endwindow"
		tabix -h /data0/cbattey2/locator/data/pf3k/SNP_INDEL_Pf3D7_$chr\_v3.combined.filtered.vcf.gz_FIELD.vcf.gz \
		Pf3D7_$chr\_v3\:$startwindow\-$endwindow > data/pf3k/tmp.vcf
		
		python scripts/locator.py \
		--vcf data/pf3k/tmp.vcf \
		--sample_data data/pf3k/pf3k_sample_data_train0.9.txt \
		--out out/pf3k/windows_200kb_ts0.9/$chr\_$startwindow\_$endwindow \
		--gpu_number 1 --impute_missing False 
		
		endwindow=$((endwindow+step))
		rm data/pf3k/tmp.vcf
	done
done