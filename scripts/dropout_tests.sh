#!/bin/bash
cd ~/locator

for i in {"dense1","dense2","dense3","dense4"}
do
  for j in {0.5,0.25,0.1}
  do
    for k in {1..5}
    do
      python3 scripts/locator.py \
      --vcf data/ag1000g2L_1e6_to_3e6.vcf.gz \
      --sample_data data/anopheles_samples_sp.txt \
      --dropout_prop $j \
      --model $i \
      --outdir out \
      --outname anopheles_2L13_$i\_$j\_$k \
      --train_split 0.85 \
      --gpu_number 1 \
      --patience 500 \
      --seed 12345
    done
  done
done

for i in {"dense1","dense2","dense3","dense4"}
do
  for j in {0.5,0.25,0.1}
  do
    for k in {1..5}
    do
      python3 scripts/locator.py \
      --vcf data/ag1000g2L_1e6_to_5e6.vcf.gz \
      --sample_data data/anopheles_samples_sp.txt \
      --dropout_prop $j \
      --model $i \
      --outdir out \
      --outname anopheles_2L15_$i\_$j\_$k \
      --train_split 0.85 \
      --gpu_number 1 \
      --patience 500 \
      --seed 12345
    done
  done
done

for i in {"dense1","dense2","dense3","dense4"}
do
  for j in {0.5,0.1}
  do
    for k in {1}
    do
      python3 scripts/locator.py \
      --vcf data/ag1000g2L_0_to_1e7.vcf.gz \
      --sample_data data/anopheles_samples_sp.txt \
      --dropout_prop $j \
      --model $i \
      --outdir out \
      --outname anopheles_2L15_$i\_$j\_$k \
      --train_split 0.85 \
      --gpu_number 1 \
      --patience 500 \
      --seed 12345
    done
  done
done
