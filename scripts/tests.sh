#!/bin/bash
cd ~/locator

#slim tests
for i in {"dense5","dense2"}
do
  for j in {100,1000,10000,100000,300000}
  do
    for k in {3048,5930,3745,1005,4957}
    do
      echo sigma_0.45_$i\_$j\_$k
      python3 scripts/locator.py \
      --vcf data/sigma_0.45_rand500.vcf    \
      --sample_data data/sigma_0.45_rand500_samples.txt \
      --impute_missing False \
      --dropout_prop 0.5 \
      --model $i \
      --max_SNPs $j \
      --min_mac 2 \
      --outdir /data0/cbattey2/locator/out \
      --outname sigma_0.45_$i\_$j\_$k \
      --train_split 0.9 \
      --gpu_number 0 \
      --patience 10000 \
      --max_epochs 100000 \
      --seed $k \
      --plot True
    done
  done
done

for i in {"dense5","dense2"}
do
  for j in {100,1000,10000,100000,300000}
  do
    for k in {3048,5930,3745,1005,4957}
    do
      echo sigma_0.63_$i\_$j\_$k
      python3 scripts/locator.py \
      --vcf data/sigma_0.65_rand500.vcf    \
      --sample_data data/sigma_0.65_rand500_samples.txt \
      --impute_missing False \
      --dropout_prop 0.5 \
      --model $i \
      --max_SNPs $j \
      --min_mac 2 \
      --outdir /data0/cbattey2/locator/out \
      --outname sigma_0.63_$i\_$j\_$k \
      --train_split 0.9 \
      --gpu_number 3 \
      --patience 5000 \
      --max_epochs 100000 \
      --seed $k \
      --plot True
    done
  done
done

for i in {"dense5","dense2"}
do
  for j in {100,1000,10000,100000,300000}
  do
    for k in {3048,5930,3745,1005,4957}
    do
      echo sigma_1.29_$i\_$j\_$k
      python3 scripts/locator.py \
      --vcf data/sigma_1.29_rand500.vcf    \
      --sample_data data/sigma_1.29_rand500_samples.txt \
      --impute_missing False \
      --dropout_prop 0.5 \
      --model $i \
      --max_SNPs $j \
      --min_mac 2 \
      --outdir /data0/cbattey2/locator/out \
      --outname sigma_1.29_$i\_$j\_$k \
      --train_split 0.9 \
      --gpu_number 1 \
      --patience 5000 \
      --max_epochs 100000 \
      --seed $k \
      --plot True
    done
  done
done

#anopheles tests
for i in {"dense7","dense9","dense10"}
do
  for j in {100,1000,10000,100000,300000}
  do
    for k in {3048,5930,3745,1005,4957}
    do
      echo anopheles_2L15_$i\_$j\_$k
      python3 scripts/locator.py \
      --zarr data/ag1000g2L_1e6_to_5e6.zarr \
      --sample_data data/anopheles_samples_sp.txt \
      --impute_missing True \
      --dropout_prop 0.5 \
      --model $i \
      --max_SNPs $j \
      --min_mac 2 \
      --outdir /data0/cbattey2/locator/out \
      --outname anopheles_2L15_$i\_$j\_$k \
      --train_split 0.9 \
      --gpu_number 2 \
      --patience 500 \
      --seed $k \
      --plot True
    done
  done
done

#pabu tests
for i in {"dense5","dense2"}
do
  for k in {3048,5930,3745,1005,4957,9084,1837,2975,8503,1029}
  do
    echo pabu_c48h40_$i\_$k
    python3 scripts/locator.py \
    --vcf data/pabu_c48h40.vcf    \
    --sample_data data/pabu_sample_data.txt \
    --impute_missing True \
    --mode predict \
    --model $i \
    --min_mac 2 \
    --outdir /data0/cbattey2/locator/out \
    --outname pabu_c48h40_$i\_$k \
    --train_split 0.5 \
    --gpu_number 1 \
    --patience 10000 \
    --max_epochs 100000 \
    --seed $k \
    --plot True
  done
done
