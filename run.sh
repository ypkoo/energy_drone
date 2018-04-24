#!/bin/sh


for s in 1 2
do
    for hs in 140 150 160 170 180
    do
      for d in 4 5 6
      do
        for b in 25 32 40
        do
          python main.py --depth $d --seed $s --h_size $hs --b_s $b
        done
      done
    done
done