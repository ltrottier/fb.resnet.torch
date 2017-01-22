#!/bin/sh
mkdir results/drop_5_3
nohup th main.lua -depth 20 -batchSize 16 -nEpochs 200 -nGPU 1 -nThreads 2 -netType preresnet-drop -dataset cifar10 -save results/drop_5_3 -resume results/drop_5_3 >> results/drop_5_3/results.out &
