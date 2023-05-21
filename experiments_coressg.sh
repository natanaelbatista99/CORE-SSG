#!/bin/bash

DIR="dataset/dataset"

SEPARATOR=","

DEFAULT_EPSILON=0.06
DEFAULT_KMAX=200
DEFAULT_DIM=4
DEFAULT_POINTS=500
DEFAULT_DATA=500
DEFAULT_SUMMARIZATION=10


minpoints() {

    for minpoint in 100 150 200 250 300;
    do
        
        python main_experiments_coressg.py "${DIR}/${DEFAULT_POINTS}k-${DEFAULT_DIM}d.csv" ${minpoint} " " "core" ${DEFAULT_EPSILON} >> "experiment-coredb-minpoints.results"
        
    done
}

points() {

    for point in 100 250 500 750 1000;
    do
        
        python main_experiments_coressg.py "${DIR}/${point}k-${DEFAULT_DIM}d.csv" ${DEFAULT_KMAX} " " "core" ${DEFAULT_EPSILON} >> "experiment-coredb-points.results"
        
    done
}

dimensions() {

    for dimension in 2 4 6 8;
    do
        
        python main_experiments_coressg.py "${DIR}/${DEFAULT_POINTS}k-${dimension}d.csv" ${DEFAULT_KMAX} " " "core" ${DEFAULT_EPSILON} >> "experiment-coredb-dimension.results"
        
    done
}

summarizations() {

    for summarization in 0.074 0.06 0.052;
    do
        
        python main_experiments_coressg.py "${DIR}/${DEFAULT_POINTS}k-${DEFAULT_DIM}d.csv" ${DEFAULT_KMAX} " " "core" ${DEFAULT_EPSILON} >> "experiment-coredb-summarization.results"
        
    done
}

for i in $(seq 1)
do
	minpoints
    points
    dimensions
    summarizations
done
