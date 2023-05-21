#!/bin/bash

DIR="dataset/dataset"

SEPARATOR=","

DEFAULT_KMAX=200
DEFAULT_DIM=4
DEFAULT_POINTS=500
DEFAULT_DATA=500
DEFAULT_SUMMARIZATION=10


minpoints() {

    for minpoint in 100 150 200 250 300;
    do        
        python main_experiments_coresg.py "${DIR}/${DEFAULT_POINTS}k-${DEFAULT_DIM}d.csv" ${minpoint} " " "core" >> "experiment-core-minpoints.results"
        
    done
}

points() {

    for point in 100 250 500 750 1000;
    do        
        python main_experiments_coresgdb.py "${DIR}/${point}k-${DEFAULT_DIM}d.csv" ${DEFAULT_KMAX} " " "core" >> "experiment-core-points.results"
        
    done
}

dimensions() {

    for dimension in 2 4 6 8;
    do        
        python main_experiments_coresgdb.py "${DIR}/${DEFAULT_POINTS}k-${dimension}d.csv" ${DEFAULT_KMAX} " " "core" >> "experiment-core-dimension.results"
        
    done
}


for i in $(seq 1)
do	
	minpoints
    points
    dimensions
done
