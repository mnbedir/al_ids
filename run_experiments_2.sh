#!/bin/bash
if [[ $# -ge 2 ]]; then
  scriptFile=$1
  input=$2
  echo Running test using $scriptFile, test file $input
else
  echo Please provide text file that contains config parameters
  echo Run this script using following command
  echo "nohup ./run_experiments.sh {test_script} {PARAMETERS_FILE} > run_experiments.log 2>&1 &"
  exit
fi


dt=$(date '+%Y_%m_%d--%H_%M_%S');

rm -rf results
mkdir results


i=0
for filepath in $input/*
do
	i=$((i+1))
	echo "[$i] -> python3 $scriptFile $filepath"
	python $scriptFile $filepath > results/execution_logs_$i.log 2>&1
done

hostInfo=$(hostname)
tar -cjvf results_$dt.tar.gz results 
mv results_$dt.tar.gz experiments2/ 
git pull
git add experiments2/results_$dt.tar.gz
git commit -m "Added experiment $scriptFile $input -- $dt from $hostInfo"
git push


