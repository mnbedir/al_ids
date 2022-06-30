#!/bin/bash

if [[ $# -ge 1 ]]; then
	base_folder=$1
	echo Submitting batch of tests using $base_folder
else
  echo "Provide base folder that contains test configurations"
  exit
fi


for folder_path in $base_folder/*
do
	sbatch submit_test_arg.sh $folder_path
done

