#!/usr/bin/env bash

# Check if both input arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <input_image_path> <input_image_meta_path>"
  exit 1
fi

# Store the input arguments
INPUT_IMAGE=$1
INPUT_META=$2

# Delete previous output from PBS
rm -rf *.qsub_out

# Submit the job to the queue with the input arguments
qsub -v INPUT_IMAGE="$INPUT_IMAGE",INPUT_META="$INPUT_META" submission.pbs

# Wait for the job to get picked up and start producing output
until [ -f *.qsub_out ]
do
  sleep 1
done

# Open the output file and follow the file as new output is added
less +F *.qsub_out

