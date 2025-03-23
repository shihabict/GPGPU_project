#!/usr/bin/env bash


# Delete previous output from PBS
rm -rf *.qsub_out

# Submit the job to the queue with the matrix dimension as an argument
qsub submission.pbs

# Wait for the job to get picked up and start producing output
until [ -f *.qsub_out ]
do
  sleep 1
done

# Open the output file and follow the file as new output is added
less +F *.qsub_out
