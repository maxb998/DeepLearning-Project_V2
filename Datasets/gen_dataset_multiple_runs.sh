#!/bin/bash

for i in {1..40}
do
   python dataset_creator2.py -o Generated_Dataset/train/ -b Professor_Material/backgrounds/ -n 100 -t 200 -f 100
done