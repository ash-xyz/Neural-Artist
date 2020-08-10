#! /bin/bash
set -x
set -e

mkdir data
cd data 
wget http://images.cocodataset.org/zips/train2014.zip
unzip -q train2014.zip
rm train2014.zip
mv train2014 train
if [[ -f train ]]
then
    echo "Download Complete."
fi