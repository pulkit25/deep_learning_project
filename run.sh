#!/usr/bin/env bash
source activate tensorflow_p36
mkdir project
cd project
git clone https://github.com/pulkit25/deep_learning_project.git
mkdir /dev/shm/dataset
wget -O /dev/shm/dataset/DIV2K_train_HR.zip http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip /dev/shm/dataset/DIV2K_train_HR.zip -d /dev/shm/dataset/
git clone https://www.github.com/keras-team/keras-contrib.git
cd keras-contrib
python setup.py install
cd ../../
chown -R ubuntu project
chown -R ubuntu /dev/shm/dataset/
cd project/deep_learning_project