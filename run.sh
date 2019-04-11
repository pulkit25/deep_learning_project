source activate tensorflow_p36
mkdir project
cd project
git clone https://github.com/eriklindernoren/Keras-GAN.git
mkdir /run/user/1000/dataset
wget -O /run/user/1000/dataset/val2017.zip http://images.cocodataset.org/zips/val2017.zip
unzip /run/user/1000/dataset/val2017.zip -d /run/user/1000/dataset/
git clone https://www.github.com/keras-team/keras-contrib.git
cd keras-contrib
python setup.py install
cd ../../
chown -R ubuntu project
chown -R ubuntu /run/user/1000/dataset/


