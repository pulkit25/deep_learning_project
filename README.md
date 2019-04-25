Super-resolution of DIV2K using GAN

The dataset can be downloaded from: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
Other datasets can also be used. No changes need to be done to the code. 

If you have a different machine setup, do not execute the run.sh file. Set up the server and datasets and then run the srgan.py file. Instruction on running the script on Ubuntu GPU machine AWS for deep learning(p3.2xlarge):
1. Execute the run.sh file.
2. It sets up the tensorflow_p36 anaconda environment
3. It creates the folder project in current directory
4. Clones the codes from https://github.com/pulkit25/deep_learning_project.git
5. It gets the data and unzips it into the dataset folder in /dev/shm/dataset/DIV2K_train_HR
6. Installs keras_contrib in the active environment (tensorflow_p36 in this case)
7. Gives access permissions to the folders created by the script
8. Run the script using command 'python srgan.py --datadir your/data/dir --outputdir your/output/dir --upscale upscaling_value(int) --inputdim lowres_img_dimension(considered to be a square image, int) --nresblocks number_of_residual_blocks(int) --lr learning_rate(float) --epochs number_epochs(int) --batch_size batch_size(int) --pretrain 0/1(whether to pretrain with SRResNet or take existing pretrained model, int) --load_img_cnt number_of_images_for_SRResNet_output(int)'. Remove any argument to use defaults.

Last Updated: 04/25/2019