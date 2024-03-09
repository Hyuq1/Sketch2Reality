# Sketch2Reality：Precision-Friendly 3D Generation Technology
![image](https://github.com/Hyuq1/Sketch2Reality/blob/main/img/network.png)
# Environment
`git clone https://github.com/Hyuq1/Sketch2Reality`

python >= 3.6

pytorch >=1.8.0

install dependencies:`pip install -r requirements.txt`

build and install Soft Rasterizer
# Data
Our data includes mesh models, input sketches, and SDF samples.
We provide our own dataset for the [car](https://github.com/Hyuq1/Sketch2Reality/edit/main/README.md) class. You can simply download and unzip it into the `/data` folder to get going.the corresponding training [weights]().
# Mesh data
We use data from [ShapeNetCore.v1](https://shapenet.org/).

# SDF data
We use process_data in [DeepSDF](https://github.com/facebookresearch/DeepSDF) to perform SDF sampling on the mesh of the ShapeNetCore.v1 dataset.

# Sketch data
We use the canny algorithm to extract lines from the Image in [DISN](https://github.com/laughtervv/DISN) to obtain the input synthetic sketches.

# Trainning
you can training a model for car:

`python train.py -e experiments/car_new`

After the model is trained, you can refine the rough model by running:

`python refine.py -e experiments/car_new`

Finally, you can test chamfer distance and Voxel IoU by running:

`python CHD_normalize.py`

`python compute_IoU.py`
