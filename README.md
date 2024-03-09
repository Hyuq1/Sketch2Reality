# Sketch2Reality：Precision-Friendly 3D Generation Technology
# Environment
`git clone https://github.com/Hyuq1/Sketch2Reality`

python >= 3.6

pytorch >=1.8.0

install dependencies:`pip install -r requirements.txt`

build and install Soft Rasterizer
# Data
Our data includes mesh models, input sketches, and SDF samples.
We provide our own dataset for the [car](https://github.com/Hyuq1/Sketch2Reality/edit/main/README.md) class and the corresponding training weights. You can simply download and unzip it into the `/data` folder to get going.
# Trainning
you can training a model for car:

`python train.py -e experiments/car_new`

After the model is trained, you can refine the rough model by running:

`python reconstruct_last_sym.py`

Finally, you can test chamfer distance and Voxel IoU by running:

`python CHD_normalize.py`

`python compute_IoU.py`
