# Rendering and Reconstruction Based 3D Portrait Stylization

Code for our paper:
> **Rendering and Reconstruction Based 3D Portrait Stylization**
> <br>Shaoxu Li, Ye Pan<br>
> Accepted by ICME2023

Our code is based on code from DST's pytorch code.

## Dependencies
Python 3 (e.g. python 3.7.3)
pytorch3d, pytorch, torchvision, cudatoolkit, numpy, matplotlib,opencv-python
```
conda create -n RRPS python=3.7
conda activate RRPS
conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.2
conda install pytorch3d -c pytorch3d
conda install matplotlib
pip install opencv-python
```

## Usage
A simple demo is supplied for running with my code. More instructions are waiting for updates.
### 1. Model acquisition
	
A content model is required for the following procedures. '.obj' is used in our code.
	
For human faces, "Deep 3d portrait from a single image" is recommended to accomplish the 3D reconstruction from images to haired face models. Code not included.

We offer 5 models in  "../example/models/", which are generated using the method from "Deep 3d portrait from a single image".

### 2. Run rendering to get content images and parameters
```
python rendering.py
```
Input folder is "../example/models/".
Output folder is "../example/content/".

### 3. Get correspondences

4 pairs of results have been provided for step 4. "../example/content/"

For human face style transfer, Dlib is used for real human landmarks detection, the "face-of-art" is used for caricatures and other artistic images. Code not included.
	
For arbitrary objects, Neural Best-Buddies(NBB) is recommended.

### 4. Run image style transfer and 3D reconstruction
```
python run.py
```
Input folder are "../example/content/" and "../example/style/".
Output folder is "../output/".	
Using our code, all content are randomly assigned two styles for stylization. And parameters could be adjusted for different degrees of stylization. We offer 11 contents in  "../example/content/".
### 5. Model somooth
Here meshlab is used to smooth the final result. After testing some Taubin by different versions, that's most suitable for our work.
	
Import model -> Filters -> Smoothing, Fairing and Deformation -> Taubin Smooth (Lambda = 0.7  mu = -0.53  Smoothing steps = 20)
## Acknowledgment
### Our code is based on code from the following paper:

Deformable style transfer. ECCV, 2020. (https://github.com/sunniesuhyoung/DST)

### Related data are mainly from the following papers:

The face of art: Landmark detection and geometric style in portraits. ACM TOG (https://faculty.idc.ac.il/arik/site/foa/artistic-faces-dataset.asp)

Progressive growing of gans for improved quality, stability, and variation. (https://github.com/tkarras/progressive_growing_of_gans)

### For more exploration, you may need the following papers: 

Deep 3d portrait from a single image. CVPR2020. (https://github.com/sicxu/Deep3dPortrait)

Landmark Detection and 3D Face Reconstruction for Caricature Using a Nonlinear Parametric Model. Graphical Models. (https://github.com/Juyong/CaricatureFace)

Style Transfer by Relaxed Optimal Transport and Self-Similarity. CVPR 2019. (https://github.com/nkolkin13/STROTSS)

WarpGAN: Automatic Caricature Generation. CVPR 2019. (https://github.com/seasonSH/WarpGAN)

Neural Best-Buddies: Sparse Cross-Domain Correspondence. SIGGRAPH 2018. (https://github.com/kfiraberman/neural_best_buddies)


