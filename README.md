<h1 align="center">Hi 👋, This is the repository of M2O </h1>


<p align="center">
<img src="https://github.com/RodrigoGantier/M2O/blob/main/imgs/M2O%20logo.png" height="200" />
</p>

<h3 align="center">Many-to-One Training Scheme for Event-to-Video Reconstruction</h3>

You can try M2O on [Google Colab](https://colab.research.google.com/drive/1_a7sUYXiC94thhQ0fXRL1UvhddLt9Fje#scrollTo=aHITjhQP2HkC) 
[<img src="https://img.shields.io/badge/-colab-05122A?style=flat&logo=googlecolab"/>](https://colab.research.google.com/drive/1y3S9V3smWDdXoCqh_7dRofaONx0TS79E?usp=sharing)<br>

Once you are in Google Colab, make sure to run all sections. At the end, you can download the resulting video. The script in Google Colab reconstructs only one sequence due to permanent storage limitations. To reconstruct the entire test set, you can download the code and data locally.


<h3 align="center">The following images show some of our results</h3>


<p align="center">
<img src="https://github.com/RodrigoGantier/M2O/blob/main/imgs/hot_pixel.gif" height="200" />
</p>

Our M2O-E2VID model, trained with additional real event camera noise, is capable of removing hot pixels. 

<p align="center">
<img src="https://github.com/RodrigoGantier/M2O/blob/main/imgs/Checkerboard.gif" height="200" />
</p>


Additionally, the inclusion of the L2 loss reduces the checkerboard pattern.


# <img src="https://cdn.dribbble.com/users/1163047/screenshots/2697773/media/fb3030fceae825b853e91e747e11dc77.gif" width ="25"> <b>The training， validation and test set are on Baidu Netdisk</b> 


The train set is at this link:
[train_set](https://pan.baidu.com/s/1odWXhJtzdermrvBWrBl_6Q?pwd=1234)
pass：1234<br>
Place all training files (*.h5) inside the tr_m2o_data1 folder, as shown in the file tree below<br>

The validation set is at this link:
[val_set](https://pan.baidu.com/s/1CxEbCluxYqvObYwaICAP6g?pwd=1234) 
pass：1234<br>
Place all validation files (*.h5) inside the val_data folder, as shown in the file tree below<br>

├── M2O<br>
│   ├── ECD<br>
│   ├── datasets_path<br>
│   ├── imgs<br>
│   ├── models_<br>
│   ├── tr_m2o_data1<br>
│   │   ├── tr_000000000.h5<br>
│   │   ├── tr_000000000.h5<br>
│   │   ...<br>
│   │   └── tr_000005135.h5<br>
│   ├── utils_<br>
│   ├── val_data<br>
│   │   ├── 000000001.h5<br>
│   │   ├── 000000002.h5<br>
│   │   ├── 000000003.h5<br>
│   │   ├── 000000004.h5<br>
│   │   └── 000000005.h5<br>
└──   ...<br>


The test set is at [test set](https://pan.baidu.com/s/16J-iJdenhVUBrGb2klo_-A?pwd=1234)
pass：1234<br>
Place all test files (*.h5) inside the datasets_path folder


To run the code you need: <br>
python <= 3.10 <br>
pytorch <= 2.1.0 <br> 
numpy <= 1.24.4 <br>
opencv <= 4.9.0 <br>
tqdm <= 4.66.2 <br>
argparse <= 1.1 <br>




