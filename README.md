# EQ_vis Fusion Module readme
Feature-fusion based method for post-earthquake damage assessment with satellite imaging 

This is a readme file describing the operation and processing of data for the purposes of implementing **Fuse Module** solution from **Multi-step feature fusion for natural disaster damage assessment on satellite images** paper. The work was done by me ([Mateusz Żarski, PhD](https://www.iitis.pl/en/node/3227)) and my associate ([Jarosław A. Miszczak, DSc](https://iitis.pl/pl/person/jmiszczak)) at the [Institute of Theoretical and Applied Informatics of the Polish Academy of Sciences](https://www.iitis.pl/en). 

Our Fuse Module:

<img src="https://i.ibb.co/VtMdcbn/Fig2.png" alt="Fuse Module" width="600"/>

##  Table of contents

* [General info](#general-info)
* [Dependencies](#dependencies)
* [Directory structure](#directory-structure)
* [Usage](#usage)
* [Use examples](#use-examples)

# General Info

For the purposes of the paper, we have created a robust method of introducing our Fuse Module in any Deep Learning application using **Python 3** in which we utilized PyTorch for matrix operations. The Fuse Module operates in the task of two-images classification and verification by providing the exchange of information between two feature extraction paths horizontally and vertically. It can be applied to any DL model in single step before the classifier MLP or in multiple steps, after each feature extraction stage in the network.

Our CNN example using both Horizontal and Vertical fusion:

<img src="https://i.ibb.co/vJB6Dc0/Fig1.png" alt="Fuse HV" width="800"/>

In total, we used 13 fusion methods including:

 - Matrix batch multiplication
 - L1 and L2 norms,
 - MIN, MAX, MEAN operations,
 - etc.

A detailed description of the individual operations performed within the pipeline is described in the associated paper (will be uploaded after the publication). It also contains a description of the tests of other machine learning methods against which the solution from the project was compared.

## Dependencies

Our solution requires the following dependencies (packages in the latest version as of June 25, 2024, unless specified otherwise):

* PyTorch == 2.2
* Scikit-learn 
* Numpy == 1.16.2
* OpenCV == 4.4.0
* Matplotlib
* Progressbar 
* Imutils 
* Pillow

Python version 3.10 was used, but different versions will also probably work fine (but we didn't check them).

Also please note, that strings containing paths to folders in our Python scripts may need to be changed in order to run properly on your system (we did all the work on Linux machine, so check your backslash).

## Directory structure

To use the solution we propose, a certain directory structure should be maintained, that is also consistent with out repository structure. 

However, the most important files for usage are located in `cfgs` directory. There, you can place `.py` or `.yml` file containing the recipe for training (including model name that is consistent with `src.models` file, dataset path, learning rate, optimizer, scheduler, number of epochs, feature fusion methods etc.). 

For dataset path, provide string pointing to `B` tensors subset -- the subset should also contain the final class of the tensor in a manner: `{dataset}/i_B/{image_id}_{image_class}.png`.

## Usage

In order to use our solution with the dataset provided in Project or reproduce our results, cetrain steps have to be followed in order.

1. Put your dataset in a directory under `DS` and organize it in a fashion described above.
2. Prepare your training recipe in `.py` or `.yml` file, specifying correct data path and training parameters.
3. Run your training with `python3 main.py` for using default recipe, `python3 main.py {path_to_your_recipe}` to run training with single prepared recipe, or for performing training on a grid of parameters, simply run `sh runner.sh`, first modyfying `search_dir` to the path of your directory with multiple training recipes.

And thats it.

Your training will be logged with tensorboard in `lightning_logs` folder, and you can track the progress with tensorboard over your browser. At the end of the training, another log, containing results of the training will also be logged in `result_graphs` directory along with training parameters.

## Use examples

Below is an image showing example crops from the datasets we were using:

<img src="https://i.ibb.co/pZnKyrX/Fig3.png" alt="Dataset" width="800"/>

And some of the results we obtained with both our custom CNN model and predefined models using our Fuse Module:

<img src="https://i.ibb.co/Pc0Kky7/image-2024-06-25-141239247.png" alt="Dataset" width="400"/>
