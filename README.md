# Hough Transform with CUDA
***
This project is the result of the **High Performance Computing** exam of the Master's Degree Course in Machine Learning and Big Data (LM-18) of the *University of Naples Parthenopoe*.


## Goal
The final objective of this project is to learn the main notions of parallelization on GPUs in particular using the **CUDA** environment; to do this we decided to solve some problems that can be solved more efficiently on GPU.
In the case of this repository the objective is to apply the Hough Transform to identify the lines.

The **idea** is to apply some preprocessing operations before carrying out the algorithm, in particular:
- *Converting image to grayscale*: this operation is done by the **OpenCV2** library for Cuda.
- *Resize*: larger images may be more sensitive to noise. The library function is used.
- *Histogram equalization*: allows to improve the contrast of the lines, increasing the visibility of the narrows. Cuda kernel code is written.
- *Gaussian Blurring*: for noise reduction, this is a fundamental step given that the algorithm is based on a voting strategy and doing a Gaussian blurring (reducing noise and blurring details) allows us to clean it from noise and therefore reduce the voting problem spurious, i.e. cells that accumulate spurious votes. The library function is used.

<u>*Operations are performed on both CPU and GPU for comparison purposes*</u>

## Instructions 🚀
To run and test the code you will need:
- Download the repository.
- Through the terminal go to the local folder where the downloaded repository is located.
- Run the command `./compile.sh image.cu out`.
- The program will start automatically, for a new execution it will not be necessary to use the compile.sh file but simply execute the `./out` command.

**N.B**: *to run the software a local installation of OpenCV and OpenCV for cuda is required, therefore an Nvidia graphics card is required*. 
## Tools 🛠
- [Cuda](https://developer.nvidia.com/cuda-toolkit)
- [C](https://en.wikipedia.org/wiki/C_(programming_language) )
- [OpenCV](https://opencv.org/)
- [OpenCV for Cuda](https://opencv.org/platforms/cuda/)

## Contacts 🪪
- [mail] renato [ dot ] esposito001 [ at ] studenti [ dot ] uniparthenope [ dot ] it (you can write to me in english or italian).

For more details, read the [report](./Peport.pdf).
