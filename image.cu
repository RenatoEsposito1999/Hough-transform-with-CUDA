#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

/*N.B: non posso comparare un metodo di libreria con un metodo eseguito a mano quindi o comparo due metodi di libreria o scrivo sia sequenziale che kernel per
comparare.*/

/*Al momento ho fatto la prima operazione di preprocessing cioè la trasformazione in scala di grigio e l'ho fatta usando l'operazione sia su CPU che GPU,
in entrambi i casi ho usato la libreari quindi sono comparabili.*/

/*Prossimo step: prendere i tempi e stamparli*/
cv::Mat cpu_RGBtoGRAYSCALE(cv::Mat);
cv::Mat gpu_RGBtoGRAYSCALE(cv::Mat);

int main() {
    cv::Mat cpu_outputGrayScale;
    cv::Mat gpu_outputGrayScale;
    cudaEvent_t Start, Stop;
    //Read the input image
    cv::Mat input = cv::imread("foto.jpg");

    if (input.empty()) {
        fprintf(stderr, "Unable to load image\n");
        return -1;
    }
    //Time measurement CPU
    cudaEventCreate(&Start);
    cudaEventCreate(&Stop);
    cudaEventRecord(Start, 0);
    //RGB to Grayscale function (CPU)
    cpu_outputGrayScale = cpu_RGBtoGRAYSCALE(input);
    cudaEventRecord(Stop, 0);
    cudaEventSynchronize(Stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, Start, Stop);
    printf("Execution time on CPU: %f msec\n", elapsedTime);

    elapsedTime=0;
    //Time measurement GPU
    cudaEventRecord(Start, 0);
    //RGB to Grayscale function (GPU)
    gpu_outputGrayScale = gpu_RGBtoGRAYSCALE(input);
    cudaEventRecord(Stop, 0);
    cudaEventSynchronize(Stop);
    cudaEventElapsedTime(&elapsedTime, Start, Stop);
    printf("Execution time on GPU: %f msec\n", elapsedTime);
    return 0;
}

//Converting RGB to Grayscale using OpenCV (CPU)
cv::Mat cpu_RGBtoGRAYSCALE(cv::Mat in){
    //Output image
    cv::Mat out;
    //BGR to Grayscale
    cv::cvtColor(in,out,cv::COLOR_BGR2GRAY);
    return out;
}

//Converting RGB to Grayscale using OpenCV for CUDA (GPU)
cv::Mat gpu_RGBtoGRAYSCALE(cv::Mat in){
    //Output image
    cv::Mat out;
    
    //Grayscale image on GPU
    cv::cuda::GpuMat gray;

    //Loading of the image from the cpu to gpu
    cv::cuda::GpuMat gpuImage(in);
    //BGR to Grayscale
    cv::cuda::cvtColor(gpuImage,gray,cv::COLOR_BGR2GRAY);

    //Download from gpu to cpu
    gray.download(out);

    return out;
}


    /*

    //Saving to disk
    //cv::imwrite("GrayscaleImageGPU.jpg",gpu_outputGrayScale);
    //cv::imshow("GrayscaleImageGPU", gpu_outputGrayScale);
    //cv::waitKey(0);
    // Trasferisci l'immagine sulla memoria GPU
    cv::cuda::GpuMat gpuInput;
    cv::cuda::GpuMat gpuGrayImage;

    gpuInput.upload(input);
    // Esegui operazioni di elaborazione dell'immagine sulla GPU (ad esempio, cv::cuda::resize)
    cv::cuda::resize(gpuInput, gpuInput, cv::Size(640, 480));
    
    gpuGrayImage.create(gpuInput.size(), CV_8UC1);  // CV_8UC1 indica un singolo canale di 8 bit per pixel
    cv::cuda::cvtColor(gpuInput, gpuGrayImage, cv::COLOR_BGR2GRAY);
    // Trasferisci l'immagine elaborata dalla GPU alla CPU
    cv::Mat output;
    gpuGrayImage.download(output);

    // Visualizza l'immagine originale e quella elaborata
    //cv::imshow("Input", input);
    cv::imwrite("output.jpg", output);
    //printf("Input\n");
    //cv::imshow("Output.jpg", output);
    //printf("Output\n");
    //cv::waitKey(0);
    */