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

cv::Mat cpu_RGBtoGRAYSCALE(cv::Mat);
cv::Mat cpu_resizeImage(cv::Mat,cv::Size);

cv::cuda::GpuMat gpu_RGBtoGRAYSCALE(cv::cuda::GpuMat);
cv::cuda::GpuMat gpu_resizeImage(cv::cuda::GpuMat, cv::Size size);

//cv::Mat metodoHough è l'unico che ritorna l'output finale.

int main(int argn, char *argv[]) {
    cv::Mat cpu_grayscaleImage, cpu_resizedImage;
    cv::cuda::GpuMat gpu_grayscaleImage, gpu_resizedImage;
    cv::Mat output; //Final output image (downloaded from GPU)
    cudaEvent_t Start, Stop;
    cv::cuda::GpuMat gpuImage;
    cv::Size size(600,600);
    //Read the input image
    cv::Mat input = cv::imread("foto.jpg");

    if (input.empty()) {
        fprintf(stderr, "Unable to load image\n");
        return -1;
    }

    //Loading of the image from the cpu to gpu
    gpuImage.upload(input);
    
    //Time measurement CPU
    cudaEventCreate(&Start);
    cudaEventCreate(&Stop);
    cudaEventRecord(Start, 0);
    //RGB to Grayscale function (CPU)
    cpu_grayscaleImage = cpu_RGBtoGRAYSCALE(input);
    cudaEventRecord(Stop, 0);
    cudaEventSynchronize(Stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, Start, Stop);
    printf("[RGB to Grayscale] Execution time on CPU: %f msec\n", elapsedTime);

    elapsedTime=0;
    //Time measurement GPU
    cudaEventRecord(Start, 0);
    //RGB to Grayscale function (GPU)
    gpu_grayscaleImage = gpu_RGBtoGRAYSCALE(gpuImage);
    cudaEventRecord(Stop, 0);
    cudaEventSynchronize(Stop);
    cudaEventElapsedTime(&elapsedTime, Start, Stop);
    printf("[RGB to Grayscale] Execution time on GPU: %f msec\n", elapsedTime);

    elapsedTime=0;
    cudaEventRecord(Start, 0);
    //Resize on CPU with GPU image as input
    cpu_resizedImage=cpu_resizeImage(cpu_grayscaleImage,size);
    cudaEventRecord(Stop, 0);
    cudaEventSynchronize(Stop);
    cudaEventElapsedTime(&elapsedTime, Start, Stop);
    printf("[Resize] Execution time on CPU: %f msec\n", elapsedTime);

    gpu_resizedImage= gpu_resizeImage(gpu_grayscaleImage,size);


    cv::imshow("Input image", input);
    
    gpu_resizedImage.download(output);
    cv::imshow("Resized and converted to grayscale image", output);
    cv::waitKey(0);

    cudaEventDestroy(Start);
    cudaEventDestroy(Stop);
    return 0;
}



//Resize of the image using OpenCV (CPU)
cv::Mat cpu_resizeImage(cv::Mat in,cv::Size size){
    cv::Mat out;
    cv::resize(in, out, size);
    return out;
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
cv::cuda::GpuMat gpu_RGBtoGRAYSCALE(cv::cuda::GpuMat gpuImage){
    cv::cuda::GpuMat out;
    //BGR to Grayscale
    cv::cuda::cvtColor(gpuImage,out,cv::COLOR_BGR2GRAY);
    return out;
}

cv::cuda::GpuMat gpu_resizeImage(cv::cuda::GpuMat gpuImage, cv::Size size){
    cv::cuda::GpuMat out;
    cv::cuda::resize(gpuImage, out, size);
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