#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

/*N.B: non posso comparare un metodo di libreria con un metodo eseguito a mano quindi o comparo due metodi di libreria o scrivo sia sequenziale che kernel per
comparare.*/

/*Al momento ho fatto la prima operazione di preprocessing cioè la trasformazione in scala di grigio e l'ho fatta usando l'operazione sia su CPU che GPU,
in entrambi i casi ho usato la libreari quindi sono comparabili.*/

cv::Mat cpu_RGBtoGRAYSCALE(cv::Mat, float*);
cv::Mat cpu_resizeImage(cv::Mat,cv::Size, float*);

cv::cuda::GpuMat gpu_RGBtoGRAYSCALE(cv::cuda::GpuMat, cudaEvent_t*, float&);
cv::cuda::GpuMat gpu_resizeImage(cv::cuda::GpuMat, cv::Size size, cudaEvent_t*, float&);

//cv::Mat metodoHough è l'unico che ritorna l'output finale.

int main(int argn, char *argv[]) {
    //Variables
    cv::Mat cpu_grayscaleImage, cpu_resizedImage;
    cv::cuda::GpuMat gpu_grayscaleImage, gpu_resizedImage;
    cv::Mat output; //Final output image (downloaded from GPU)
    cudaEvent_t timer[2];
    cv::cuda::GpuMat gpuImage;
    float GPUelapsedTime, CPUelapsedTime;
    cv::Size size(600,600);

    //Read the input image
    cv::Mat input = cv::imread("foto.jpg");

    if (input.empty()) {
        fprintf(stderr, "Unable to load image\n");
        return -1;
    }

    //Loading of the image from the cpu to gpu
    gpuImage.upload(input);
    //Timer Evenet creation
    cudaEventCreate(&timer[0]);
    cudaEventCreate(&timer[1]);
    
    
    //RGB to Grayscale function (CPU)
    cpu_grayscaleImage = cpu_RGBtoGRAYSCALE(input, &CPUelapsedTime);
    printf("[RGB to Grayscale] Execution time on CPU: %f msec\n", CPUelapsedTime);

    //RGB to Grayscale function (GPU)
    gpu_grayscaleImage = gpu_RGBtoGRAYSCALE(gpuImage, timer, GPUelapsedTime);
    printf("[RGB to Grayscale] Execution time on GPU: %f msec\n", GPUelapsedTime);

    //Resize on CPU with GPU image as input
    cpu_resizedImage=cpu_resizeImage(cpu_grayscaleImage,size, &CPUelapsedTime);
    printf("[Resize] Execution time on CPU: %f msec\n", CPUelapsedTime);

    gpu_resizedImage= gpu_resizeImage(gpu_grayscaleImage,size, timer, GPUelapsedTime);
    printf("[Resize] Execution time on GPU: %f msec\n", GPUelapsedTime);

    //cv::imshow("Input image", input);
    //gpu_resizedImage.download(output);
    //cv::imshow("Resized and converted to grayscale image", output);
    //cv::waitKey(0);
    
    //The memory of cv::cuda::GpuMat and cv::Mat objects is automatically deallocated by the library
    cudaEventDestroy(timer[0]);
    cudaEventDestroy(timer[1]);
    return 0;
}



//Resize of the image using OpenCV (CPU)
cv::Mat cpu_resizeImage(cv::Mat in,cv::Size size, float *elapsedTime){
    cv::Mat out;
    clock_t start = clock();
    cv::resize(in, out, size);
    clock_t stop = clock();
    *elapsedTime = ((float)(stop - start)) / CLOCKS_PER_SEC * 1000;
    return out;
}
//Converting RGB to Grayscale using OpenCV (CPU)
cv::Mat cpu_RGBtoGRAYSCALE(cv::Mat in, float *elapsedTime){
    //Output image
    cv::Mat out;
    clock_t start = clock();
    //BGR to Grayscale
    cv::cvtColor(in,out,cv::COLOR_BGR2GRAY);
    clock_t stop = clock();
    *elapsedTime = ((float)(stop - start)) / CLOCKS_PER_SEC * 1000;
    return out;
}

//Converting RGB to Grayscale using OpenCV for CUDA (GPU)
cv::cuda::GpuMat gpu_RGBtoGRAYSCALE(cv::cuda::GpuMat gpuImage, cudaEvent_t* timer, float& elapsedTime){
    cv::cuda::GpuMat out;
    //Timer's start
    cudaEventRecord(timer[0], 0);
    //BGR to Grayscale
    cv::cuda::cvtColor(gpuImage,out,cv::COLOR_BGR2GRAY);
    //Timer's end
    cudaEventRecord(timer[1], 0);
    cudaEventSynchronize(timer[1]);
    //Elapsed time calculation
    cudaEventElapsedTime(&elapsedTime, timer[0], timer[1]);

    return out;
}

//Resize of the image using OpenCV for CUDA (GPU)
cv::cuda::GpuMat gpu_resizeImage(cv::cuda::GpuMat gpuImage, cv::Size size, cudaEvent_t* timer, float& elapsedTime){
    cv::cuda::GpuMat out;
    //Timer's start
    cudaEventRecord(timer[0], 0);
    cv::cuda::resize(gpuImage, out, size);
    //Timer's end
    cudaEventRecord(timer[1], 0);
    cudaEventSynchronize(timer[1]);
    //Elapsed time calculation
    cudaEventElapsedTime(&elapsedTime, timer[0], timer[1]);
    return out;
}