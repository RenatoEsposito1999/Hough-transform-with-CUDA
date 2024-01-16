#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui/highgui.hpp>
//Mettere l'istogramma cumulativo nella SM, quindi provare entrambi gli approcci.
//Scrivere i tempi su una tabella magari grafica.
//trovre un numero di thread ottimale
//migliorare la formula dell'equalizzazione.


cv::Mat cpu_RGBtoGRAYSCALE(cv::Mat, float*);
cv::Mat cpu_resizeImage(cv::Mat,cv::Size, float*);
cv::Mat cpu_equalization(cv::Mat, int*, float*);
void calcCumHist(cv::Mat, int*);
cv::Mat cpu_HoughTransformLine(cv::Mat, float *); 

cv::cuda::GpuMat gpu_RGBtoGRAYSCALE(cv::cuda::GpuMat, cudaEvent_t*, float&);
cv::cuda::GpuMat gpu_resizeImage(cv::cuda::GpuMat, cv::Size size, cudaEvent_t*, float&);
cv::cuda::GpuMat equalizeHistOnGPU(cv::cuda::GpuMat, cv::Mat);
__global__ void equalizeHistCUDA(unsigned char*, unsigned char*,int* , int, int);




int main(int argn, char *argv[]){
    //Variables
    cv::Mat cpu_grayscaleImage, cpu_resizedImage, cpu_equalizedImage;
    cv::cuda::GpuMat gpuImage, gpu_grayscaleImage, gpu_resizedImage;
    int cumHist[256]={0};
    int *cumHist_device;
    dim3 nThreadPerBlocco, numBlocks;
    cudaEvent_t timer[2];
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

    //Resize on CPU with CPU image as input
    cpu_resizedImage=cpu_resizeImage(cpu_grayscaleImage,size, &CPUelapsedTime);
    printf("[Resize] Execution time on CPU: %f msec\n", CPUelapsedTime);

    //Resize on GPU
    gpu_resizedImage = gpu_resizeImage(gpu_grayscaleImage,size, timer, GPUelapsedTime);
    printf("[Resize] Execution time on GPU: %f msec\n", GPUelapsedTime);
    
    //CPU Equalization by myself 
    calcCumHist(cpu_resizedImage,cumHist);
    cpu_equalizedImage = cpu_equalization(cpu_resizedImage,cumHist,&CPUelapsedTime);
    printf("[Equalization] Execution time on CPU: %f msec\n", CPUelapsedTime);


    //Equalization on GPU
    //Mem. allocation on GPU for cumHist
    cudaMalloc((void**)&cumHist_device,256*sizeof(int));
    cudaMemcpy(cumHist_device, cumHist, 256*sizeof(int), cudaMemcpyHostToDevice);


    nThreadPerBlocco = dim3(4, 3);
    numBlocks.y = gpu_resizedImage.rows / nThreadPerBlocco.y + ((gpu_resizedImage.rows % nThreadPerBlocco.y) == 0 ? 0 : 1);
    numBlocks.x = gpu_resizedImage.cols / nThreadPerBlocco.x + ((gpu_resizedImage.cols % nThreadPerBlocco.x) == 0 ? 0 : 1);
    
    cv::cuda::GpuMat gpu_equalizedImage = cv::cuda::createContinuous(600,600,CV_8UC1);
    
    //Timer's start
    cudaEventRecord(timer[0], 0);
    equalizeHistCUDA<<<numBlocks,nThreadPerBlocco>>>(gpu_resizedImage.data,gpu_equalizedImage.data,cumHist_device,gpu_resizedImage.cols,gpu_resizedImage.rows);
    cudaDeviceSynchronize();
    //Timer's end
    cudaEventRecord(timer[1], 0);
    cudaEventSynchronize(timer[1]);
    //Elapsed time calculation
    cudaEventElapsedTime(&GPUelapsedTime, timer[0], timer[1]);
    printf("[Equalization] Execution time on GPU: %f msec\n", GPUelapsedTime);
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess)
        fprintf(stderr, "Errore CUDA: %s\n", cudaGetErrorString(cudaErr));

    cv::Mat img;
    gpu_equalizedImage.download(img);
    cv::imwrite("Equalized by myself.jpg", img);
    

    //The memory of cv::cuda::GpuMat and cv::Mat objects is automatically deallocated by the library.
    //But to avoid any problem I do it manually.
    cpu_grayscaleImage.release();
    cpu_resizedImage.release();
    cpu_equalizedImage.release();
    gpuImage.release();
    gpu_grayscaleImage.release();
    gpu_resizedImage.release();
    gpu_equalizedImage.release();
    cudaFree(cumHist_device);
    cudaEventDestroy(timer[0]);
    cudaEventDestroy(timer[1]);
    return 0;
}


//Cumulative Histogram computation
void calcCumHist(cv::Mat image, int *cumHist){
    int nBins = 256, sum=0;
    int hist[nBins];
    memset(hist,0,sizeof(hist));
    //Histogram
    for (int i = 0; i<image.rows; i++){
        for(int j = 0; j<image.cols; j++){
            unsigned char pixel_value= image.at<unsigned char>(i, j);
            hist[pixel_value]++;
        }
    }

    for (int i = 0; i<nBins; i++){
        sum+=hist[i];
        cumHist[i]=sum;
    }
}


//Histogram equalization on CPU
cv::Mat cpu_equalization(cv::Mat image,int *cumulative_hist, float *elapsedTime){
    struct timespec start_time, end_time;
    cv::Mat equalizedImage(cv::Size(image.rows,image.cols),CV_8UC1,cv::Scalar(255));
    int area = image.rows*image.cols, ngraylevel=256;
    uchar pixel_value;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    //Equalization
    for (int i =0; i<image.rows; i++){
        for(int j = 0; j<image.cols; j++){
            pixel_value = image.at<uchar>(i,j);
            equalizedImage.at<uchar>(i,j) = ((double)ngraylevel/area)*cumulative_hist[pixel_value];
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    *elapsedTime = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
    
    return equalizedImage;
}

//Resize of the image using OpenCV (CPU)
cv::Mat cpu_resizeImage(cv::Mat in,cv::Size size, float *elapsedTime){
    struct timespec start_time, end_time;
    cv::Mat out;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    cv::resize(in, out, size);
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    *elapsedTime = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
    return out;
}

//Converting RGB to Grayscale using OpenCV (CPU)
cv::Mat cpu_RGBtoGRAYSCALE(cv::Mat in, float *elapsedTime){
    struct timespec start_time, end_time;
    //Output image
    cv::Mat out;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    //BGR to Grayscale
    cv::cvtColor(in,out,cv::COLOR_BGR2GRAY);
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    *elapsedTime = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
    return out;
}

//HoughTransform for line
cv::Mat cpu_HoughTransformLine(cv::Mat image, float *elapsedTime){
    struct timespec start_time, end_time;
    cv::Mat output=image.clone();

    std::vector<cv::Vec2f> lines;  // Vector for lines feature

    clock_gettime(CLOCK_MONOTONIC, &start_time);
    cv::HoughLines(image, lines, 1, CV_PI / 180, 100);

    for (size_t i = 0; i < lines.size(); ++i) {
        float rho = lines[i][0];
        float theta = lines[i][1];
        cv::Point pt1, pt2;

        double a = cos(theta);
        double b = sin(theta);
        double x0 = a * rho;
        double y0 = b * rho;

        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));

        cv::line(output, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    *elapsedTime = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
    return output;
}

//Converting RGB to Grayscale using OpenCV for CUDA (GPU)
cv::cuda::GpuMat gpu_RGBtoGRAYSCALE(cv::cuda::GpuMat gpuImage, cudaEvent_t* timer, float& elapsedTime){
    cv::cuda::GpuMat out = cv::cuda::createContinuous(gpuImage.size(),gpuImage.type());
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
cv::cuda::GpuMat gpu_resizeImage(cv::cuda::GpuMat gpuImage, cv::Size outputSize, cudaEvent_t* timer, float& elapsedTime){

    cv::cuda::GpuMat out = cv::cuda::createContinuous(outputSize,gpuImage.type());
    //Timer's start
    cudaEventRecord(timer[0], 0);
    cv::cuda::resize(gpuImage, out, outputSize);
    //Timer's end
    cudaEventRecord(timer[1], 0);
    cudaEventSynchronize(timer[1]);
    //Elapsed time calculation
    cudaEventElapsedTime(&elapsedTime, timer[0], timer[1]);
    return out;
}


__global__ void equalizeHistCUDA(unsigned char* input, unsigned char* output, int *cumulative_hist, int cols, int rows) {
    int nGrayLevels = 256, area = cols*rows;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols){
        int pixelValue = input[i * cols + j];
        output[i * cols + j] = static_cast<uchar>((static_cast<double>(nGrayLevels) / area) * cumulative_hist[pixelValue]);
    }
}