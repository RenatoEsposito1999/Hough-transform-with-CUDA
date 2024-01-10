#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui/highgui.hpp>

/*N.B: non posso comparare un metodo di libreria con un metodo eseguito a mano quindi o comparo due metodi di libreria o scrivo sia sequenziale che kernel per
comparare.*/

//Bisognoa provare anche la shared memory perché può essere più veloce se l'istogramma cumulativo è piccolo e entra tutto 
/*
    Prossima cosa da fare: Equalizzazione dell'istogramma su GPU. 
*/

__global__ void calculateHistogram(const uchar* image, int size, int* histogram, int histSize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < size) {
        atomicAdd(&histogram[image[tid]], 1);
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void normalizeAndComputeCDF(float* hist, float* cdf, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0;

    while (tid < size) {
        sum += hist[tid];
        cdf[tid] = sum;
        tid += blockDim.x * gridDim.x;
    }
}


__global__ void equalizeHistCUDA(uchar* data, float* cdf, int size, float scale) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < size) {
        data[tid] = static_cast<uchar>(255.0 * (cdf[data[tid]] / scale));
        tid += blockDim.x * gridDim.x;
    }
}


cv::Mat cpu_RGBtoGRAYSCALE(cv::Mat, float*);
cv::Mat cpu_resizeImage(cv::Mat,cv::Size, float*);
cv::Mat cpu_equalization(cv::Mat, cv::Mat, float*);
cv::Mat calcHist(cv::Mat);
cv::Mat cpu_HoughTransformLine(cv::Mat, float *); //da vedere perché ritorna un output tutto nero. 

cv::cuda::GpuMat gpu_RGBtoGRAYSCALE(cv::cuda::GpuMat, cudaEvent_t*, float&);
cv::cuda::GpuMat gpu_resizeImage(cv::cuda::GpuMat, cv::Size size, cudaEvent_t*, float&);
cv::cuda::GpuMat equalizeHistOnGPU(cv::cuda::GpuMat, cv::Mat);
__global__ void gpu_equalizeImage(uchar*,uchar*, uchar*, int,int);



//cv::Mat metodoHough è l'unico che ritorna l'output finale.


int main(int argn, char *argv[]) {
    //Variables
    cv::Mat cpu_grayscaleImage, cpu_resizedImage, cpu_Hist, cpu_equalizedImage, output;
    cv::cuda::GpuMat gpu_grayscaleImage, gpu_resizedImage, gpu_Hist, gpu_equalizedImage;
    cv::Mat gpu_output; //Final output image (downloaded from GPU)
    cv::Mat cpu_output;
    cv::Mat cumHist;
    dim3 threadsBlock(16,16), numBlocks; 
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

    //Resize on CPU with CPU image as input
    cpu_resizedImage=cpu_resizeImage(cpu_grayscaleImage,size, &CPUelapsedTime);
    printf("[Resize] Execution time on CPU: %f msec\n", CPUelapsedTime);

    //Resize on GPU
    gpu_resizedImage= gpu_resizeImage(gpu_grayscaleImage,size, timer, GPUelapsedTime);
    printf("[Resize] Execution time on GPU: %f msec\n", GPUelapsedTime);

    cumHist = calcHist(cpu_resizedImage);

    //Equalization on CPU
    cpu_equalizedImage = cpu_equalization( cpu_resizedImage , cumHist, &CPUelapsedTime);
    printf("[Equalization] Execution time on CPU: %f msec\n", CPUelapsedTime);

    //cv::imwrite("Input of Equalization.jpg", cpu_resizedImage);
    //cv::imwrite("Output_by_myself.jpg", cpu_equalizedImage);

    cv::cuda::equalizeHist(gpu_resizedImage, gpu_equalizedImage);

    gpu_equalizedImage.download(output);
    cv::imwrite("Rotuine.jpg", output);

    //cv::Mat output2;
    //out.download(output2);
    //cv::imwrite("gpu by myself.jpg", output2);

    //MIA IMPLEMENTAZIONE 
    int histSize = 256;
    int histDataSize = gpu_resizedImage.rows * gpu_resizedImage.cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (histDataSize + threadsPerBlock - 1) / threadsPerBlock;

    // Allocazione di memoria per l'istogramma sulla GPU
    cv::cuda::GpuMat d_histogram(1, histSize, CV_32S, cv::Scalar(0));

    // Lancia il kernel CUDA per il calcolo dell'istogramma
    calculateHistogram<<<blocksPerGrid, threadsPerBlock>>>(gpu_resizedImage.ptr<uchar>(), histDataSize, d_histogram.ptr<int>(), histSize);

    // Normalizza e calcola la distribuzione cumulativa sulla GPU
    cv::cuda::GpuMat d_hist(d_histogram);
    cv::cuda::GpuMat d_cdf(d_histogram.size(), d_histogram.type());
    normalizeAndComputeCDF<<<1, histSize>>>(d_hist.ptr<float>(), d_cdf.ptr<float>(), histSize);

    // Scarica la distribuzione cumulativa dalla GPU
    cv::Mat cumDist;
    d_cdf.download(cumDist);

    // Copia l'immagine sulla GPU
    cv::cuda::GpuMat d_image(cpu_resizedImage);

    // Lancia il kernel CUDA per l'equalizzazione dell'istogramma
    int threadsxBlock = 256;
    int blocksxGrid = (cpu_resizedImage.rows * cpu_resizedImage.cols + threadsxBlock - 1) / threadsxBlock;
    float scale = cumDist.at<float>(histSize - 1);
    equalizeHistCUDA<<<blocksxGrid, threadsxBlock>>>(d_image.ptr<uchar>(), d_cdf.ptr<float>(), cpu_resizedImage.rows * cpu_resizedImage.cols, scale);

    // Scarica l'immagine equalizzata dalla GPU
    cv::Mat equalized;
    d_image.download(equalized);
    cv::imwrite("CUDA.jpg",)
    
    /*/Calcolo istogramma cumulativo x equalizzazione
    cv::cuda::GpuMat gpu_cumHist(cumHist);
    int threadsPerBlock = 256;
    int blocksPerGrid = (gpu_resizedImage.rows * gpu_resizedImage.cols + threadsPerBlock - 1) / threadsPerBlock;
    equalizeHistCUDA<<<blocksPerGrid, threadsPerBlock>>>(gpu_resizedImage.ptr<uchar>(), gpu_cumHist.ptr<float>(), gpu_resizedImage.rows * gpu_resizedImage.cols);
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess)
        fprintf(stderr, "Errore CUDA: %s\n", cudaGetErrorString(cuda_error));

    // Scarica l'immagine equalizzata dalla GPU
    cv::Mat equalized;
    gpu_resizedImage.download(equalized);
    cv::imwrite("Myself.jpg",equalized);

    //cv::cuda::GpuMat out = equalizeHistOnGPU(gpu_resizedImage, cumHist);

    */
    //FINE MIA IMPLEMENTAZIONE
    
    
    
    //Hough Transform for line on CPU
    //cpu_output=cpu_HoughTransformLine(cpu_equalizedImage,&CPUelapsedTime);
    //printf("[Hough Transform] Execution time on CPU: %f msec\n", CPUelapsedTime);
    //cv::imwrite("output.jpg",cpu_output);

    
    //The memory of cv::cuda::GpuMat and cv::Mat objects is automatically deallocated by the library
    cudaEventDestroy(timer[0]);
    cudaEventDestroy(timer[1]);
    return 0;
}


//Histogram computation - Equalized and Normalized cumulativ hist. 
cv::Mat calcHist(cv::Mat image){
    int histSize = 256;
    float sum = 0;
    //Histogram calculation
    cv::Mat hist = cv::Mat::zeros(1, histSize, CV_32F);
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            int pixel_value = static_cast<int>(image.at<uchar>(i*image.cols+j));
            hist.at<float>(pixel_value)++;
        }
    }

    //Cumulative histogram
    cv::Mat cumulative_hist = cv::Mat::zeros(hist.size(), hist.type());
    for (int i = 1; i < histSize; ++i){
        sum += hist.at<float>(i);
        cumulative_hist.at<float>(i) = sum;
    }    
    //Normalization between 0-1
    //cumulative_hist /= image.total();

    return cumulative_hist;
}

//Histogram equalization on CPU
cv::Mat cpu_equalization(cv::Mat image, cv::Mat cumulative_hist, float *elapsedTime){
    struct timespec start_time, end_time;
    cv::Mat equalizedImage = image.clone();

    clock_gettime(CLOCK_MONOTONIC, &start_time);
    //Equalization
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            int pixel_value = static_cast<int>(image.at<uchar>(i*image.cols+j));
            equalizedImage.at<uchar>(i*image.cols+j) = cv::saturate_cast<uchar>(cumulative_hist.at<float>(pixel_value) * 255.0); //sature_cast is used to guarantee values between 0-255
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

