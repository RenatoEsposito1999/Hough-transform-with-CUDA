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
//Inserire anche l'equalizazzione usando le chiamate della libreria cuda e di opencv
//Capire se la routine openCV x CUDA usa la SM.
//Scrivere i tempi su una tabella magari grafica.

void calcCumHist(cv::Mat, int*);
void CalcCudaGrid(dim3&, dim3&, int, int);

cv::Mat cpu_RGBtoGRAYSCALE(cv::Mat, float*);
cv::Mat cpu_resizeImage(cv::Mat,cv::Size, float*);
cv::Mat cpu_equalization(cv::Mat, int*, float*);
cv::Mat cpu_HoughTransformLine(cv::Mat, float *); 

cv::cuda::GpuMat gpu_RGBtoGRAYSCALE(cv::cuda::GpuMat, cudaEvent_t*, float&);
cv::cuda::GpuMat gpu_resizeImage(cv::cuda::GpuMat, cv::Size size, cudaEvent_t*, float&);
cv::cuda::GpuMat equalizeHistOnGPU(cv::cuda::GpuMat, cv::Mat);
__global__ void equalizeHistCUDA(unsigned char*, unsigned char*,int* , int, int);
__global__ void equalizeHistCUDASM(unsigned char*, unsigned char*, int *, int , int ) ;
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

    //Kernel settings
    CalcCudaGrid(numBlocks,nThreadPerBlocco, size.height,size.width);
    printf("\t***Kernel settings***:\nNumber of blocks: %dx%d\tNumber of threads x bloc: %dx%d\n",numBlocks.y,numBlocks.x,nThreadPerBlocco.y, nThreadPerBlocco.x);

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

//Equalization on GPU - NO SM

    //Mem. allocation on GPU for cumHist
    cudaMalloc((void**)&cumHist_device,256*sizeof(int));
    cudaMemcpy(cumHist_device, cumHist, 256*sizeof(int), cudaMemcpyHostToDevice);

    cv::cuda::GpuMat gpu_equalizedImage = cv::cuda::createContinuous(gpu_resizedImage.rows,gpu_resizedImage.cols,CV_8UC1);
    
    //Timer's start
    cudaEventRecord(timer[0], 0);

    equalizeHistCUDA<<<numBlocks,nThreadPerBlocco>>>(gpu_resizedImage.data,gpu_equalizedImage.data,cumHist_device,gpu_resizedImage.cols,gpu_resizedImage.rows);
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess)
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaErr));


    cudaDeviceSynchronize();
    //Timer's end
    cudaEventRecord(timer[1], 0);
    cudaEventSynchronize(timer[1]);
    //Elapsed time calculation
    cudaEventElapsedTime(&GPUelapsedTime, timer[0], timer[1]);
    printf("[Equalization without SM] Execution time on GPU: %f msec\n", GPUelapsedTime);
    
    cv::Mat img;
    gpu_equalizedImage.download(img);
    cv::imwrite("EqualizedWithoutSM.jpg", img);
    
// END Equalization - NO SM

//Start Equalization with SM
    
    cv::cuda::GpuMat gpu_equalizedImageSM = cv::cuda::createContinuous(gpu_resizedImage.rows,gpu_resizedImage.cols,CV_8UC1);
    
    //Timer's start
    cudaEventRecord(timer[0], 0);
    equalizeHistCUDASM<<<numBlocks,nThreadPerBlocco>>>(gpu_resizedImage.data,gpu_equalizedImageSM.data,cumHist_device,gpu_resizedImage.cols,gpu_resizedImage.rows);
    cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess)
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaErr));
    cudaThreadSynchronize();
    //Timer's end
    cudaEventRecord(timer[1], 0);
    cudaEventSynchronize(timer[1]);
    //Elapsed time calculation
    cudaEventElapsedTime(&GPUelapsedTime, timer[0], timer[1]);
    printf("[Equalization with SM] Execution time on GPU: %f msec\n", GPUelapsedTime);

    cv::Mat SM;
    gpu_equalizedImageSM.download(SM);
    cv::imwrite("EqualizedWithSM.jpg", SM);
    
//END Equalization with SM


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

void CalcCudaGrid(dim3 &numBlocks, dim3 &nThreadPerBlocco, int rows, int cols){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // 0 device's index
    //Max thread's num. x block of the gpu
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    nThreadPerBlocco.x = min(cols, int(sqrt(maxThreadsPerBlock)));  // Max for x
    nThreadPerBlocco.y = min(rows, maxThreadsPerBlock / nThreadPerBlocco.x);  // Max for y
    numBlocks.x = (cols + nThreadPerBlocco.x - 1) / nThreadPerBlocco.x;
    numBlocks.y = (rows + nThreadPerBlocco.y - 1) / nThreadPerBlocco.y;
}

__global__ void equalizeHistCUDA(unsigned char* input, unsigned char* output, int *cumulative_hist, int cols, int rows) {
    int nGrayLevels = 256, area = cols*rows;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols){
        int index = i * cols + j;
        int pixelValue = input[index];
        output[index] = static_cast<uchar>((static_cast<double>(nGrayLevels) / area) * cumulative_hist[pixelValue]);
    }
}

__global__ void equalizeHistCUDASM(unsigned char* input, unsigned char* output, int *cumulative_hist, int cols, int rows) {
    int nGrayLevels = 256, area = cols * rows;
    __shared__ int shared_cumulative_hist[256];
    int elements_per_thread = ( 256/(blockDim.x*blockDim.y) > 1 ) ? (256/blockDim.x*blockDim.y) : 1;
    int InBlockThreadID = threadIdx.x + blockDim.x * threadIdx.y; //from 0 to 1023 x block of 32x32 threads
    int start_index = InBlockThreadID * elements_per_thread;
    for (int i = 0; i < elements_per_thread; i++) {
        int index = start_index + i;
        if (index < 256)
            shared_cumulative_hist[index] = cumulative_hist[index];
    }
    __syncthreads();

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {
        int index = i * cols + j;
        int pixelValue = input[index];
        output[index] = static_cast<unsigned char>((static_cast<double>(nGrayLevels) / area) * shared_cumulative_hist[pixelValue]);
    }
}

//SOLUZIONe:
// Un solo thread per blocco carica i dati nella shared memory
/*for (int i = threadIdx.x; i < 256; i += blockDim.x) {
shared_cumulative_hist[i] = cumulative_hist[i];
}
Oppure
//int elements_per_thread = 256 / (blockDim.x * blockDim.y);
    //int start_index = (blockIdx.x * blockDim.x + threadIdx.x) * elements_per_thread;
*/

