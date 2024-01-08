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
//N.B al momento l'operazione da parallelizzare è l'equalizzazione e non il calcolo dell'istogramma eventualmente
//domando alla prof se devo fare pure quello. 
/* Prossima cosa da fare:
    Devo dedicarmi all'istogramma quindi l'obiettivo è [CON METODI OPENCV2]calcolare e visualizzare l'istogramma 
    dell'ultima immagine ottenuta dal preprocessing CPU, equalizzo e visualizzo l'immagine equalizzata e originale
    Dopo di questo devo scrivere il codice CPU io, faccio il passo di prima per capire qual'è l'output dell'operazione.
    Dopodiché fatto con la CPU devo scrivere codice cuda kernel per fare l'operazione di equalizzazione.
*/

cv::Mat cpu_RGBtoGRAYSCALE(cv::Mat, float*);
cv::Mat cpu_resizeImage(cv::Mat,cv::Size, float*);
cv::Mat cpu_calcHist(cv::Mat, float*);

cv::cuda::GpuMat gpu_RGBtoGRAYSCALE(cv::cuda::GpuMat, cudaEvent_t*, float&);
cv::cuda::GpuMat gpu_resizeImage(cv::cuda::GpuMat, cv::Size size, cudaEvent_t*, float&);
cv::cuda::GpuMat gpu_calcHist(cv::cuda::GpuMat ,cudaEvent_t* , float& );

//cv::Mat metodoHough è l'unico che ritorna l'output finale.

int main(int argn, char *argv[]) {
    //Variables
    cv::Mat cpu_grayscaleImage, cpu_resizedImage, cpu_Hist;
    cv::cuda::GpuMat gpu_grayscaleImage, gpu_resizedImage, gpu_Hist;
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

    cpu_Hist = cpu_calcHist(cpu_grayscaleImage, &CPUelapsedTime);
    printf("[Histogram Calculation] Execution time on CPU: %f msec\n", CPUelapsedTime);

    gpu_Hist = gpu_calcHist(gpu_resizedImage, timer, GPUelapsedTime);
    printf("[Histogram Calculation] Execution time on GPU: %f msec\n", GPUelapsedTime);
 
    //cv::imshow("Input image", input);
    //gpu_resizedImage.download(output);
    //cv::imshow("Resized and converted to grayscale image", output);
    //cv::waitKey(0);
    
    //The memory of cv::cuda::GpuMat and cv::Mat objects is automatically deallocated by the library
    cudaEventDestroy(timer[0]);
    cudaEventDestroy(timer[1]);
    return 0;
}


//Histogram calculation by using OpenCV routine, return the hist and not the image of the hist.
cv::Mat cpu_calcHist(cv::Mat image, float *elapsedTime){
    struct timespec start_time, end_time;
    cv::Mat hist, hist_image; //hist_image is the graphical rappresentation of hist.
    int histSize = 256;  // Bin's number
    float range[] = {0, 256};  // Range pixel value
    const float* histRange = {range};
    bool uniform = true;
    bool accumulate = false;
    
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
    //Histogram normalization with values between 0 and 1 with MinMax method, no masks
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    *elapsedTime = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
    return hist;
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

cv::cuda::GpuMat gpu_calcHist(cv::cuda::GpuMat gpuImage,cudaEvent_t* timer, float& elapsedTime){
    cv::cuda::GpuMat hist;
    //Timer's start
    cudaEventRecord(timer[0], 0);
    cv::cuda::calcHist(gpuImage, hist);
    //Timer's end
    cudaEventRecord(timer[1], 0);
    cudaEventSynchronize(timer[1]);
    //Elapsed time calculation
    cudaEventElapsedTime(&elapsedTime, timer[0], timer[1]);
    return hist;
}