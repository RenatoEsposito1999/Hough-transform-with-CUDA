#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>

__global__ void equalizeHistKernel(unsigned char* input, unsigned char* output, int width, int height, float* cdf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //printf("cdf[0]: %f\n", cdf[0]);
    if (i < height && j < width) {
        int idx = i * width + j;
        // Calcola il valore normalizzato
        //float normalizedValue = (cdf[input[idx]] - cdf[0]) / (width * height - 1);
        // Scala il valore normalizzato nel range 0-255 e assegna all'output
        //output[idx] = static_cast<unsigned char>(255.0 * normalizedValue);
        //printf("[INIZIO] cdf[input[idx]] = %u\ncdf[0] = %f\nwidht: = %d\theight: = %d\nwidth * height - 1 = %f\nNormalizedValue = %f\nRisultato senza funzione = %f\nRisultato con funzione =%f\n[Fine] output = %f\n", (unsigned int)input[idx],cdf[0],width,height, (width * height)-1,normalizedValue, 255.0*normalizedValue,static_cast<unsigned char>(255.0 * normalizedValue),(unsigned char)output[idx]);
        int pixel_value = static_cast<int>(input[i*width+j]);
        output[i*width+j]= static_cast<unsigned char>(cdf[pixel_value]* 255.0);
    }
}

int main(int argc, char** argv) {
    
    // Carica l'immagine in scala di grigi
    cv::Mat originalImage = cv::imread("foto.jpg", cv::IMREAD_GRAYSCALE);
    if (originalImage.empty()) {
        printf("Could not open or find the image\n");
        return -1;
    }

    int width = originalImage.cols;
    int height = originalImage.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    // Converte l'immagine originale in un array di byte
    unsigned char* h_input = originalImage.data;

    // Alloca la memoria per l'immagine di output sul device
    unsigned char* d_output;
    cudaMalloc((void**)&d_output, imageSize);

    // Calcola l'istogramma dell'immagine
    int histSize = 256;
    float hist[histSize];
    memset(hist, 0, histSize * sizeof(float));

    for (int i = 0; i < width * height; ++i) {
        hist[h_input[i]]++;
    }

    // Calcola la funzione di distribuzione cumulativa (CDF)
    float cdf[histSize];
    cdf[0] = hist[0];
    //printf("cfd[0] originale = %f\n", cdf[0]);
    for (int i = 1; i < histSize; ++i) {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    // Normalizza la CDF
    for (int i = 0; i < histSize; ++i) {
        cdf[i] /= cdf[histSize - 1];
    }
    //printf("cfd[0] normalizzata = %f\n", cdf[0]);
    
    // Alloca la memoria per la CDF sul device
    float* d_cdf;
    cudaMalloc((void**)&d_cdf, histSize * sizeof(float));
    cudaMemcpy(d_cdf, cdf, histSize * sizeof(float), cudaMemcpyHostToDevice);

    // Converte l'immagine originale in un array di byte sul device
    unsigned char* d_input;
    cudaMalloc((void**)&d_input, imageSize);
    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);

    // Definisci la griglia e il blocco dei thread
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Applica l'equalizzazione dell'istogramma utilizzando il kernel CUDA
    equalizeHistKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, d_cdf);

    // Copia il risultato dal device alla memoria host
    unsigned char* h_output = new unsigned char[imageSize];
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    // Libera la memoria sul device
    cudaFree(d_output);
    cudaFree(d_cdf);
    cudaFree(d_input);

    // Crea un oggetto cv::Mat per l'immagine di output
    cv::Mat equalizedImage(height, width, CV_8UC1, h_output);

    // Mostra l'immagine originale e quella equalizzata
    cv::imwrite("TTOriginal Image.jpg", originalImage);
    cv::imwrite("TTEqualized Image.jpg", equalizedImage);

    cv::waitKey(0);

    // Libera la memoria sulla CPU
    delete[] h_output;

    return 0;
}
