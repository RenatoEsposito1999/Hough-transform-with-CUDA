#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>


int main() {
    cv::Mat input = cv::imread("foto.jpg", cv::IMREAD_ANYCOLOR);

    if (input.empty()) {
        fprintf(stderr, "Impossibile caricare l'immagine\n");
        return -1;
        }
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

    return 0;
}