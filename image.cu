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
    cv::Mat input = cv::imread("foto.jpg");

    if (input.empty()) {
        fprintf(stderr, "Impossibile caricare l'immagine\n");
        return -1;
    }
    
    cpu_outputGrayScale = cpu_RGBtoGRAYSCALE(input);
    
    cv::imwrite("GrayscaleImageCPU.jpg",cpu_outputGrayScale);
    
    gpu_outputGrayScale = gpu_RGBtoGRAYSCALE(input);
    
    cv::imwrite("GrayscaleImageGPU.jpg",gpu_outputGrayScale);

    /*
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
    return 0;
}

cv::Mat cpu_RGBtoGRAYSCALE(cv::Mat in){
    cv::Mat out;
    cv::cvtColor(in,out,cv::COLOR_BGR2GRAY);
    return out;
}

cv::Mat gpu_RGBtoGRAYSCALE(cv::Mat in){
    cv::Mat out;
    cv::cuda::GpuMat gray;
    cv::cuda::GpuMat gpuImage(in);
    cv::cuda::cvtColor(gpuImage,gray,cv::COLOR_BGR2GRAY);
    gray.download(out);

    return out;
}