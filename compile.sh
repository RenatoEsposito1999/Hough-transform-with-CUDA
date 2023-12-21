#!/bin/bash

# Verifica il numero corretto di argomenti
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_file.cu output_executable"
    exit 1
fi

# Assegna gli argomenti a variabili
input_file="$1"
output_executable="$2"

# Verifica che il file di input .cu esista
if [ ! -f "$input_file" ]; then
    echo "Il file $input_file non esiste."
    exit 1
fi

# Esegui il comando nvcc per compilare il file .cu e cattura gli errori
if errors=$(nvcc "$input_file" -o "$output_executable" -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_cudaimgproc -lopencv_cudawarping -lopencv_cudaarithm -lopencv_cudafilters -lopencv_cudafeatures2d -lopencv_cudaobjdetect -lopencv_cudabgsegm -lopencv_cudastereo -lopencv_cudev -lcudart -lnppicom -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lnpps -lcufft -lnvrtc -lnvcuvid -lcudart -lpthread -lm -ldl 2>&1); then
    echo "Compilazione completata con successo. Eseguibile creato: $output_executable"
    ./"$output_executable"
else
    echo "Errore durante la compilazione:"
    echo "$errors"
fi
