#!/bin/bash

# Check the correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_file.cu output_executable"
    exit 1
fi

# Assign arguments to variables
input_file="$1"
output_executable="$2"

# Verify that the .cu input file exists
if [ ! -f "$input_file" ]; then
    echo "The file $input_file doesn't exist."
    exit 1
fi

# Run the nvcc command to compile the .cu file and catch errors
if errors=$(nvcc "$input_file" -o "$output_executable" -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_cudaimgproc -lopencv_cudawarping -lopencv_cudaarithm -lopencv_cudafilters -lopencv_cudafeatures2d -lopencv_cudaobjdetect -lopencv_cudabgsegm -lopencv_cudastereo -lopencv_cudev -lcudart -lnppicom -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lnpps -lcufft -lnvrtc -lnvcuvid -lcudart -lpthread -lm -ldl 2>&1); then
    echo "Compilation completed successfully. Executable created: $output_executable"
    echo "Running..."
    ./"$output_executable"
else
    echo "Error during compilation:"
    echo "$errors"
fi
