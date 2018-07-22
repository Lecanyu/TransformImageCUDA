# TransformImageCUDA
CUDA implementation for transform dst image to src image. It will stitch two images and detect intersection.

## Dependencies
The core functions only depend on CUDA.
But if you want to compile and run the example (main.cpp), you need to install opencv.

## Export to standalone module
The core function is a low-level implementation. Therefore, you can easily export this program to a standalone module for other program loading. 

For example, you can export this program to a dll module and then load into your python programs.

## Bugs report
If you find any bugs, feel free to open an issue.
