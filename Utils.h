#pragma once
#include "cuda_runtime.h"

typedef unsigned char uint8;

struct UV_i {
	int row_i;
	int col_i;
	uint8 r;
	uint8 g;
	uint8 b;
};


/***
 * inv(R|t) = R^T|-R^T*t 
 */
double* InverseMat(double* affine_mat);

/*
 *affine_mat = R|t
 *	affine_mat[0], affine_mat[1], affine_mat[2],
 *	affine_mat[3], affine_mat[4], affine_mat[5],
 */
//UV_i transformUVi(UV_i src_uv, double* affine_mat);



/**
 * show uint8 image array
 */
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
void Showuint8Array(uint8* image_array, int rows, int cols);
void ShowUViArray(UV_i* image_array, int rows, int cols, int offset_row, int offset_col);