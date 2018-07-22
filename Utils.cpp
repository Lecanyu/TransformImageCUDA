#include "Utils.h"


/***
* inv(R|t) = R^T | -R^T*t
*/
double* InverseMat(double* affine_mat)
{
	double* affine_inverse = new double[6];
	affine_inverse[0] = affine_mat[0];
	affine_inverse[1] = affine_mat[3];
	affine_inverse[2] = -(affine_mat[0] * affine_mat[2] + affine_mat[3] * affine_mat[5]);
	affine_inverse[3] = affine_mat[1];
	affine_inverse[4] = affine_mat[4];
	affine_inverse[5] = -(affine_mat[1] * affine_mat[2] + affine_mat[4] * affine_mat[5]);

	return affine_inverse;
}


//UV_i transformUVi(UV_i src_uv, double* affine_mat)
//{
//	UV_i new_uv;
//	new_uv.row_i = (int)(affine_mat[0] * (double)src_uv.row_i + affine_mat[1] * (double)src_uv.col_i + affine_mat[2]);
//	new_uv.col_i = (int)(affine_mat[3] * (double)src_uv.row_i + affine_mat[4] * (double)src_uv.col_i + affine_mat[5]);
//	new_uv.b = src_uv.b;
//	new_uv.g = src_uv.g;
//	new_uv.r = src_uv.r;
//
//	return new_uv;
//}



void Showuint8Array(uint8* image_array, int rows, int cols)
{
	cv::Mat img(rows, cols, CV_8UC3);
	for (int i = 0; i<rows; ++i)
	{
		for (int j = 0; j<cols; ++j)
		{
			cv::Vec3b intensity;
			intensity.val[0] = image_array[i*cols * 3 + j * 3];
			intensity.val[1] = image_array[i*cols * 3 + j * 3 + 1];
			intensity.val[2] = image_array[i*cols * 3 + j * 3 + 2];
			img.at<cv::Vec3b>(i, j) = intensity;
		}
	}

	cv::imshow("1", img);
	cv::waitKey();

}

void ShowUViArray(UV_i* image_array, int rows, int cols, int offset_row, int offset_col)
{
	cv::Mat img(rows, cols, CV_8UC3);
	for (int i = 0; i<rows; ++i)
	{
		for (int j = 0; j<cols; ++j)
		{
			cv::Vec3b intensity;
			intensity.val[0] = image_array[i*cols + j].r;
			intensity.val[1] = image_array[i*cols + j].g;
			intensity.val[2] = image_array[i*cols + j].b;
			int ro = image_array[i*cols + j].row_i;
			int co = image_array[i*cols + j].col_i;
			img.at<cv::Vec3b>(ro - offset_row, co-offset_col) = intensity;
		}
	}

	cv::imshow("1", img);
	cv::waitKey();
}
