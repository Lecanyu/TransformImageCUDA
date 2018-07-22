/***
 * main function driver. Test all of CUDA implementations are correct.
 * NOTE: you should preprocess replace background color to 0,0,0, because we only use 0,0,0 as background color in CUDA implementation
 */


#include "TransformImage.h"
#include <ctime>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main()
{
	int bg_r = 0;
	int bg_g = 0;
	int bg_b = 0;

	std::vector<std::string> img_filename = {
		"E:/ImageDebug/botanical32/fragment_0001.png",
		"E:/ImageDebug/botanical32/fragment_0002.png",
		/*"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0003.png",
		"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0004.png",*/
		/*"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0005.png",
		"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0006.png",
		"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0007.png",
		"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0008.png",
		"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0009.png",
		"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0010.png",
		"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0011.png",
		"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0012.png",
		"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0013.png",
		"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0014.png",
		"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0015.png",
		"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0016.png",
		"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0017.png",
		"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0018.png",
		"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0019.png",
		"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0020.png",
		"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0021.png",
		"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0022.png",
		"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0023.png",
		"C:/Users/range/Dropbox/JigsawGame/SquareFragmentsTestingData/mountain/5x5/fragment_0024.png",*/
	};

	double total_time = 0;
	InitCUDA();
	for(int i=0;i<img_filename.size();++i)
	{
		for(int j=i+1;j<img_filename.size();++j)
		{
			auto img1 = cv::imread(img_filename[i]);
			auto img2 = cv::imread(img_filename[j]);

			uint8* img1_array = new uint8[img1.rows*img1.cols * 3];
			uint8* img2_array = new uint8[img2.rows*img2.cols * 3];
			for (int i = 0; i<img1.rows; ++i)
			{
				for (int j = 0; j<img1.cols; ++j)
				{
					cv::Vec3b intensity;
					intensity = img1.at<cv::Vec3b>(i, j);
					if (intensity.val[0] == bg_r && intensity.val[1] == bg_g && intensity.val[2] == bg_b)
					{
						img1_array[i*img1.cols * 3 + j * 3] = 0;
						img1_array[i*img1.cols * 3 + j * 3 + 1] = 0;
						img1_array[i*img1.cols * 3 + j * 3 + 2] = 0;
					}
					else
					{
						img1_array[i*img1.cols * 3 + j * 3] = intensity.val[0];
						img1_array[i*img1.cols * 3 + j * 3 + 1] = intensity.val[1];
						img1_array[i*img1.cols * 3 + j * 3 + 2] = intensity.val[2];
					}
				}
			}
			for (int i = 0; i<img2.rows; ++i)
			{
				for (int j = 0; j<img2.cols; ++j)
				{
					cv::Vec3b intensity;
					intensity = img2.at<cv::Vec3b>(i, j);
					if (intensity.val[0] == bg_r && intensity.val[1] == bg_g && intensity.val[2] == bg_b)
					{
						img2_array[i*img2.cols * 3 + j * 3] = 0;
						img2_array[i*img2.cols * 3 + j * 3 + 1] = 0;
						img2_array[i*img2.cols * 3 + j * 3 + 2] = 0;
					}
					else
					{
						img2_array[i*img2.cols * 3 + j * 3] = intensity.val[0];
						img2_array[i*img2.cols * 3 + j * 3 + 1] = intensity.val[1];
						img2_array[i*img2.cols * 3 + j * 3 + 2] = intensity.val[2];
					}
				}
			}

			double transform_mat[6] = {
				1,0,0,
				0,1,759
			};

			double overlap_ratio = -1;
			int overlap_pixel_num = 0;
			int fusion_img_rows = 0;
			int fusion_img_cols = 0;
			int offset_rows = 0;
			int offset_cols = 0;

			int start_s = clock();
			uint8* fusion_img = TransformImage(img1_array, img1.rows, img1.cols, img2_array, img2.rows, img2.cols, transform_mat, fusion_img_rows, fusion_img_cols, overlap_ratio, offset_rows, offset_cols, overlap_pixel_num);
			int stop_s = clock();
			/*std::cout << "time: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " ms" << std::endl;
			std::cout << "overlap ratio: " << overlap_ratio << "\n";*/
			total_time += (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000;
			Showuint8Array(fusion_img, fusion_img_rows, fusion_img_cols);

			delete[] img1_array;
			delete[] img2_array;
			DeleteFusionImage(fusion_img);
		}
	}
	CloseCUDA();
	
	std::cout << "----------------------------\n";
	std::cout<< "total time: " << total_time << " ms\n";
	
	return 0;
}