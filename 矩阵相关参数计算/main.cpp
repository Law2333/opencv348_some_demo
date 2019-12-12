//利用opencv相关API计算矩阵的均值、标准差
//计算协方差矩阵、特征值及特征向量


#include <iostream>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc,char** argv)
{
	Mat src = imread("C:\\Users\\FB\\Pictures\\Screenshots\\封面\\数值分析.jpg");
	if (src.empty())
	{
		cout << "无法读取图片" << endl;
		return -1;
	}
	imshow("input", src);

	Mat means, stddev;
	meanStdDev(src, means, stddev);
	cout << "均值行：" << means.rows << " 均值列：" << means.cols << endl;
	cout << "标准方差行：" << stddev.rows << " 标准方差列：" << stddev.cols << endl;

	//输出各通道均值和方差
	for (int row = 0; row < means.rows; row++)
	{
		cout << "mean" << row << " = " << means.at<double>(row) << endl;;
		cout << "stddev" << row << " = " << stddev.at<double>(row) << endl;;
	}
	//直接赋值
	Mat samples = (Mat_<double>(5, 3) <<
		90, 60, 90,
		90, 90, 30,
		60, 60, 60,
		60, 60, 90,
		30, 30, 30);

	Mat cov, mu;
	//按列计算协方差矩阵，一列为一个维度
	calcCovarMatrix(samples, cov, mu, CV_COVAR_NORMAL |CV_COVAR_ROWS);

	cout << "///////////////////////////////////" << endl;
	cout << "cov: " << endl;
	cout << cov << endl;
	cout << "means:" << endl;
	cout << mu << endl;
	cout << "///////////////////////////////////" << endl;

	Mat data = (Mat_<double>(2, 2) << 1, 2, 2, 1);
	Mat eigenVal, eigenVec;
	eigen(data, eigenVal, eigenVec);

	for (int i = 0; i < eigenVal.rows; i++)
	{
		cout << "特征值" << i << ": " << eigenVal.at<double>(i) << endl;
	}
	cout << "特征向量:" << endl;
	cout << eigenVec << endl;

	waitKey();
	return 0;
}
