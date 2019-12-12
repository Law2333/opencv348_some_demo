//使用PCA提取图像特征
//PCA原理
//样本数据->减去均值->计算协方差矩阵->计算特征值和特征向量
//->根据特征值排序保留前K个主成分特征向量->形成新的数据样本

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

double calcPCAOrientation(vector<Point> &pts, Mat &img);

int main()
{
	Mat src = imread("D:\\opencv348\\opencv\\sources\\samples\\data\\pca_test1.jpg");
	if (src.empty())
	{
		cout << "无法读取图像" << endl;
		return 0;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", src);
	//预处理
	Mat gray, binary;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
//	imshow("binary", binary);
	//确定轮廓
	vector<Vec4i> hireachy;
	vector<vector<Point>> contours;
	findContours(binary, contours, hireachy, RETR_LIST, CV_CHAIN_APPROX_NONE);
	//绘制轮廓
	Mat result = src.clone();
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > 1e5 || area < 100) continue;
		drawContours(result, contours, i, Scalar(0, 0, 255), 2);
		double theta = calcPCAOrientation(contours[i], result);	
		cout << "angle: " << 180 * (theta / CV_PI) << endl;
	}

	imshow("contours res", result);

	waitKey();
	return 0;
}

//计算PCA相关参数
double calcPCAOrientation(vector<Point>& pts, Mat & img)
{
	int size = static_cast<int>(pts.size());
	Mat data_pts = Mat(size, 2, CV_64FC1);
	for (int i = 0; i < size; i++)
	{
		data_pts.at<double>(i, 0) = pts[i].x;
		data_pts.at<double>(i, 1) = pts[i].y;
	}

	//定义PCA，分析PCA
	PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);
	Point cnt = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
		static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
	//标记中心点
	circle(img, cnt, 2, Scalar(0, 255, 0), 2);

	vector<Point2d> eigen_vectors(2);
	vector<double> eigen_vals(2);
	//获取特征向量和特征值
	for (int i = 0; i < 2; i++)
	{
		eigen_vectors[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
			pca_analysis.eigenvectors.at<double>(i, 1));
		eigen_vals[i] = pca_analysis.eigenvalues.at<double>(i, 0);
	}
	//特征向量方向
	Point p1 = cnt + 0.02*Point(static_cast<int>(eigen_vectors[0].x*eigen_vals[0]),
		static_cast<int>(eigen_vectors[0].y*eigen_vals[0]));
	Point p2 = cnt + 0.05*Point(static_cast<int>(eigen_vectors[1].x*eigen_vals[1]),
		static_cast<int>(eigen_vectors[1].y*eigen_vals[1]));
	line(img, cnt, p1, Scalar(255, 0, 0), 2);
	line(img, cnt, p2, Scalar(255, 255, 0), 2);

	//计算角度
	double angle = atan2(eigen_vectors[0].y, eigen_vectors[0].x);
	return angle;
}
