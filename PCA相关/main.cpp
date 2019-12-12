//ʹ��PCA��ȡͼ������
//PCAԭ��
//��������->��ȥ��ֵ->����Э�������->��������ֵ����������
//->��������ֵ������ǰK�����ɷ���������->�γ��µ���������

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
		cout << "�޷���ȡͼ��" << endl;
		return 0;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", src);
	//Ԥ����
	Mat gray, binary;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
//	imshow("binary", binary);
	//ȷ������
	vector<Vec4i> hireachy;
	vector<vector<Point>> contours;
	findContours(binary, contours, hireachy, RETR_LIST, CV_CHAIN_APPROX_NONE);
	//��������
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

//����PCA��ز���
double calcPCAOrientation(vector<Point>& pts, Mat & img)
{
	int size = static_cast<int>(pts.size());
	Mat data_pts = Mat(size, 2, CV_64FC1);
	for (int i = 0; i < size; i++)
	{
		data_pts.at<double>(i, 0) = pts[i].x;
		data_pts.at<double>(i, 1) = pts[i].y;
	}

	//����PCA������PCA
	PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);
	Point cnt = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
		static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
	//������ĵ�
	circle(img, cnt, 2, Scalar(0, 255, 0), 2);

	vector<Point2d> eigen_vectors(2);
	vector<double> eigen_vals(2);
	//��ȡ��������������ֵ
	for (int i = 0; i < 2; i++)
	{
		eigen_vectors[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
			pca_analysis.eigenvectors.at<double>(i, 1));
		eigen_vals[i] = pca_analysis.eigenvalues.at<double>(i, 0);
	}
	//������������
	Point p1 = cnt + 0.02*Point(static_cast<int>(eigen_vectors[0].x*eigen_vals[0]),
		static_cast<int>(eigen_vectors[0].y*eigen_vals[0]));
	Point p2 = cnt + 0.05*Point(static_cast<int>(eigen_vectors[1].x*eigen_vals[1]),
		static_cast<int>(eigen_vectors[1].y*eigen_vals[1]));
	line(img, cnt, p1, Scalar(255, 0, 0), 2);
	line(img, cnt, p2, Scalar(255, 255, 0), 2);

	//����Ƕ�
	double angle = atan2(eigen_vectors[0].y, eigen_vectors[0].x);
	return angle;
}
