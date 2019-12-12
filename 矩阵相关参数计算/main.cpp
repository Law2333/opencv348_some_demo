//����opencv���API�������ľ�ֵ����׼��
//����Э�����������ֵ����������


#include <iostream>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc,char** argv)
{
	Mat src = imread("C:\\Users\\FB\\Pictures\\Screenshots\\����\\��ֵ����.jpg");
	if (src.empty())
	{
		cout << "�޷���ȡͼƬ" << endl;
		return -1;
	}
	imshow("input", src);

	Mat means, stddev;
	meanStdDev(src, means, stddev);
	cout << "��ֵ�У�" << means.rows << " ��ֵ�У�" << means.cols << endl;
	cout << "��׼�����У�" << stddev.rows << " ��׼�����У�" << stddev.cols << endl;

	//�����ͨ����ֵ�ͷ���
	for (int row = 0; row < means.rows; row++)
	{
		cout << "mean" << row << " = " << means.at<double>(row) << endl;;
		cout << "stddev" << row << " = " << stddev.at<double>(row) << endl;;
	}
	//ֱ�Ӹ�ֵ
	Mat samples = (Mat_<double>(5, 3) <<
		90, 60, 90,
		90, 90, 30,
		60, 60, 60,
		60, 60, 90,
		30, 30, 30);

	Mat cov, mu;
	//���м���Э�������һ��Ϊһ��ά��
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
		cout << "����ֵ" << i << ": " << eigenVal.at<double>(i) << endl;
	}
	cout << "��������:" << endl;
	cout << eigenVec << endl;

	waitKey();
	return 0;
}
