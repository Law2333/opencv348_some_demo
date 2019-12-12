//--------------------------------------main.cpp----------------------------------------------
//LBPH应用
//图像灰度化--LBP特征提取0~256--ULBP降维(减少计算量)--分割为多个方格(Cell)--每个方格生成直方图
//--直方图链接，特征向量集合--与数据库中直方图比较--得到分类结果


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <fstream>

using namespace cv;
using namespace std;

int main()
{
	string faceFileName = "E:\\project\\facedata\\orl_faces\\faces.txt";
	ifstream faceFile(faceFileName.c_str(), ifstream::in);
	if (!faceFile)
	{
		cout << "读取人脸数据失败" << endl;
		return -1;
	}

	string line, path, classLabel;
	vector<Mat> images;
	vector<int> labels;
	char separator = ';';

	while (getline(faceFile, line))
	{
		stringstream liness(line);
		//以separator(;)为终止符读取一行
		getline(liness, path, separator);
		//读取分号后的label
		getline(liness, classLabel);
		if (!path.empty() && !classLabel.empty())
		{
			//cout << "path:" << path << endl;
			images.push_back(imread(path, IMREAD_GRAYSCALE));
			//atoi string转换为int型
			labels.push_back(atoi(classLabel.c_str()));
		}
	}
	//判断是否读取成功
	if (images.size() < 1 || labels.size() < 1)
	{
		cout << "invaild image path" << endl;
		return -1;
	}
	//训练图像宽高
	int height = images[0].rows;
	int width = images[0].cols;
	cout << "图像宽：" << width << " 图像高：" << height << endl;

	//提取一个测试图像
	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();
	labels.pop_back();

	//其他数据用于训练
	Ptr<face::LBPHFaceRecognizer> model = face::LBPHFaceRecognizer::create();
	model->train(images, labels);

	//预测判断图像
	int predictedLabel = model->predict(testSample);
	//显示预测结果label和实际的对比
	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	cout << result_message << endl;

	//定义相关参数
	int radius = model->getRadius();
	int neibs = model->getNeighbors();
	int grad_x = model->getGridX();
	int grad_y = model->getGridY();
	double t = model->getThreshold();

	//LBP相关特征显示
	cout << "半径距离: " << radius << endl;
	cout << "邻近点数: " << neibs << endl;
	cout << "x梯度: " << grad_x << endl;
	cout << "y梯度: " << grad_y << endl;
	cout << "LBP阈值: " << t << endl;

	waitKey();
	return 0;
}

