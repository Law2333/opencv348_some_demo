//--------------------------------------main.cpp----------------------------------------------
//LBPHӦ��
//ͼ��ҶȻ�--LBP������ȡ0~256--ULBP��ά(���ټ�����)--�ָ�Ϊ�������(Cell)--ÿ����������ֱ��ͼ
//--ֱ��ͼ���ӣ�������������--�����ݿ���ֱ��ͼ�Ƚ�--�õ�������


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
		cout << "��ȡ��������ʧ��" << endl;
		return -1;
	}

	string line, path, classLabel;
	vector<Mat> images;
	vector<int> labels;
	char separator = ';';

	while (getline(faceFile, line))
	{
		stringstream liness(line);
		//��separator(;)Ϊ��ֹ����ȡһ��
		getline(liness, path, separator);
		//��ȡ�ֺź��label
		getline(liness, classLabel);
		if (!path.empty() && !classLabel.empty())
		{
			//cout << "path:" << path << endl;
			images.push_back(imread(path, IMREAD_GRAYSCALE));
			//atoi stringת��Ϊint��
			labels.push_back(atoi(classLabel.c_str()));
		}
	}
	//�ж��Ƿ��ȡ�ɹ�
	if (images.size() < 1 || labels.size() < 1)
	{
		cout << "invaild image path" << endl;
		return -1;
	}
	//ѵ��ͼ����
	int height = images[0].rows;
	int width = images[0].cols;
	cout << "ͼ���" << width << " ͼ��ߣ�" << height << endl;

	//��ȡһ������ͼ��
	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();
	labels.pop_back();

	//������������ѵ��
	Ptr<face::LBPHFaceRecognizer> model = face::LBPHFaceRecognizer::create();
	model->train(images, labels);

	//Ԥ���ж�ͼ��
	int predictedLabel = model->predict(testSample);
	//��ʾԤ����label��ʵ�ʵĶԱ�
	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	cout << result_message << endl;

	//������ز���
	int radius = model->getRadius();
	int neibs = model->getNeighbors();
	int grad_x = model->getGridX();
	int grad_y = model->getGridY();
	double t = model->getThreshold();

	//LBP���������ʾ
	cout << "�뾶����: " << radius << endl;
	cout << "�ڽ�����: " << neibs << endl;
	cout << "x�ݶ�: " << grad_x << endl;
	cout << "y�ݶ�: " << grad_y << endl;
	cout << "LBP��ֵ: " << t << endl;

	waitKey();
	return 0;
}

