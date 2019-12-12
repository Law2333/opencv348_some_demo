//------------------------------------main.cpp
//PCA�ҹ�ͬ��LDA�Ҳ���
//LDAԭ��---ͳ��ѧ����
//�����䷽���С���ڷ���
//��������---��ȥ��ֵ---������ɢ����---��������ֵ����������---����ǰK���������ֵ---ͶӰ���ӿռ�

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
	Ptr<face::FisherFaceRecognizer> model = face::FisherFaceRecognizer::create();
	model->train(images, labels);

	//Ԥ���ж�ͼ��
	int predictedLabel = model->predict(testSample);
	//��ʾԤ����label��ʵ�ʵĶԱ�
	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	cout << result_message << endl;

	//��������������ֵ����ֵ
	Mat eigenVal = model->getEigenValues();
	Mat eigenVec = model->getEigenVectors();
	Mat means = model->getMean();
	//��ֵ��������reshape
	Mat meanFace = means.reshape(1, height);

	Mat dst;
	//��һ����������ʾ
	if (meanFace.channels() == 1)
	{
		normalize(meanFace, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	}
	else if (meanFace.channels() == 3)
	{
		normalize(meanFace, dst, 0, 255, NORM_MINMAX, CV_8UC3);
	}
	//��ʾ��ֵͼ��
	imshow("mean", dst);

	//��ʾ����ͼ��
	for (int i = 0; i < min(16, eigenVec.cols); i++)
	{
		// get eigenvector #i
		Mat  ev = eigenVec.col(i).clone();
		Mat grayScale;
		// ��ԭ��ԭ��С &��һ����[0...255]��������ʾ
		Mat eigenFace = ev.reshape(1, height);
		if (eigenFace.channels() == 1)
		{
			normalize(eigenFace, grayScale, 0, 255, NORM_MINMAX, CV_8UC1);
		}
		else if (eigenFace.channels() == 3)
		{
			normalize(eigenFace, grayScale, 0, 255, NORM_MINMAX, CV_8UC3);
		}
		// ���Jetɫ�ʣ����ڸ��ù۲�
		Mat colorScale;
		applyColorMap(grayScale, colorScale, COLORMAP_HOT);

		imshow(format("eigenface_%d", i), colorScale);
	}

	//�ؽ�����
	for (int i = 0; i < min(16, eigenVec.cols); i++)
	{
		//��ģ���зָ���������,��ȡeigenVectors��ǰi������
		Mat evs = eigenVec.col(i);
		//�ؽ���һ��ͼ
		Mat projection = LDA::subspaceProject(evs, means, images[0].reshape(1, 1));
		Mat reconstruction = LDA::subspaceReconstruct(evs, means, projection);

		// ��ԭ��ԭ��С &��һ����[0...255]��������ʾ
		Mat result = reconstruction.reshape(1, height);
		if (result.channels() == 1)
		{
			normalize(result, reconstruction, 0, 255, NORM_MINMAX, CV_8UC1);
		}
		else if (result.channels() == 3)
		{
			normalize(result, reconstruction, 0, 255, NORM_MINMAX, CV_8UC3);
		}
		imshow(format("reconstruction_face_%d", i), reconstruction);
	}

	waitKey();
	return 0;
}


