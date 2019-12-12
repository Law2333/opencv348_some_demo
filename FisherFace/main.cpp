//------------------------------------main.cpp
//PCA找共同，LDA找差异
//LDA原理---统计学方法
//最大类间方差，最小类内方差
//输入数据---减去均值---计算离散矩阵---计算特征值和特征向量---计算前K个最大特征值---投影到子空间

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
	Ptr<face::FisherFaceRecognizer> model = face::FisherFaceRecognizer::create();
	model->train(images, labels);

	//预测判断图像
	int predictedLabel = model->predict(testSample);
	//显示预测结果label和实际的对比
	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	cout << result_message << endl;

	//特征向量，特征值，均值
	Mat eigenVal = model->getEigenValues();
	Mat eigenVec = model->getEigenVectors();
	Mat means = model->getMean();
	//均值数据重排reshape
	Mat meanFace = means.reshape(1, height);

	Mat dst;
	//归一化，便于显示
	if (meanFace.channels() == 1)
	{
		normalize(meanFace, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	}
	else if (meanFace.channels() == 3)
	{
		normalize(meanFace, dst, 0, 255, NORM_MINMAX, CV_8UC3);
	}
	//显示均值图像
	imshow("mean", dst);

	//显示特征图像
	for (int i = 0; i < min(16, eigenVec.cols); i++)
	{
		// get eigenvector #i
		Mat  ev = eigenVec.col(i).clone();
		Mat grayScale;
		// 还原成原大小 &归一化到[0...255]，方便显示
		Mat eigenFace = ev.reshape(1, height);
		if (eigenFace.channels() == 1)
		{
			normalize(eigenFace, grayScale, 0, 255, NORM_MINMAX, CV_8UC1);
		}
		else if (eigenFace.channels() == 3)
		{
			normalize(eigenFace, grayScale, 0, 255, NORM_MINMAX, CV_8UC3);
		}
		// 添加Jet色彩，便于更好观察
		Mat colorScale;
		applyColorMap(grayScale, colorScale, COLORMAP_HOT);

		imshow(format("eigenface_%d", i), colorScale);
	}

	//重建人脸
	for (int i = 0; i < min(16, eigenVec.cols); i++)
	{
		//从模型中分割特征向量,提取eigenVectors的前i列数据
		Mat evs = eigenVec.col(i);
		//重建第一幅图
		Mat projection = LDA::subspaceProject(evs, means, images[0].reshape(1, 1));
		Mat reconstruction = LDA::subspaceReconstruct(evs, means, projection);

		// 还原成原大小 &归一化到[0...255]，方便显示
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


