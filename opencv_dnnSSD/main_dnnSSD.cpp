#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;

const unsigned int width = 300;
const unsigned int height = 300;
const String labelFile = ":/SSD_300x300/labelmap_ilsvrc_det.prototxt";
const String modelFile = ":/SSD_300x300/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel";
const String modelTextFile = ":/SSD_300x300/deploy.prototxt";
//model meanVal 模型
//模型mean值
const int meanValues[3] = { 104,117,123 };

vector<String> readLabels();
//static Mat getMean(const unsigned int &w, const unsigned int &h);
//static Mat preprocess(const Mat &frame);

int main()
{
	Mat src = imread(":/pic/messi5.jpg");
	if (src.empty())
	{
		cout << "could not load pic" << endl;
		system("pause");
		return -1;
	}

	namedWindow("pic", WINDOW_AUTOSIZE);
	
	while (src.cols > 800 || src.rows > 800)
	{
		resize(src, src, Size(src.cols/2, src.rows/2));
	}
	imshow("pic", src);

	//读取label数据
	vector<String> objNames = readLabels();
	dnn::Net net;
	//加载caffe_ssd模型
	try
	{
		 net = dnn::readNetFromCaffe(modelTextFile,modelFile);
	}	
	catch (cv::Exception  &ee)
	{
		cout << ee.what() << endl;
		if (net.empty())
		{
			cout << "could not load ssd model" << endl;
			system("pause");
			return -1;
		}
	}

	//图像预处理
//	Mat inputImg = preprocess(src);
	//生成blob
	Mat blobImg = dnn::blobFromImage(src,1.0f, Size(300, 300), Scalar(104, 117, 123), false, false);
	//将blob输入网络
	net.setInput(blobImg, "data");
	//检测分类
	Mat detection = net.forward("detection_out");
	//Mat(int rows, int cols, int type, const Scalar& s);
	//7 x 10 0,检测结果索引,置信概率,目标位置比率(tl_x,tl_y,br_x,br_y)
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	float confidenceThreshold = 0.16;
	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);
		//确定检测分类结果
		if (confidence > confidenceThreshold)
		{
			unsigned int objIndex = (unsigned int)(detectionMat.at<float>(i, 1));
			float tl_x = detectionMat.at<float>(i, 3) * src.cols;
			float tl_y = detectionMat.at<float>(i, 4) * src.rows;
			float br_x = detectionMat.at<float>(i, 5) * src.cols;
			float br_y = detectionMat.at<float>(i, 6) * src.rows;

			//标记分类结果
			Rect objectBox((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y));
			rectangle(src, objectBox, Scalar(0, 255, 0), 2, 8, 0);
			putText(src, format("%s:%f", objNames[objIndex].c_str(), detectionMat.at<float>(i, 2)),
					Point(tl_x, tl_y), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 255), 1);
		}
	}

	imshow("detection", src);

	waitKey();
	return 0;
}

//加载标签数据
vector<String> readLabels()
{
	vector<String> objName;
	ifstream fp(labelFile);
	if (!fp.is_open())
	{
		cout << "could not load label file" << endl;
		system("pause");
		exit(-1);
	}

	string labelName;
	while (!fp.eof())
	{
		getline(fp, labelName);
		//根据标签文件加载数据
		if (labelName.length() && labelName.find("  display_name:") == 0)
		{
			string temp = labelName.substr(17);
			temp.replace(temp.end() - 1, temp.end(), "");
			objName.push_back(temp);
		}
	}

	return objName;
}


//Mat getMean(const unsigned int & w, const unsigned int & h)
//{
//	Mat mean;
//	vector<Mat> channels;
//	for (int i = 0; i < 3;i++)
//	{
//		Mat channel(h, w, CV_32F, Scalar(meanValues[i]));
//		channels.push_back(channel);
//	}
//
//	merge(channels, mean);
//	return mean;
//}
//
//
//Mat preprocess(const Mat & frame)
//{
//	Mat preprocessd;
//	frame.convertTo(preprocessd, CV_32F);
//	resize(preprocessd, preprocessd, Size(width, height));
//
//	Mat mean = getMean(width, height);
//
//	subtract(preprocessd, mean, preprocessd);
//	return preprocessd;
//}
