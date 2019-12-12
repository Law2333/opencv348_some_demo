#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;

const unsigned int width = 500;
const unsigned int height = 500;
const String colorFile = "E:\\project\\caffemodel\\fcn\\pascal-classes.txt";
const String modelFile = "E:\\project\\caffemodel\\fcn\\fcn8s-heavy-pascal.caffemodel";
const String modelTextFile = "E:\\project\\caffemodel\\fcn\\fcn8s-heavy-pascal.prototxt";

void readColors(vector<string> &objName,vector<Vec3b> &objColor);


int main()
{
	Mat src = imread("E:\\project\\pic\\rgb.jpg");
	if (src.empty())
	{
		cout << "could not load pic" << endl;
		system("pause");
		return -1;
	}

	namedWindow("pic", WINDOW_AUTOSIZE);

	while (src.cols > 800 || src.rows > 800)
	{
		resize(src, src, Size(src.cols / 2, src.rows / 2));
	}
	imshow("pic", src);

	//加载分割对象数据
	vector<string> objNames;
	vector<Vec3b> objColors;
	readColors(objNames,objColors);
	dnn::Net net;
	//加载caffe_fcn8s模型
	try
	{
		net = dnn::readNetFromCaffe(modelTextFile, modelFile);
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

	//生成blob
	Mat blobImg = dnn::blobFromImage(src, 1.0f, Size(500, 500));
	//向网络中输入blob
	net.setInput(blobImg, "data");
	//生成检测结果score数据
	//score 21通道 每个通道500*500像素的结果
//21*500*500	查找表	
	Mat score = net.forward("score");

	const int rows = score.size[2];
	const int cols = score.size[3];
	const int channels = score.size[1];


	Mat maxColor(rows, cols, CV_8UC1);		//存储最终选定的通道（21分之一）
	Mat maxVal(rows, cols, CV_32FC1);		//存储label的key

	//建立查找表
	for (int i = 0; i < channels; i++)
	{
		for (int row = 0; row < rows; row++)
		{
			const float *ptrScore = score.ptr<float>(0, i, row);	//指向每个通道第一个位置，存储了21个label的置信度
			uchar *ptrMaxColorCh = maxColor.ptr<uchar>(row);
			float *ptrColorMaxVal = maxVal.ptr<float>(row);
			for (int col = 0; col < cols; col++)
			{
				//确定color最大值
				if (ptrScore[col] > ptrColorMaxVal[col])
				{
					ptrColorMaxVal[col] = ptrScore[col];					//color的结果
					ptrMaxColorCh[col] = i;							//存储最终选择的通道
				}
			}
		}
	}
	
	//填充结果
	Mat result = Mat::zeros(rows, cols, CV_8UC3);
	for (int row = 0; row < rows; row++)
	{
		//指向对应的查找表的行
		const uchar *ptrMaxColor = maxColor.ptr<uchar>(row);
		//结果位置的颜色
		Vec3b *ptrColor = result.ptr<Vec3b>(row);
		for (int col = 0; col < cols;col++)
		{
			//对应位置的颜色
			ptrColor[col] = objColors[ptrMaxColor[col]];
		}
	}

	resize(result, result, Size(src.cols, src.rows));
	imshow("res", result);

	Mat dst;
	addWeighted(src, 0.3, result, 0.7, 0, dst);
	imshow("FCN", dst);

	waitKey();
	return 0;
}

//加载分割对象相关数据
void readColors(vector<string> &objName, vector<Vec3b> &objColor)
{
	ifstream fp(colorFile);
	if (!fp.is_open())
	{
		cout << "无法加载分类数据" << endl;
		system("pause");
		exit(-1);
	}

	string line;
	while (!fp.eof())
	{
		getline(fp, line);
		//加载颜色和对象名称数据
		if (line.length())
		{
			stringstream ss(line);
			string name;
			ss >> name;
			int colorTmp;
			Vec3b color;
			ss >> colorTmp;
			color[0] = colorTmp;
			ss >> colorTmp;
			color[1] = colorTmp;		
			ss >> colorTmp;
			color[2] = colorTmp;
			objName.push_back(name);
			objColor.push_back(color);
		}
	}
}
