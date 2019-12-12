//-------------------------detectFace.cpp-----------------------------
//先采集人脸数据，再利用采集好的人脸数据，检测摄像头中的人脸


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <fstream>


using namespace cv;
using namespace std;

void collectFace();

int main()
{
	//采集人脸
//	collectFace();

	//加载人脸数据
	string faceFileName = "E:\\project\\facedata\\san\\faces.txt";
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
	//根据txt读取数据
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
	//训练并保存
	Ptr<face::LBPHFaceRecognizer> model = face::LBPHFaceRecognizer::create();
	model->train(images, labels);
	model->save("E:\\project\\facedata\\san.xml");

	//加载训练的模型数据
	const String lbpFile = "E:\\project\\facedata\\san.xml";
	CascadeClassifier faceDetector;
	faceDetector.load(lbpFile);
	if (faceDetector.empty())
	{
		cout << "加载分类器失败" << endl;
		return -1;
	}

	//打开摄像头
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cout << "无法打开摄像头" << endl;
		return -1;
	}
	
	namedWindow("识别人脸", WINDOW_AUTOSIZE);
	
	vector<Rect> faces;
	Mat dst, testSample;
	Mat frame,frameTmp;

	double fps, t = 0;
	while (cap.read(frame))
	{
		t = (double)getTickCount();
		//左右翻转每一帧图像
		flip(frame, frame, 1);
		frameTmp = frame.clone();
		//检测前预处理
		cvtColor(frameTmp, frameTmp, COLOR_BGR2GRAY);
		equalizeHist(frameTmp, frameTmp);  // 直方图均衡化 
		faceDetector.detectMultiScale(frameTmp, faces, 1.15, 4, 0, Size(30, 30));

		for (int i = 0; i < faces.size(); i++)
		{
			Mat roi = frameTmp(faces[i]);
			resize(roi, testSample, Size(96, 112));
			//预测
			int label = model->predict(testSample);
			rectangle(frame, faces[i], Scalar(255, 0, 0));
			if (label == 3)
			{
				putText(frame, "FBI", faces[i].tl(), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0));
			}
			else
			{
				putText(frame, "unkown", faces[i].tl(), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0));
			}
		}

		//计算帧数
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		fps = 1.0 / t;
		putText(frame, format("FPS:%lf", fps), Point(0, cap.get(CV_CAP_PROP_FRAME_HEIGHT)), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

		//esc退出
		char c = waitKey(5);
		if (c == 27)
		{
			break;
		}
		imshow("识别人脸", frame);

	}

	waitKey(0);
	return 0;
}

//采集人脸数据
void collectFace()
{
	//启动摄像头
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cout << "无法打开摄像头" << endl;
		return;
	}
	//获取录制视频相关参数
	Size S = Size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	double fps, t = 0;

	namedWindow("demo", WINDOW_AUTOSIZE);
	//读取视频每一帧数据
	Mat frame, dst;
	//获取人脸数据的区域
	Rect faceRect = Rect(200, 100, 192, 224);
	//采集数据数量
	int faceNum = 0;

	while (cap.read(frame))
	{
		t = (double)getTickCount();
		//左右翻转每一帧图像
		flip(frame, frame, 1);

		char c = waitKey(5);

		//按下space采集数据
		if (c == ' ')
		{
			faceNum++;
			resize(frame(faceRect), dst, Size(96, 112));
			imwrite(format("E:\\project\\facedata\\san\\face_%d.png", faceNum), dst);
			cout << "采集第" << faceNum << "个数据已完成" << endl;
		}
		//标记采集数据的位置
		rectangle(frame, faceRect, Scalar(0, 0, 255));
		//计算帧数
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		fps = 1.0 / t;
		putText(frame, format("FPS:%lf", fps), Point(0, S.height), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

		//esc退出
		if (c == 27)
		{
			break;
		}
		imshow("demo", frame);

	}
	//释放相机视频
	cap.release();

	waitKey(0);
	destroyAllWindows();
}