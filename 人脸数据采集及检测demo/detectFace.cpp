//-------------------------detectFace.cpp-----------------------------
//�Ȳɼ��������ݣ������òɼ��õ��������ݣ��������ͷ�е�����


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <fstream>


using namespace cv;
using namespace std;

void collectFace();

int main()
{
	//�ɼ�����
//	collectFace();

	//������������
	string faceFileName = "E:\\project\\facedata\\san\\faces.txt";
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
	//����txt��ȡ����
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
	//ѵ��������
	Ptr<face::LBPHFaceRecognizer> model = face::LBPHFaceRecognizer::create();
	model->train(images, labels);
	model->save("E:\\project\\facedata\\san.xml");

	//����ѵ����ģ������
	const String lbpFile = "E:\\project\\facedata\\san.xml";
	CascadeClassifier faceDetector;
	faceDetector.load(lbpFile);
	if (faceDetector.empty())
	{
		cout << "���ط�����ʧ��" << endl;
		return -1;
	}

	//������ͷ
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cout << "�޷�������ͷ" << endl;
		return -1;
	}
	
	namedWindow("ʶ������", WINDOW_AUTOSIZE);
	
	vector<Rect> faces;
	Mat dst, testSample;
	Mat frame,frameTmp;

	double fps, t = 0;
	while (cap.read(frame))
	{
		t = (double)getTickCount();
		//���ҷ�תÿһ֡ͼ��
		flip(frame, frame, 1);
		frameTmp = frame.clone();
		//���ǰԤ����
		cvtColor(frameTmp, frameTmp, COLOR_BGR2GRAY);
		equalizeHist(frameTmp, frameTmp);  // ֱ��ͼ���⻯ 
		faceDetector.detectMultiScale(frameTmp, faces, 1.15, 4, 0, Size(30, 30));

		for (int i = 0; i < faces.size(); i++)
		{
			Mat roi = frameTmp(faces[i]);
			resize(roi, testSample, Size(96, 112));
			//Ԥ��
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

		//����֡��
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		fps = 1.0 / t;
		putText(frame, format("FPS:%lf", fps), Point(0, cap.get(CV_CAP_PROP_FRAME_HEIGHT)), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

		//esc�˳�
		char c = waitKey(5);
		if (c == 27)
		{
			break;
		}
		imshow("ʶ������", frame);

	}

	waitKey(0);
	return 0;
}

//�ɼ���������
void collectFace()
{
	//��������ͷ
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cout << "�޷�������ͷ" << endl;
		return;
	}
	//��ȡ¼����Ƶ��ز���
	Size S = Size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	double fps, t = 0;

	namedWindow("demo", WINDOW_AUTOSIZE);
	//��ȡ��Ƶÿһ֡����
	Mat frame, dst;
	//��ȡ�������ݵ�����
	Rect faceRect = Rect(200, 100, 192, 224);
	//�ɼ���������
	int faceNum = 0;

	while (cap.read(frame))
	{
		t = (double)getTickCount();
		//���ҷ�תÿһ֡ͼ��
		flip(frame, frame, 1);

		char c = waitKey(5);

		//����space�ɼ�����
		if (c == ' ')
		{
			faceNum++;
			resize(frame(faceRect), dst, Size(96, 112));
			imwrite(format("E:\\project\\facedata\\san\\face_%d.png", faceNum), dst);
			cout << "�ɼ���" << faceNum << "�����������" << endl;
		}
		//��ǲɼ����ݵ�λ��
		rectangle(frame, faceRect, Scalar(0, 0, 255));
		//����֡��
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		fps = 1.0 / t;
		putText(frame, format("FPS:%lf", fps), Point(0, S.height), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

		//esc�˳�
		if (c == 27)
		{
			break;
		}
		imshow("demo", frame);

	}
	//�ͷ������Ƶ
	cap.release();

	waitKey(0);
	destroyAllWindows();
}