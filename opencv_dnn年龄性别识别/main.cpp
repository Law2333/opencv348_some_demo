#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//������·��
//const string harr_file = "D:\\opencv348\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt_tree.xml";
//����·��
const string age_model = "E:\\project\\caffemodel\\age_gender\\age_net.caffemodel";
const string age_txt = "E:\\project\\caffemodel\\age_gender\\deploy_age.prototxt";
const string gender_model = "E:\\project\\caffemodel\\age_gender\\gender_net.caffemodel";
const string gender_txt = "E:\\project\\caffemodel\\age_gender\\deploy_gender.prototxt";
//��������·��
const string face_model = "E:\\project\\caffemodel\\MobileNet\\MobileNetSSD_deploy.caffemodel";
const string face_model_txt = "E:\\project\\caffemodel\\MobileNet\\MobileNetSSD_deploy.prototxt";

void predictFace(dnn::Net &net, Mat &img,vector<Rect> &faces);
void predictAge(dnn::Net &net, Mat &img);
void predictGender(dnn::Net &net, Mat &img);
vector<string> ageLabels();


int main()
{
	Mat src = imread("E:\\project\\pic\\face.jpg");
	if (src.empty())
	{
		cout << "�޷���ȡͼƬ" << endl;
		system("pause");
		return -1;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", src);

	//��������
	dnn::Net ageNet = dnn::readNetFromCaffe(age_txt, age_model);
	dnn::Net genderNet = dnn::readNetFromCaffe(gender_txt, gender_model);
	dnn::Net faceNet = dnn::readNetFromCaffe(face_model_txt, face_model);

	//�������������
//	CascadeClassifier faceDetector;
//	faceDetector.load(harr_file);
	vector<Rect> faces;	
	//�Ҷ�ת��
//	Mat grayImg;
//	cvtColor(src, grayImg, COLOR_BGR2GRAY);
	//�������
//	faceDetector.detectMultiScale(grayImg, faces, 1.02, 1, 0);
	predictFace(faceNet, src, faces);

	for (unsigned int i = 0; i < faces.size(); i++)
	{
		rectangle(src, faces[i], Scalar(201, 255, 0));
		predictAge(ageNet, src(faces[i]));
		predictGender(genderNet, src(faces[i]));
	}

	imshow("res", src);

	waitKey();
	return 0;
}

//��ʼ��ageLabels
vector<string> ageLabels()
{
	vector<string> ages;
	ages.push_back("0-2");
	ages.push_back("4-6");
	ages.push_back("8-13");
	ages.push_back("15-20");
	ages.push_back("25-32");
	ages.push_back("38-43");
	ages.push_back("48-53");
	ages.push_back("60-");
	return ages;
}

//�������
void predictFace(dnn::Net & net, Mat & img, vector<Rect> &faces)
{
	//ѵ��ģ����ز���
	const unsigned int width = 300;
	const unsigned int height = 300;
	const float meanValue = 127.5;
	const float scaleFactor = 0.007843f;

	Mat inputBlob, detection, detectionMat;
	//ȷ��blob
	inputBlob = dnn::blobFromImage(img, scaleFactor, Size(width, height), meanValue);
	//����blob
	net.setInput(inputBlob, "data");

	detection = net.forward("detection_out");
	//7 x 10 0,Ŀ�������±꣬���Ÿ��ʣ�Ŀ����ͼ���ϵı��ʣ�������������㣩
	detectionMat = Mat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	float confidenceThreshold = 0.5;
	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);
		//ʶ�����Ÿ��ʴ�����ֵ
		if (confidence > confidenceThreshold)
		{
			unsigned int objIndex = (unsigned int)(detectionMat.at<float>(i, 1));
			//������Ϊ����
			if (objIndex == 15)
			{
				float tl_x = detectionMat.at<float>(i, 3) * img.cols;
				float tl_y = detectionMat.at<float>(i, 4) * img.rows;
				float br_x = detectionMat.at<float>(i, 5) * img.cols;
				float br_y = detectionMat.at<float>(i, 6) * img.rows;

				//ȷ������λ��
				Rect objectBox((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y));
				faces.push_back(objectBox);
			}
		}
	}

}

//Ԥ������
void predictAge(dnn::Net & net, Mat & img)
{
	//����blob
	Mat blob = dnn::blobFromImage(img, 1.0, Size(227, 227));
	net.setInput(blob, "data");
	//Ԥ�����
	Mat prob = net.forward("prob");
	Mat probMat = prob.reshape(1, 1);
	//ȷ�����п��ܵ�����
	Point classNum;
	double classProb;
	minMaxLoc(probMat, NULL, &classProb, NULL, &classNum);
	//��ʾ����
	vector<string> ages = ageLabels();
	int classIdx = classNum.x;
	putText(img, format("A:%s", ages.at(classIdx).c_str()), 
		Point(0, 10), FONT_HERSHEY_PLAIN, 0.8, Scalar(0, 0, 255), 1);

}

//Ԥ���Ա�
void predictGender(dnn::Net & net, Mat & img)
{
	//����blob
	Mat blob = dnn::blobFromImage(img, 1.0, Size(227, 227));
	net.setInput(blob, "data");
	//Ԥ�����
	Mat prob = net.forward("prob");
	Mat probMat = prob.reshape(1, 1);

	putText(img, format("G:%s", probMat.at<float>(0,0) > probMat.at<float>(0,1) ? "M":"F"), 
		Point(0, 20), FONT_HERSHEY_PLAIN, 0.8, Scalar(0, 0, 255), 1);
}
