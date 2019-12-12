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

	//���طָ��������
	vector<string> objNames;
	vector<Vec3b> objColors;
	readColors(objNames,objColors);
	dnn::Net net;
	//����caffe_fcn8sģ��
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

	//����blob
	Mat blobImg = dnn::blobFromImage(src, 1.0f, Size(500, 500));
	//������������blob
	net.setInput(blobImg, "data");
	//���ɼ����score����
	//score 21ͨ�� ÿ��ͨ��500*500���صĽ��
//21*500*500	���ұ�	
	Mat score = net.forward("score");

	const int rows = score.size[2];
	const int cols = score.size[3];
	const int channels = score.size[1];


	Mat maxColor(rows, cols, CV_8UC1);		//�洢����ѡ����ͨ����21��֮һ��
	Mat maxVal(rows, cols, CV_32FC1);		//�洢label��key

	//�������ұ�
	for (int i = 0; i < channels; i++)
	{
		for (int row = 0; row < rows; row++)
		{
			const float *ptrScore = score.ptr<float>(0, i, row);	//ָ��ÿ��ͨ����һ��λ�ã��洢��21��label�����Ŷ�
			uchar *ptrMaxColorCh = maxColor.ptr<uchar>(row);
			float *ptrColorMaxVal = maxVal.ptr<float>(row);
			for (int col = 0; col < cols; col++)
			{
				//ȷ��color���ֵ
				if (ptrScore[col] > ptrColorMaxVal[col])
				{
					ptrColorMaxVal[col] = ptrScore[col];					//color�Ľ��
					ptrMaxColorCh[col] = i;							//�洢����ѡ���ͨ��
				}
			}
		}
	}
	
	//�����
	Mat result = Mat::zeros(rows, cols, CV_8UC3);
	for (int row = 0; row < rows; row++)
	{
		//ָ���Ӧ�Ĳ��ұ����
		const uchar *ptrMaxColor = maxColor.ptr<uchar>(row);
		//���λ�õ���ɫ
		Vec3b *ptrColor = result.ptr<Vec3b>(row);
		for (int col = 0; col < cols;col++)
		{
			//��Ӧλ�õ���ɫ
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

//���طָ�����������
void readColors(vector<string> &objName, vector<Vec3b> &objColor)
{
	ifstream fp(colorFile);
	if (!fp.is_open())
	{
		cout << "�޷����ط�������" << endl;
		system("pause");
		exit(-1);
	}

	string line;
	while (!fp.eof())
	{
		getline(fp, line);
		//������ɫ�Ͷ�����������
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
