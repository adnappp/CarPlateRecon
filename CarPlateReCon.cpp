#include<opencv2\opencv.hpp>  
#include<iostream>  
#include<vector>  
#define WINDOW_NAME  "Plate recognize"  

using namespace std;
using namespace cv;
//-----------------------------------【全局变量声明部分】--------------------------------------  
//      描述：全局变量声明  
//-----------------------------------------------------------------------------------------------  
int blockSize = 5;
int constValue;
Mat gaussianFilImg, sobelImg;
Mat threshImg, closeImg;


//-----------------
Mat oldSrcImage;
Mat srcImage;
Mat grayImage;
Mat blurImage;
Mat hsvImage;
Mat blueImage;
Mat gradImage;
Mat blueGradImage;
Mat morphImage;
Mat openCloseImage;
Mat firstCutImage;
Mat firstCutImage_hsv;
Mat whiteImage;
Mat secondCutImage;
//-----------------
void ColorHistogramEqualization(){
	Mat imageRGB[3];
	split(srcImage, imageRGB);
	for (int i = 0; i < 3; i++){
		equalizeHist(imageRGB[i], imageRGB[i]);
	}
	merge(imageRGB, 3, srcImage);
}
void getWhite(){
	cvtColor(firstCutImage, firstCutImage_hsv, CV_BGR2HSV);
	vector<Mat> hsv_vec;
	Mat img_h, img_s, img_v;
	split(firstCutImage_hsv, hsv_vec);
	img_h = hsv_vec[0];
	img_s = hsv_vec[1];
	img_v = hsv_vec[2];
	img_h.convertTo(img_h, CV_32F);
	img_s.convertTo(img_s, CV_32F);
	img_v.convertTo(img_v, CV_32F);
	normalize(img_h, img_h, 0, 1, NORM_MINMAX);
	normalize(img_s, img_s, 0, 1, NORM_MINMAX);
	normalize(img_v, img_v, 0, 1, NORM_MINMAX);
	whiteImage = ((img_s < 0.16)&(img_v > 0.823));
	imshow("提取白色", whiteImage);
	morphologyEx(whiteImage, openCloseImage, MORPH_CLOSE, Mat::ones(20, 50, CV_8UC1));
	imshow("第二次闭操作后", openCloseImage);
	morphologyEx(openCloseImage, openCloseImage, MORPH_OPEN, Mat::ones(5, 20, CV_8UC1));
	imshow("第二次开操作后", openCloseImage);
}
void getBlue(){
	blueImage = Mat(hsvImage.rows, hsvImage.cols, CV_8U, cv::Scalar(255));
	vector<Mat> hsv_vec;
	Mat img_h, img_s, img_v;
	split(hsvImage, hsv_vec);
	img_h = hsv_vec[0];
	img_s = hsv_vec[1];
	img_v = hsv_vec[2];
	img_h.convertTo(img_h, CV_32F);
	img_s.convertTo(img_s, CV_32F);
	img_v.convertTo(img_v, CV_32F);
	normalize(img_h, img_h, 0, 1, NORM_MINMAX);
	normalize(img_s, img_s, 0, 1, NORM_MINMAX);
	normalize(img_v, img_v, 0, 1, NORM_MINMAX);
	blueImage = ((img_h > 0.51)&(img_h < 0.70)&(img_s > 0.15)&(img_v > 0.25));

	/*double H = 0.0, S = 0.0, V = 0.0;
	for (int i = 0; i < srcImage.rows; i++){
		for (int j = 0; j < srcImage.cols; j++){
			H = hsvImage.at<Vec3b>(i, j)[0];
			S = hsvImage.at<Vec3b>(i, j)[1];
			V = hsvImage.at<Vec3b>(i, j)[2];
			if ((H >= 90 && H <= 120) && (S >= 80 && S <= 220) && (V >=80 && V <= 255)){
				blueImage.at<uchar>(i, j) = 0;
			}
			else
				blueImage.at<uchar>(i, j) = 255;
		}
	}*/
	//blueImage.convertTo(blueImage, CV_HSV2RGB);
	imshow("提取蓝色", blueImage);
}
void openClose(){
	morphologyEx(blueImage, openCloseImage, MORPH_CLOSE, Mat::ones(2, 25, CV_8UC1));
	imshow("第一次闭操作后", openCloseImage);
	morphologyEx(openCloseImage, openCloseImage, MORPH_OPEN, Mat::ones(5, 25, CV_8UC1));
	imshow("第一次开操作后", openCloseImage);

}
void getContours(){
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(openCloseImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(-1, -1));
	Rect rec_adapt;
	int cmin = 100;
	for (size_t i = 0; i < contours.size(); i++){
		if (contours[i].size() > cmin){
		rec_adapt = boundingRect(contours[i]);
		drawContours(morphImage, contours, static_cast<int>(i), Scalar(255, 255, 255), 2);
	}
	}
	firstCutImage = srcImage(rec_adapt);
	imshow("圈出车牌", firstCutImage);
}
Rect getContoursTwice(){
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(openCloseImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(-1, -1));
	Rect rec_adapt;
	Rect fin_rec;
	int cmin = 20;
	int flag = 0;
	for (size_t i = 0; i < contours.size(); i++){
		if (contours[i].size() > cmin){
			rec_adapt = boundingRect(contours[i]);
			float ratio = rec_adapt.width / rec_adapt.height;
			if ( ratio < 5){
				fin_rec = rec_adapt;
				
				flag = 1;
			}
		}
	}
	
	secondCutImage = firstCutImage(fin_rec);
	imshow("第二次圈出车牌", secondCutImage);
	return fin_rec;
}
int main()
{
	srcImage = imread("G:\\picture\\lua22222.jpg", 1);
	if (!srcImage.data) { cout << "error in read image please check it\n"; return false; }
	//CvSize cvsize = cvSize(400, 400);
	//resize(oldSrcImage, srcImage, cvsize, 0, 0, CV_INTER_LINEAR);
	imshow("原图", srcImage);
	cvtColor(srcImage, grayImage, CV_BGR2GRAY);
	//imshow("灰度图", grayImage);
	//medianBlur(grayImage, blurImage, 3);
	//imshow("滤波后图片", blurImage);
	//ColorHistogramEqualization();
	cvtColor(srcImage, hsvImage, CV_BGR2HSV);
	//gradientDetection();
	//imshow("hsv", hsvImage);
	getBlue();
	openClose();
	getContours();//第一次获取轮廓
	getWhite();
	getContoursTwice();//梅林变换
	waitKey();
	return 0;
}

