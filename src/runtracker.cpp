#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include "kcftracker.hpp"
#include "roiselector.h"



//#include <dirent.h>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]){

	if (argc > 5) return -1;

	bool HOG = false;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = false;
	bool SILENT = true;
	bool LAB = false;
	bool CN = false;
	Rect roi;
	CvFont font;
	double hScale = 0.5;
	double vScale = 0.5;
	int lineWidth = 2;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, hScale, vScale, 0, lineWidth);
	for(int i = 2; i < argc; i++){
		if ( strcmp (argv[i], "hog") == 0 )
			HOG = true;
		if ( strcmp (argv[i], "fixed_window") == 0 )
			FIXEDWINDOW = true;
		if ( strcmp (argv[i], "multi_window") == 0 )
			MULTISCALE = true;
		if ( strcmp (argv[i], "show") == 0 )
			SILENT = false;
		if ( strcmp (argv[i], "lab") == 0 ){
			LAB = true;
			HOG = true;
		}
		if (strcmp(argv[i], "cn") == 0){
			CN = true;
		}
		if ( strcmp (argv[i], "gray") == 0 )
			HOG = false;
	}
	
	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB,CN);

	// Frame readed
	Mat frame;

	// Tracker results
	Rect result;

	// Path to list.txt
	ifstream listFile;
    std::string fileName = "images.txt";
    listFile.open(fileName.c_str());

  	// Read groundtruth for the 1st frame
  	ifstream groundtruthFile;
	string groundtruth = "region.txt";
    groundtruthFile.open(groundtruth.c_str());
  	string firstLine;
  	getline(groundtruthFile, firstLine);
	groundtruthFile.close();
  	
  	istringstream ss(firstLine);

  	// Read groundtruth like a dumb
  	float x1, y1, x2, y2, x3, y3, x4, y4;
  	char ch;
	ss >> x1;
	ss >> ch;
	ss >> y1;
	ss >> ch;
	ss >> x2;
	ss >> ch;
	ss >> y2;
	ss >> ch;
	ss >> x3;
	ss >> ch;
	ss >> y3;
	ss >> ch;
	ss >> x4;
	ss >> ch;
	ss >> y4; 

	// Using min and max of X and Y for groundtruth rectangle
	float xMin =  min(x1, min(x2, min(x3, x4)));
	float yMin =  min(y1, min(y2, min(y3, y4)));
	float width = max(x1, max(x2, max(x3, x4))) - xMin;
	float height = max(y1, max(y2, max(y3, y4))) - yMin;

	
	// Read Images
	/*
	ifstream listFramesFile;
	string listFrames = "images.txt";
	listFramesFile.open(listFrames);
	string frameName;
	*/

	// Write Results
	ofstream resultsFile;
	string resultsPath = "output.txt";
    resultsFile.open(resultsPath.c_str());
	char imfilename[5];

	// Frame counter
	int nFrames = 0;

	VideoCapture capture(argv[1]);

	if (!capture.isOpened())
	{
		cerr << "video read failed" << endl;
//		system("pause");
	}


//	while ( getline(listFramesFile, frameName) ){
	while (capture.read(frame)){
		
//		frameName = frameName;
		capture >> frame;
		// Read each frame from the list
//		frame = imread(frameName, CV_LOAD_IMAGE_COLOR);
		
		// First frame, give the groundtruth to the tracker
		if (nFrames == 0) {
			roi = selectROI("Image", frame);  
			/*
			roi.x = 693;  //zjx debug  Los_Angeles_Car_Chase_05September2014_KABC.avi
			roi.y = 337;
			roi.width = 134;
			roi.height = 72;
			roi.x = 284;  //zjx debug  2-26-2013.avi
			roi.y = 233;
			roi.width = 88;
			roi.height = 64;*/

			tracker.init(roi, frame);
			cv::rectangle(frame, Point(roi.x, roi.y), Point(roi.x + roi.width, roi.y + roi.height), Scalar(255, 255, 255), 3, 8);
//			resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;
		}
		// Update
		else{
			double ext_time = static_cast<double>(getTickCount());
			result = tracker.update(frame);
			ext_time = ((double)getTickCount() - ext_time) / getTickFrequency();
			cout << "fps: " << 1 / ext_time << endl;
			cv::rectangle(frame, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), Scalar(255, 255, 255), 3, 8);   //»­³ö¾ØÐÎ¿ò
//			resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
		}
		
		nFrames++;

//		_itoa(nFrames, imfilename, 10);
        sprintf(imfilename,"%10d",nFrames);
		strcat(imfilename, ".jpg");
		if (!SILENT){
			if (!tracker.d_valid)  
			{
				putText(frame, "lost", Point(result.x, result.y), 1, 1, Scalar(255, 255, 0));
			}	
			imwrite(imfilename, frame);
		}
		if (nFrames == 720)
		{
//			imwrite(imfilename, frame);
		}
		imshow("Image", frame);
		waitKey(1);
	}
	resultsFile.close();

	listFile.close();

}
