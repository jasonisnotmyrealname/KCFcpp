#include "opencv2/core.hpp"
using namespace cv;

class ROISelector {
public:
	Rect select(Mat img, bool fromCenter = true);
	Rect select(const cv::String& windowName, Mat img, bool showCrossair = true, bool fromCenter = true);
	void select(const cv::String& windowName, Mat img, std::vector<Rect> & boundingBox, bool fromCenter = true);

	struct handlerT{
		// basic parameters
		bool isDrawing;
		Rect box;
		Mat image;

		// parameters for drawing from the center
		bool drawFromCenter;
		Point2f center;

		// initializer list
		handlerT() : isDrawing(false), drawFromCenter(true) {};
	}selectorParams;

	// to store the tracked objects
	std::vector<handlerT> objects;

private:
	static void mouseHandler(int event, int x, int y, int flags, void *param);
	void opencv_mouse_callback(int event, int x, int y, int, void *param);

	// save the keypressed characted
	int key;
};

Rect  selectROI(Mat img, bool fromCenter = true);
Rect  selectROI(const cv::String& windowName, Mat img, bool showCrossair = true, bool fromCenter = true);
void  selectROI(const cv::String& windowName, Mat img, std::vector<Rect> & boundingBox, bool fromCenter = true);