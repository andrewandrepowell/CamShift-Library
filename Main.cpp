/** \author Andrew Powell \date 6/15/2014 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ios>
#include <limits>
#include <exception>
#include "CamShift.h"

using namespace camShift;
using namespace std;

void mouseFunction(int event, int x, int y, int, void* parameter) {
	static cv::Point startPoint;
	CamShift& camShift = *((CamShift*)((void**)parameter)[0]);
	bool& selectionHasBeenSet = *((bool*)((void**)parameter)[1]);

	switch (event) {
	case cv::EVENT_LBUTTONDOWN: 
		startPoint = cv::Point(x, y);
		break;
	case cv::EVENT_LBUTTONUP: 
		camShift.setSelection(cv::Rect(startPoint, cv::Point(x, y)));		
		selectionHasBeenSet = true;
		break;
	default: break;
	}
}

int main(int argc, char* argv[]) {

	/*
	 * Instructions:
	 *
	 * In this example, the CamShift class is demonstrated. On the example's start, two windows should open in
	 * addition to the console. If a camera is properly connected, one of the two windows should start to 
	 * display a live video feed and the other window should be gray. 
	 *
	 * The user can then click and drag an invisble rectangle around an area within the window displaying the
	 * live video feed. The area should contain a portion of the desired object to track. Ideally, the area 
	 * should mostly consist of a single color and the color should differ greatly from the colors shown in 
	 * the rest of the live video feed. 
	 *
	 * Once the user makes their selection, the CAMShift algorithm will begin to repeatedly execute and thus a
	 * red ellipse should appear over the desired object. As the object moves, the red ellipse should continue
	 * remain on the object. The gray window should also begin to display the backprojections.
	 */

	try {
		/*-- Declarations --*/

		CamShift camShift;
		bool selectionHasBeenSet = false;
		void* sharedPointers[] = { &camShift, &selectionHasBeenSet };

		char* windowName = "Example Window";
		char* backWindowName = "Backprojection";
		cv::namedWindow(windowName, 0);
		cv::setMouseCallback(windowName, mouseFunction, sharedPointers);
		cv::namedWindow(backWindowName, 0);

		cv::VideoCapture videoCapture(0);
		cv::Mat capturedRawFrame;

		/*-- Main loop --*/
		while (true) {

			/* First, capture raw frame from camera */
			videoCapture >> capturedRawFrame;

			/* Set the captured raw frame of the CamShift object */
			camShift.setCapturedRawFrame(capturedRawFrame);

			/* Execute the following instructions once the user selects the desired object to track */
			if (selectionHasBeenSet) {

				/* Execute the CAMShift algorithm */
				camShift.runCamShift();

				/* Draw an ellipse on the captured raw frame, hopefully indicating where the tracked 
				   object is located in the frame */
				cv::ellipse(capturedRawFrame, camShift.getRotatedTrack(), cv::Scalar(0,0,255), 3, CV_AA);

				/* Update the window that displays the backprojections */
				cv::imshow(backWindowName, camShift.getBackprojection());
			}

			/* Update the window that displays the captured raw frames */
			cv::imshow(windowName, capturedRawFrame);

			/* The wait operation is necessary for OpenCV operations to execute properly */
			/* This is where the mouse callback handler is dispatched */
			cv::waitKey(1);
		}

	/* Report any errors */
	} catch (exception& e) {
		cout << e.what() << endl;
	}

	/* Prevent program from closing, immediately */
	cin.ignore(numeric_limits<streamsize>::max(), '\n');
	return 0;
}