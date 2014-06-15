/** \author Andrew Powell \date 6/4/2014 */

#include "CamShift.h"

namespace camShift {

	CamShift::CamShift() : 
			medianBlurAmount(MEDIAN_BLUR),
			thresholdAmount(THRESHOLD) {
	
		histoRanges[HUE][MINI] = HUE_MIN;
		histoRanges[HUE][MAXI] = HUE_MAX;
		histoRanges[SAT][MINI] = SAT_MIN;
		histoRanges[SAT][MAXI] = SAT_MAX;
		histoRanges[VAL][MINI] = VAL_MIN;
		histoRanges[VAL][MAXI] = VAL_MAX;
		maskRanges[MINI] = cv::Scalar(0, 0, 0);
		maskRanges[MAXI] = cv::Scalar(HUE_MAX, SAT_MAX, VAL_MAX);
		histoBins[HUE] = HUE_BINS;
		histoBins[SAT] = SAT_BINS;
		histoBins[VAL] = VAL_BINS;
		for (int i = 0; i < CHANNELS; i++)
			channels[i] = i;

		erosionElement = (cv::Mat_<uchar>(3,3) << 
			0,1,0,
			1,1,1,
			0,1,0);
		dilationElement = (cv::Mat_<uchar>(7,7) <<
			0,0,0,1,0,0,0,
			0,0,1,1,1,0,0,
			0,1,1,1,1,1,0,
			1,1,1,1,1,1,1,
			0,1,1,1,1,1,0,
			0,0,1,1,1,0,0,
			0,0,0,1,0,0,0);
	}

	CamShift::~CamShift() { }

	void CamShift::setSelection(cv::Rect& selection) {
		if (selection.height <= 0 || selection.width <= 0)
			throw std::runtime_error("Invalid selection");
		setHsvFrame();
		cv::Mat regionOfInterestFrame(hsvFrame, selection);
		cv::Mat maskOfMaskFrame(maskFrame, selection);
		cv::calcHist(
			&regionOfInterestFrame, 1, 
			channels, 
			maskOfMaskFrame, 
			histoFrame, CHANNELS, histoBins, getConstantHistoRanges());
		track = selection;
	}

	void CamShift::setCapturedRawFrame(cv::Mat& capturedRawFrame) {
		this->capturedRawFrame = capturedRawFrame;
	}

	void CamShift::runCamShift() {
		setHsvFrame();
		cv::calcBackProject(&hsvFrame, 1, 
			channels, 
			histoFrame, 
			backProjectionFrame, 
			getConstantHistoRanges());
		backProjectionFrame &= maskFrame; // intersection between bpf and mf? This might be useless
		cv::threshold(backProjectionFrame, backProjectionFrame, thresholdAmount, 255, cv::THRESH_BINARY);
		cv::medianBlur(backProjectionFrame, backProjectionFrame, medianBlurAmount);
		cv::erode(backProjectionFrame, backProjectionFrame, erosionElement);
		cv::dilate(backProjectionFrame, backProjectionFrame, dilationElement);

		cv::RotatedRect prevTrackRotated = trackRotated;
		trackRotated = cv::CamShift(backProjectionFrame, track,
					cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
		if (trackRotated.size.width < WIDTH_MINI) {
			trackRotated.size.width = WIDTH_MINI;
		}
		if (trackRotated.size.height < HEIGHT_MAXI) {
			trackRotated.size.height = HEIGHT_MAXI;
		}
		if (trackRotated.center.x <= 0 || 
			trackRotated.center.x > backProjectionFrame.cols) {
			trackRotated.center.x = prevTrackRotated.center.x;
		}
		if (trackRotated.center.y <= 0 || 
			trackRotated.center.y > backProjectionFrame.rows) {
			trackRotated.center.y = prevTrackRotated.center.y;
		}
		track = trackRotated.boundingRect() & 
			cv::Rect(0, 0, backProjectionFrame.cols, backProjectionFrame.rows);
	}

	cv::Mat& CamShift::getBackprojection() {
		if (backProjectionFrame.rows == 0 || backProjectionFrame.cols == 0)
			throw std::runtime_error("Backprojection has not been set");
		return backProjectionFrame;
	}

	cv::Rect& CamShift::getTrack() {
		if (track.height == 0 || track.width == 0)
			throw std::runtime_error("Track has not been set");
		return track;
	}

	cv::RotatedRect& CamShift::getRotatedTrack() {
		if (trackRotated.boundingRect().height <= 0 || trackRotated.boundingRect().width <= 0)
			throw std::runtime_error("Rotated track has not been set");
		return trackRotated;
	}

	void CamShift::setHsvFrame() {
		if (capturedRawFrame.rows == 0 || capturedRawFrame.cols == 0)
			throw std::runtime_error("Captured raw frame has not been set");
		cv::cvtColor(capturedRawFrame, hsvFrame, cv::COLOR_BGR2HSV);
			cv::inRange(hsvFrame, 
				maskRanges[MINI], 
				maskRanges[MAXI],
				maskFrame);
	}

	const float** CamShift::getConstantHistoRanges() {
		static const float* constantHistoRanges[3] = {
			histoRanges[HUE], 
			histoRanges[SAT], 
			histoRanges[VAL]
		};
		return constantHistoRanges;
	}

	void CamShift::setParameter(Parameter parameter, long newParameter) {
		char* errorMessage = NULL;
		char greaterThanZero[] = "parameter must be greater than or equal to 0";
		switch (parameter) {
		case HUE_BINS_C:
			if (newParameter >= 0) {
				histoBins[HUE] = newParameter;
			} else { errorMessage = greaterThanZero; }
			break;
		case SAT_BINS_C:
			if (newParameter >= 0) {
				histoBins[SAT] = newParameter;
			} else { errorMessage = greaterThanZero; }
			break;
		case VAL_BINS_C:
			if (newParameter >= 0) {
				histoBins[VAL] = newParameter;
			} else { errorMessage = greaterThanZero; }
			break;
		case MEDIAN_BLUR_C:
			if (newParameter > 1 && (long)newParameter % 2 == 1) {
				medianBlurAmount = newParameter;
			} else { errorMessage = "parameter must be greater than 1 and odd"; }
			break;
		case THRESHOLD_C:
			if (newParameter >= 0 && newParameter <= 255) {
				thresholdAmount = newParameter;
			} else { errorMessage = "parameter must be greater than or equal to 0, and less than or equal to 255"; }
			break;
		}
		if (errorMessage != NULL)
			throw std::runtime_error(errorMessage);
	}

	long CamShift::getParameter(Parameter parameter) {
		switch (parameter) {
		case HUE_BINS_C:	return histoBins[HUE];
		case SAT_BINS_C:	return histoBins[SAT];
		case VAL_BINS_C:	return histoBins[VAL];
		case MEDIAN_BLUR_C: return medianBlurAmount;
		case THRESHOLD_C:	return thresholdAmount;
		default: return 0;
		}
	}
};
