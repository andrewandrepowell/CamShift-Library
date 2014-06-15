/** \author Andrew Powell \date June 14th, 2014 */

#ifndef CAM_SHIFT_H_
#define CAM_SHIFT_H_

#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <exception>


/**
 * \brief Contains the CamShift class
 * \author Andrew Powell
 * \date June 14th, 2014
 */
namespace camShift {

	/**
	 * \brief Carries out the CAMShift algorithm, utilizing OpenCV libraries
	 *
	 * The CamShift class relies on Gary Bradsky's Continuously Adaptive Meanshift (CAMshift) algorithm 
	 * implemented within OpenCV libraries. More information on the algorithm itself can be found at the
	 * webpage whose hyperlink is located below. A brief description of the CAMShift algorithm is presented
	 * here.
	 *
	 * The CAMShift algorithm is, in essence, an adaptive version of the Meanshift algorithm. Say we have a set
	 * of points and a window that encapsulates a subset of the set of points. The set of points could be a set 
	 * of pixels of an image, and the window could be a circle or another shape. The Meanshift algorithm 
	 * takes the window and shifts the window such that the maximum point density is achieved in the circle.
	 * The CAMShift algorithm extends the Meanshift algorithm by also changing the size and rotation of the 
	 * window. 
	 * 
	 * By continuously producing a different image in which the desired object to track has the highest pixel 
	 * density (i.e. the backprojection) and applying the CAMShift algorithm, the resulting windows represent 
	 * where the object is located in each image. 
	 * 
	 * In addition to the operations related to the CAMShift algorithm, the CamShift class also carries out
	 * several filtration methods so as to optimize the results of the algorithm. Please keep in mind the
	 * the CamShift generates backprojections highly based on the color of the desired object.
	 *
	 * \see <a href="http://docs.opencv.org/trunk/doc/py_tutorials/py_video/py_meanshift/py_meanshift.html">Meanshift and Camshift</a>
	 * \author Andrew Powell
	 * \date June 14th, 2014
	 */
	class CamShift {
	public:

		/** \brief Largest Threshold value */
		enum { THRESHOLD_MAXI = 255 };

		/** \brief An enumerator type used to specify a parameter to change and view with the setParameter() and getParameter() methods, respectively */
		enum Parameter { HUE_BINS_C, SAT_BINS_C, VAL_BINS_C, MEDIAN_BLUR_C, THRESHOLD_C };
		
		/** \brief Constructor */
		CamShift();

		/** \brief Destructor */
		~CamShift();

		/** 
		 * \brief Sets the captured raw frame
		 *
		 * The captured raw frame is the image over which the CAMShift algorithm is executed. If the CamShift
		 * class is being employed to determine the location of an object in real-time, the captured raw frame
		 * should be set to every new frame.
		 *
		 * \param capturedRawFrame A reference to the image over which the CAMShift algorithm is executed
		 * \warning setCapturedRawFrame() should be called prior to calling setSelection() and runCamShift().
		 */
		void setCapturedRawFrame(cv::Mat& capturedRawFrame);

		/**
		 * \brief Sets the selection window
		 *
		 * In the context of the CamShift class, selection refers to the window manually set with the 
		 * setSelection() method, whereas track refers to the window calculated as a result of the 
		 * runCamShift() method.
		 *
		 * \param selection A reference to the rectangle that acts as the new window
		 * \throw runtime_error A runtime error is thrown if the selection is invalid. Specifically, the
		 * selection's width and height both must be greater than 0. Moreover, a runtime error is thrown if
		 * the captured raw frame has not been set.
		 */
		void setSelection(cv::Rect& selection);

		/** 
		 * \brief Executes the CAMShift algorithm and other operations intended to optimize the results
		 *
		 * For every new captured raw frame, the runCamShift() should be called in order to determine a new
		 * window. 
		 *
		 * \throw runtime_error The runtime error is thrown if the captured raw frame has not been set.
		 * \warning The methods setSelection() and setCapturedRawFrame() should be called at least once prior
		 * to calling runCamShift().
		 */
		void runCamShift();

		/**
		 * \brief Gets the backprojection
		 * \return Returns a reference to the backprojection
		 * \throw runtime_error The runtime error is thrown in the event the backprojection has not been set.
		 * \warning runCamShift() should be called prior to calling getBackprojection().
		 */
		cv::Mat& getBackprojection();

		/**
		 * \brief Gets the track window
		 *
		 * In the context of the CamShift class, selection refers to the window manually set with the 
		 * setSelection() method, whereas track refers to the window calculated as a result of the 
		 * runCamShift() method.
		 *
		 * The difference between track and a rotated track is as follows. The rotated track window is the
		 * true window produced by running the CAMShift algorithm implemented within OpenCV. The track 
		 * is the rotated track's bounding rectangle, which is never rotated. 
		 *
		 * \return Returns a reference to the track window
		 * \throw runtime_error A runtime_error is thrown in the event track has not been set.
		 * \warning runCamShift() should be called prior to calling getTrack().
		 */
		cv::Rect& getTrack();

		/**
		 * \brief Gets the rotated track window
		 *
		 * In the context of the CamShift class, selection refers to the window manually set with the 
		 * setSelection() method, whereas track refers to the window calculated as a result of the 
		 * runCamShift() method.
		 *
		 * The difference between track and a rotated track is as follows. The rotated track window is the
		 * true window produced by running the CAMShift algorithm implemented within OpenCV. The track 
		 * is the rotated track's bounding rectangle, which is never rotated. 
		 *
		 * \return Returns a reference to the rotated track window
		 * \throw runtime_error A runtime_error is thrown in the event track has not been set.
		 * \warning runCamShift() should be called prior to calling getRotatedTrack().
		 */
		cv::RotatedRect& getRotatedTrack();

		/**
		 * \brief Sets a specified parameter
		 *
		 * Parameters:
		 *
		 * The ranges in parentheses are possible ranges for the parameters
		 *
		 * HUE_BINS_C		- Sets the number of hue bins in the histogram (0 to 179)
		 * SAT_BINS_C		- Sets the number of saturation bins in the histogram (0 to 255)
		 * VAL_BINS_C		- Sets the number of value bins in the histogram (0 to 255)
		 * MEDIAN_BLUR_C	- Sets the size of median blur (odd values greater than 1)
		 * THRESHOLD_C		- Sets the threshold value (0 to 179)
		 *
		 * Description:
		 *
		 * setParameter() can configure parameters that affect how the backprojection is generated. The
		 * backprojections are each generated from a histogram produced once the selection window is set. The
		 * histogram is calculated from the selection window and the captured raw frame that has been converted 
		 * from RGB (i.e. red, green, and blue) to HSV (i.e. hue, saturation, and value). Hue is indicative to
		 * color, saturation is indicative to where the color is on the gray scale, and value refers to 
		 * brightness. The number of bins for each channel (i.e. hue, saturation, and value) effectivly changes
		 * how well and how poorly the backprojections capture the desired object.
		 *
		 * \parameter parameter Specifies which parameter to modify
		 * \parameter newParameter The new value to which the specified parameter is changed
		 * \throw runtime_error A runtime error is thrown if an attempt is made to set the specified parameter
		 * to an invalid value.
		 * \see <a href="http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=medianblur#medianblur">OpenCV's median blur</a>
		 */
		void setParameter(Parameter parameter, long newParameter);

		/**
		 * \brief Gets the value of a specified parameter
		 * \param parameter Specifies which parameter to return
		 * \return Returns the value of a specified parameter
		 */
		long getParameter(Parameter parameter);

	private:
		enum { 
			HUE_MIN = 0, 
			HUE_MAX = 180, 
			HUE_BINS = 20,
			SAT_MIN = 0, 
			SAT_MAX = 256,
			SAT_BINS = 10,
			VAL_MIN = 0, 
			VAL_MAX = 256,
			VAL_BINS = 1,
			WIDTH_MINI = 20,
			HEIGHT_MAXI = 20,
			THRESHOLD = 40,
			MEDIAN_BLUR = 3,
			CHANNELS = 3
		};
		enum { HUE = 0, SAT = 1, VAL = 2, MINI = 0, MAXI = 1 };

		cv::Mat capturedRawFrame;

		float histoRanges[CHANNELS][2];
		cv::Scalar maskRanges[2];
		cv::Rect track;
		cv::RotatedRect trackRotated;
		cv::Mat hsvFrame;
		cv::Mat maskFrame;
		cv::Mat histoFrame;
		cv::Mat backProjectionFrame;
		cv::Mat erosionElement;
		cv::Mat dilationElement;
		int histoBins[CHANNELS];
		int medianBlurAmount;
		int thresholdAmount;
		int channels[CHANNELS];

		void setHsvFrame();
		const float** getConstantHistoRanges();
	};
};

#endif