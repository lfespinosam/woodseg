#ifndef CHANVESE_H
#define CHANVESE_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

struct CVsetup
{
  double dt; // time step
  double h;  // pixel spacing
  double lambda1;
  double lambda2;
  double mu; // contour length weighting parameter
  double nu; // region area weighting parameter
  unsigned int p; // length weight exponent
};

class chanvese
{
public:
    chanvese();
    void ChanVeseSegmentation(Mat img, Mat phi0, Mat &phi, struct CVsetup* pCVinputs);
    void segment(Mat im_inrgb, Mat &im_out, Mat mask);
    void segment2phi(Mat im_inrgb, Mat &im_out, Mat mask);
};

#endif // CHANVESE_H
