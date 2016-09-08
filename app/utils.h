#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>

#include "PrimalSVM.h"

using namespace cv;
using namespace std;
using namespace boost::filesystem;

void convert_ml(vector< cv::Mat> & samples, cv::Mat& trainData);
Mat get_hogdescriptor_visual_image(Mat& origImg, vector< float>& descriptors_values, Size winSize, Size cellSize, int scaleFactor, double viz_factor);
void load_images(const string & _path, vector <Mat> & images);
void get_hogs(vector <Mat> & images, vector <Mat> & gradients, const Size & size);
void train_svm(vector < Mat> & gradient, const vector <int> & labels);
void test(const Size & size);


#endif /* UTILS_H */

