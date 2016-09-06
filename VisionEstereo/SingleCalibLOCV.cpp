#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include <boost/filesystem.hpp>
//#include "Config.h"
#include <iomanip>
using namespace cv;
using namespace std;
using namespace boost::filesystem;
class calibrator {
    private:
        string path;
        vector<Mat> images;
        Mat cameraMatrix, distCoeffs;
        bool show_chess_corners;
        float side_length;
        int width, height;
        //stringstream name;
//path of folder containing chessboard images
//chessboard images
//camera matrix and distortion coefficients
//visualize the extracted chessboard corners?
//side length of a chessboard square in mm
//number of internal corners of the chessboard along width and height
        vector<vector<Point2f> > image_points;  //2D image points
        vector<vector<Point3f> > object_points; //3D object points
public:
    //collect(string);
    calibrator(string, float, int, int);
    void calibrate();
    Mat get_cameraMatrix();
    Mat get_distCoeffs();
    void calc_image_points(bool);
//constructor, reads in the images
//function to calibrate the camera
//access the camera matrix
//access the distortion coefficients
//calculate internal corners of the chessboard image
};

/*void calibrator::collect(string _path){
    path=_path
    VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);

    namedWindow("capture");

    cout << "Press 'c' to capture ..." << endl;
    char choice = 'z';
    int count = 0;
    while(choice != 'q') {
    //grab frames quickly in succession
    cap.grab();
    //execute the heavier decoding operations
    cap.retrieve(frame);
    if(frame.empty()) break;
    imshow("Left", frame);
    if(choice == 'c') {
        //save files at proper locations if user presses 'c'
        name << "imageleft" << setw(4) << setfill('0') << count << ".jpg";
        imwrite(string(path + name.str(), frame);
        cout << "Saved set " << count << endl;
        count++;
        }
    choice = char(waitKey(1));
    }
    cap.release();
}*/
calibrator::calibrator(string _path, float _side_length, int _width, int _height) {
    side_length = _side_length;
    width = _width;
    height = _height;
    path = _path;
    cout << path << endl;
    // Read images
    for(directory_iterator i(path), end_iter; i != end_iter; i++) {
        string filename = path + i->path().filename().string();
        //cout << filename << endl;
        images.push_back(imread(filename));
    }
}

void calibrator::calc_image_points(bool show) {
    // Calculate the object points in the object co-ordinate system (origin at top left corner)
    vector<Point3f> ob_p;
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            ob_p.push_back(Point3f(float(j * side_length), float(i * side_length), 0.f));
        } 
    }

    if(show) namedWindow("Chessboard corners");
    for(int i = 1; i < images.size(); i++) {
        Mat im = images[i];
        vector<Point2f> im_p;
        //find corners in the chessboard image
        bool pattern_found = findChessboardCorners(im, Size(width, height), im_p,
CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE+ CALIB_CB_FAST_CHECK);
        cout << pattern_found << endl;
        if(pattern_found) {
            object_points.push_back(ob_p);
            Mat gray;
            cvtColor(im, gray, CV_BGR2GRAY);
            cornerSubPix(gray, im_p, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS +
CV_TERMCRIT_ITER, 30, 0.1));
            image_points.push_back(im_p);
            if(show) {
                Mat im_show = im.clone();
                drawChessboardCorners(im_show, Size(width, height), im_p, true);
                imshow("Chessboard corners", im_show);
                while(char(waitKey(1)) != ' ') {}
            }
        }
        //if a valid pattern was not found, delete the entry from vector of images
        else {
            images.erase(images.begin() + i);
        }

    }
//}
//void calibrator::calibrate() {
    //cout << "calibrate" << images.size() << endl;
    vector<Mat> rvecs, tvecs;
   /* cameraMatrix.at<float>(0,0)=1;
    cameraMatrix.at<float>(1,1)=1;*/
    //cout<< images[1].size()<< endl;
    double rms_error = calibrateCamera(object_points, image_points, images[1].size(), cameraMatrix,
distCoeffs, rvecs, tvecs);
    cout << "RMS reprojection error " << rms_error << endl;
}
Mat calibrator::get_cameraMatrix() {
    return cameraMatrix;
}
Mat calibrator::get_distCoeffs() {
    return distCoeffs;
}

int main() {
    /*VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);

    namedWindow("capture");

    cout << "Press 'c' to capture ..." << endl;
    char choice = 'z';
    int count = 0;
    while(choice != 'q') {
        //grab frames quickly in succession
        cap.grab();
        //execute the heavier decoding operations
        Mat frame;
        cap.retrieve(frame);
        if(frame.empty()) break;
        imshow("Left", frame);
        if(choice == 'c') {
            //save files at proper locations if user presses 'c'
            stringstream name, filename;
            name << "imageleft" << setw(4) << setfill('0') << count << ".jpg";
            imwrite(string("IMAGES/LEFT_FOLDER") + name.str(), frame);
            cout << "Saved set " << count << endl;
            count++;
            }
        choice = char(waitKey(1));
    }
    cap.release();*/
    calibrator calib("IMAGES/LEFT_FOLDER/", 25.f, 9, 6);
    calib.calc_image_points(true);
    cout << "Calibrating camera..." << endl;
    //calib.calibrate();
    //save the calibration for future use
    string filename = "DATA/" + string("left_cam_calib.xml");
    FileStorage fs(filename, FileStorage::WRITE);
    fs << "cameraMatrix" << calib.get_cameraMatrix();
    fs << "distCoeffs" << calib.get_distCoeffs();
    fs.release();
    cout << "Saved calibration matrices to " << filename << endl;
    return 0; 
}