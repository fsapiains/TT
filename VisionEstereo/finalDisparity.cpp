#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include <boost/filesystem.hpp>
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
        vector<vector<Point2f> > image_points;  //2D image points
        vector<vector<Point3f> > object_points; //3D object points
public:
    //collect(string);
    calibrator(string, float, int, int);
    void calibrate();
    Mat get_cameraMatrix();
    Mat get_distCoeffs();
    void calc_image_points(bool);
};

class calibratorStereo {
    private:
        string l_path, r_path; //path for folders containing left and right checkerboard images
        vector<Mat> l_images, r_images; //left and right checkerboard images
        Mat l_cameraMatrix, l_distCoeffs, r_cameraMatrix, r_distCoeffs; //Mats for holding individual camera calibration information
        bool show_chess_corners; //visualize checkerboard corner detections?
        float side_length; //side length of checkerboard squares
        int width, height; //number of internal corners in checkerboard along width and height
        vector<vector<Point2f> > l_image_points, r_image_points; //left and right image points
        vector<vector<Point3f> > object_points; //object points (grid)
        Mat R, T, E, F; //stereo calibration information
    public:
        calibratorStereo(string, string, float, int, int); //constructor
        bool calibrate(); //function to calibrate stereo camera
        void calc_image_points_stereo(bool); //function to calculae image points by detecting checkerboard corners 
        void save_info(string);
        Size get_image_size();
    };

    class rectifier {
    private:
        Mat map_l1, map_l2, map_r1, map_r2; //pixel maps for rectification
        string path;
    public:
        rectifier(string, Size); //constructor
        void show_rectified(Size); //function to show live rectified feed from stereo camera
    };

class disparity {
    private:
        Mat map_l1, map_l2, map_r1, map_r2, Q;
        StereoSGBM stereo;
        int min_disp, num_disp;
        public:
        disparity(string, Size);
        void show_disparity(Size);
};


calibrator::calibrator(string _path, float _side_length, int _width, int _height) {
    side_length = _side_length;
    width = _width;
    height = _height;
    path = _path;
    // Read images
    for(directory_iterator i(path), end_iter; i != end_iter; i++) {
        string filename = path + i->path().filename().string();
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
    vector<Mat> rvecs, tvecs;
   /* cameraMatrix.at<float>(0,0)=1;
    cameraMatrix.at<float>(1,1)=1;*/
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

calibratorStereo::calibratorStereo(string _l_path, string _r_path, float _side_length, int _width, int _height)
{
    side_length = _side_length;
    width = _width;
    height = _height;
    l_path = _l_path;
    r_path = _r_path;
    // Read images
    for(directory_iterator i(l_path), end_iter; i != end_iter; i++) {
        string im_name = i->path().filename().string();
        string l_filename = l_path + im_name;
        im_name.replace(im_name.begin(), im_name.begin() + 4, string("right"));
        string r_filename = r_path + im_name;
        Mat lim = imread(l_filename), rim = imread(r_filename);
        if(!lim.empty() && !rim.empty()) {
            l_images.push_back(lim);
            r_images.push_back(rim);
        }
	} 
}

void calibratorStereo::calc_image_points_stereo(bool show) {
    // Calculate the object points in the object co-ordinate system (origin at top left corner)
    vector<Point3f> ob_p;
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            ob_p.push_back(Point3f(float(j * side_length), float(i * side_length), 0.f));
        } 
    }
    if(show) {
        namedWindow("Left Chessboard corners");
        namedWindow("Right Chessboard corners");
    }

    for(int i = 0; i < l_images.size(); i++) {
        Mat lim = l_images[i], rim = r_images[i];
        vector<Point2f> l_im_p, r_im_p;
        bool l_pattern_found = findChessboardCorners(lim, Size(width, height), l_im_p,
CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE+ CALIB_CB_FAST_CHECK);
        bool r_pattern_found = findChessboardCorners(rim, Size(width, height), r_im_p,
CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE+ CALIB_CB_FAST_CHECK);
        if(l_pattern_found && r_pattern_found) {
            object_points.push_back(ob_p);
            Mat gray;
            cvtColor(lim, gray, CV_BGR2GRAY);
            cornerSubPix(gray, l_im_p, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS +
CV_TERMCRIT_ITER, 30, 0.1));
            cvtColor(rim, gray, CV_BGR2GRAY);
            cornerSubPix(gray, r_im_p, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS +
CV_TERMCRIT_ITER, 30, 0.1));
            l_image_points.push_back(l_im_p);
            r_image_points.push_back(r_im_p);
            if(show) {
                Mat im_show = lim.clone();
                drawChessboardCorners(im_show, Size(width, height), l_im_p, true);
                imshow("Left Chessboard corners", im_show);
                im_show = rim.clone();
                drawChessboardCorners(im_show, Size(width, height), r_im_p, true);
                imshow("Right Chessboard corners", im_show);
                while(char(waitKey(1)) != ' ') {}
			} 
		}
        else {
            l_images.erase(l_images.begin() + i);
            r_images.erase(r_images.begin() + i);
		} 
	}
}
bool calibratorStereo::calibrate() {
    string filename = "DATA/"+ string("left_cam_calib.xml");
    FileStorage fs(filename, FileStorage::READ);
    fs["cameraMatrix"] >> l_cameraMatrix;
    fs["distCoeffs"] >> l_distCoeffs;
    fs.release();
    filename = "DATA/" + string("right_cam_calib.xml");
    fs.open(filename, FileStorage::READ);
    fs["cameraMatrix"] >> r_cameraMatrix;
    fs["distCoeffs"] >> r_distCoeffs;
    fs.release();

    if(!l_cameraMatrix.empty() && !l_distCoeffs.empty() && !r_cameraMatrix.empty() &&
!r_distCoeffs.empty()) {
        double rms = stereoCalibrate(object_points, l_image_points, r_image_points,
                    l_cameraMatrix, l_distCoeffs, r_cameraMatrix, r_distCoeffs, l_images[0].size() , R, T, E, F);
        cout << "Calibrated stereo camera with a RMS error of " << rms << endl;
    return true;
    }
else return false;
}

void calibratorStereo::save_info(string filename) {
        FileStorage fs(filename, FileStorage::WRITE);
        fs << "l_cameraMatrix" << l_cameraMatrix;
        fs << "r_cameraMatrix" << r_cameraMatrix;
        fs << "l_distCoeffs" << l_distCoeffs;
        fs << "r_distCoeffs" << r_distCoeffs;
        fs << "R" << R;
        fs << "T" << T;
        fs << "E" << E;
        fs << "F" << F;
        fs.release();
        cout << "Calibration parameters saved to " << filename << endl;
}

Size calibratorStereo::get_image_size() {
    return l_images[0].size();
}
rectifier::rectifier(string filename, Size image_size) {
    // Read individal camera calibration information from saved XML file
    Mat l_cameraMatrix, l_distCoeffs, r_cameraMatrix, r_distCoeffs, R, T;
    FileStorage fs(filename, FileStorage::READ);
    fs["l_cameraMatrix"] >> l_cameraMatrix;
    fs["l_distCoeffs"] >> l_distCoeffs;
    fs["r_cameraMatrix"] >> r_cameraMatrix;
    fs["r_distCoeffs"] >> r_distCoeffs;
    fs["R"] >> R;
    fs["T"] >> T;
    fs.release();
    if(l_cameraMatrix.empty() || r_cameraMatrix.empty() || l_distCoeffs.empty() ||
r_distCoeffs.empty() || R.empty() || T.empty())
        cout << "Rectifier: Loading of files not successful" << endl;
    // Calculate transforms for rectifying images
    Mat Rl, Rr, Pl, Pr, Q;
    stereoRectify(l_cameraMatrix, l_distCoeffs, r_cameraMatrix, r_distCoeffs, image_size, R, T, Rl,
Rr, Pl, Pr, Q);
    // Calculate pixel maps for efficient rectification of images via lookup tables
    initUndistortRectifyMap(l_cameraMatrix, l_distCoeffs, Rl, Pl, image_size, CV_16SC2, map_l1,
map_l2);
    initUndistortRectifyMap(r_cameraMatrix, r_distCoeffs, Rr, Pr, image_size, CV_16SC2, map_r1,
map_r2);
    fs.open(filename, FileStorage::APPEND);
    fs << "Rl" << Rl;
    fs << "Rr" << Rr;
    fs << "Pl" << Pl;
    fs << "Pr" << Pr;
    fs << "Q" << Q;
    fs << "map_l1" << map_l1;
    fs << "map_l2" << map_l2;
    fs << "map_r1" << map_r1;
    fs << "map_r2" << map_r2;
    fs.release();
}
void rectifier::show_rectified(Size image_size) {
    VideoCapture capr(0), capl(1);
    //reduce frame size
    capl.set(CV_CAP_PROP_FRAME_HEIGHT, image_size.height);
    capl.set(CV_CAP_PROP_FRAME_WIDTH, image_size.width);
    capr.set(CV_CAP_PROP_FRAME_HEIGHT, image_size.height);
    capr.set(CV_CAP_PROP_FRAME_WIDTH, image_size.width);
    destroyAllWindows();
    namedWindow("Combo");
    while(char(waitKey(1)) != 'q') {
        //grab raw frames first
        capl.grab();
        capr.grab();
        //decode later so the grabbed frames are less apart in time
        Mat framel, framel_rect, framer, framer_rect;
        capl.retrieve(framel);
        capr.retrieve(framer);
        if(framel.empty() || framer.empty()) break;
        // Remap images by pixel maps to rectify
        remap(framel, framel_rect, map_l1, map_l2, INTER_LINEAR);
        remap(framer, framer_rect, map_r1, map_r2, INTER_LINEAR);
        // Make a larger image containing the left and right rectified images side-by-side
        Mat combo(image_size.height, 2 * image_size.width, CV_8UC3);
        framel_rect.copyTo(combo(Range::all(), Range(0, image_size.width)));
        framer_rect.copyTo(combo(Range::all(), Range(image_size.width, 2*image_size.width)));
        // Draw horizontal red lines in the combo image to make comparison easier
        for(int y = 0; y < combo.rows; y += 20)
            line(combo, Point(0, y), Point(combo.cols, y), Scalar(0, 0, 255));
        imshow("Combo", combo);
    }
    capl.release();
    capr.release();
}

disparity::disparity(string filename, Size image_size) {
    FileStorage fs(filename, FileStorage::READ);
    fs["map_l1"] >> map_l1;
    fs["map_l2"] >> map_l2;
    fs["map_r1"] >> map_r1;
    fs["map_r2"] >> map_r2;
    fs["Q"] >> Q;
    if(map_l1.empty() || map_l2.empty() || map_r1.empty() || map_r2.empty() || Q.empty())
        cout << "WARNING: Loading of mapping matrices not successful" << endl;
    stereo.preFilterCap = 4;
    stereo.numberOfDisparities= 192;
    stereo.SADWindowSize = 5;
    stereo.P1 = 600;//8 * 3 * stereo.SADWindowSize * stereo.SADWindowSize;
    stereo.P2 = 2400;//32 * 3 * stereo.SADWindowSize * stereo.SADWindowSize;
    stereo.uniquenessRatio = 1;
    stereo.minDisparity= -64;
    stereo.speckleWindowSize = 150;
    stereo.speckleRange = 2;
    stereo.disp12MaxDiff = 10;
    stereo.fullDP = false;
}
void disparity::show_disparity(Size image_size) {
    VideoCapture capr(0), capl(1);
    //reduce frame size
    capl.set(CV_CAP_PROP_FRAME_HEIGHT, image_size.height);
    capl.set(CV_CAP_PROP_FRAME_WIDTH, image_size.width);
    capr.set(CV_CAP_PROP_FRAME_HEIGHT, image_size.height);
    capr.set(CV_CAP_PROP_FRAME_WIDTH, image_size.width);
    namedWindow("Disparity", CV_WINDOW_NORMAL);
    namedWindow("Left", CV_WINDOW_NORMAL);
  
    while(char(waitKey(1)) != 'q') {
        //grab raw frames first
        capl.grab();
        capr.grab();
        //decode later so the grabbed frames are less apart in time
        Mat framel, framel_rect, framer, framer_rect;
        capl.retrieve(framel);
        capr.retrieve(framer);
        if(framel.empty() || framer.empty()) break;
        remap(framel, framel_rect, map_l1, map_l2, INTER_LINEAR);
        remap(framer, framer_rect, map_r1, map_r2, INTER_LINEAR);
        Mat disp, disp_show, disp_compute, pointcloud;
        stereo(framel_rect, framer_rect, disp);
        disp.convertTo(disp_show, CV_8U, 255/(stereo.numberOfDisparities * 16.));
        disp.convertTo(disp_compute, CV_32F, 1.f/16.f);
        //normalize(disp, disp_show, 0, 255, CV_MINMAX, CV_8U);
        // Calculate 3D co-ordinates from disparity image
        reprojectImageTo3D(disp_compute, pointcloud, Q, true);
        // Draw red rectangle around 40 px wide square area im image
        int xmin = framel.cols/2 - 20, xmax = framel.cols/2 + 20, ymin = framel.rows/2 - 20,
ymax = framel.rows/2 + 20;
        rectangle(framel_rect, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255));
        // Extract depth of 40 px rectangle and print out their mean
        pointcloud = pointcloud(Range(ymin, ymax), Range(xmin, xmax));
        Mat z_roi(pointcloud.size(), CV_32FC1);
        int from_to[] = {2, 0};
        mixChannels(&pointcloud, 1, &z_roi, 1, from_to, 1);
        cout << "Depth: " << mean(z_roi) << " mm" << endl;
        imshow("Disparity", disp_show);
        imshow("Left", framel_rect);
    }
    capl.release();
    capr.release();
}

int main() {
	cout << "calibracion simple camara 1.... " << endl;
	calibrator calibSingle1("IMAGES/LEFT_FOLDER/", 25.f, 9, 6);
    calibSingle1.calc_image_points(true);
    cout << "Calibrating camera..." << endl;
    //calib.calibrate();
    //save the calibration for future use
    string filenameSingle1 = "DATA/" + string("left_cam_calib.xml");
    FileStorage fs(filenameSingle1, FileStorage::WRITE);
    fs << "cameraMatrix" << calibSingle1.get_cameraMatrix();
    fs << "distCoeffs" << calibSingle1.get_distCoeffs();
    fs.release();
    cout << "Calibracion exitosa guardada en " << filenameSingle1 << endl;

    cout << " calibracion simple camara 2.... " << endl;
	calibrator calibSingle2("IMAGES/RIGHT_FOLDER/", 25.f, 9, 6);
    calibSingle2.calc_image_points(true);
    cout << "Calibrating camera..." << endl;
    //calib.calibrate();
    //save the calibration for future use
    string filenameSingle2 = "DATA/" + string("right_cam_calib.xml");
    FileStorage fs1(filenameSingle2, FileStorage::WRITE);
    fs1 << "cameraMatrix" << calibSingle2.get_cameraMatrix();
    fs1 << "distCoeffs" << calibSingle2.get_distCoeffs();
    fs1.release();
    cout << "Calibracion exitosa guardada en " << filenameSingle2 << endl;

    cout << "calibrando ambas camaras..." << endl;
    string filenameStereo = "DATA/" + string("stereo_calib.xml");
    calibratorStereo calibStereo("IMAGES/LEFT_FOLDER/", "IMAGES/RIGHT_FOLDER/", 25.f, 9, 6);
    calibStereo.calc_image_points_stereo(true);
    bool done = calibStereo.calibrate();
    if(!done) cout << "La calibracion estereo no se completo porque las matrices individuales no pudieron ser leidas" << endl;
    calibStereo.save_info(filenameStereo);
    //Size image_size = calibStereo.get_image_size();
    Size image_size(320, 240);
    rectifier rec(filenameStereo, image_size);
    rec.show_rectified(image_size);
    disparity disp(filenameStereo, image_size);
    disp.show_disparity(image_size);
    return 0; 
}

