#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include <stdio.h>
#include <boost/filesystem.hpp>

using namespace cv;
using namespace std;
using namespace boost::filesystem;
class calibratorStereo {
    private:
        string l_path, r_path; //direccion carpetas que contienen las imagenes del patron
        vector<Mat> l_images, r_images; //vectores de imagenes izq y der del patron de calibracion
        Mat l_cameraMatrix, l_distCoeffs, r_cameraMatrix, r_distCoeffs; //matrices que contienen la información de calibracion
        bool show_chess_corners; //booleano que chekea si se visualizan las esquinas del patron de calib
        float side_length; //longitud del lado del cuadrado del patron 
        int width, height; //numero de esquinas internas del patron de calibracion a lo largo y ancho
        vector<vector<Point2f> > l_image_points, r_image_points; //imagenes izquierda y derecha
        vector<vector<Point3f> > object_points; //object points 
        Mat R, T, E, F; //informacion de calibracion estereo
    public:
        calibratorStereo(string, string, float, int, int); //constructor
        bool calibrate(); 
        void calc_image_points(bool); 
        void save_info(string);
        Size get_image_size();
    };

    class rectifier {
    private:
        Mat map_l1, map_l2, map_r1, map_r2; //pixel maps for rectification
        string path;
    public:
        rectifier(string, Size); //constructor
        void show_rectified(Size); //funcion para rectificar en tiempo real el sistema estereo
    };

calibratorStereo::calibrator(string _l_path, string _r_path, float _side_length, int _width, int _height)
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
        cout << "filename: " << l_filename << endl;
        im_name.replace(im_name.begin(), im_name.begin() + 4, string("right"));
        string r_filename = r_path + im_name;
        cout << "filename: " << r_filename << endl;
        Mat lim = imread(l_filename), rim = imread(r_filename);
        if(!lim.empty() && !rim.empty()) {
            l_images.push_back(lim);
            r_images.push_back(rim);
        }
	} 
}

void calibratorStereo::calc_image_points(bool show) {
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
    //Lee la informacion de calibracion de cada camara guardada en el archivo xml
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
    //calcula las transformaciones para rectificacion de imagenes
    Mat Rl, Rr, Pl, Pr, Q;
    stereoRectify(l_cameraMatrix, l_distCoeffs, r_cameraMatrix, r_distCoeffs, image_size, R, T, Rl,
Rr, Pl, Pr, Q);
    //calcula un mapa de pixel para una eficiente rectificacion de imagenes 
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
        //Crea solo una imagen compuesta por las imagenes rectificadas horizontalmente de ambas camaras
        Mat combo(image_size.height, 2 * image_size.width, CV_8UC3);
        framel_rect.copyTo(combo(Range::all(), Range(0, image_size.width)));
        framer_rect.copyTo(combo(Range::all(), Range(image_size.width, 2*image_size.width)));
        //Dibuja lineas horizontales en ambas imagenes para hacer mas facil la comparacion
        for(int y = 0; y < combo.rows; y += 20)
            line(combo, Point(0, y), Point(combo.cols, y), Scalar(0, 0, 255));
        imshow("Combo", combo);
    }
    capl.release();
    capr.release();
}

int main() {
     string filename = "DATA/" + string("stereo_calib.xml");
    /*
    calibrator calib("LEFT_FOLDER/", "RIGHT_FOLDER/", 25.f, 9, 6);
    calib.calc_image_points(true);
    bool done = calib.calibrate();
    if(!done) cout << "No se pudo completar la calibracion estereo ya que las matrices de las camaras no se cargaron correctamente" << endl;
    calib.save_info(filename);
    Size image_size = calib.get_image_size();*/
    Size image_size(320, 240);
    rectifier rec(filename, image_size);
    rec.show_rectified(image_size);
return 0; 
}