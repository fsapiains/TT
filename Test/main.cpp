#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <boost/filesystem.hpp>
#include <boost/exception/all.hpp>
#include <opencv2/core/core.hpp>

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <iostream>
#include <stdlib.h>

using namespace boost::filesystem;
using namespace cv;
using namespace std;

void load_images(const string & _path, vector <Mat> & images) {
    try {
        path dir(_path);
        // Read images
        if (exists(dir)) {
            if (is_directory(dir)) {
                for (directory_iterator i(dir), end_iter; i != end_iter; i++) {
                    if (is_regular_file(i->status())) {
                        string filename = dir.string() + i->path().filename().string();

                        Mat img = imread(filename);

                        if (img.empty()) {
                            cout << "No se pudo cargar: " << filename << endl;
                            break;
                        } else {
                            if (img.cols > 0 && img.rows > 0) {
                                Mat img_gray;
                                cvtColor(img, img_gray, CV_BGR2GRAY);
                                resize(img_gray, img_gray, Size(64, 128));
                                images.push_back(img_gray.clone());
                                img_gray.release();
                                cout << "Cargando: " << filename << endl;
                            } else {
                                cout << "No se pudo cargar: " << filename << endl;
                            }
                            img.release();
                        }
                    } else {
                        cout << "No es un archivo regular" << endl;
                    }
                }
                cout << "vector img" << images.size() << endl;
            } else {
                cout << "ruta '" << _path << "' No es una carpeta" << endl;
            }
        } else {
            cout << "ruta '" << _path << "' invalida" << endl;
        }
    } catch (boost::exception &e) {
        cout << "Excepcion lanzada: (" << boost::diagnostic_information(e) << ")" << endl;
    }
}

int main(int argc, char** argv) {

    vector< Mat > img_pos;
    vector< Mat > img_neg;

    load_images("first/pos/", img_pos);
    load_images("first/neg/", img_neg);

    cout << "vector pos: " << img_pos.size() << endl;
    cout << "vector neg: " << img_neg.size() << endl;

    return 0;

}