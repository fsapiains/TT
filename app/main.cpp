#include <iostream>

#include "PrimalSVM.h"
#include "utils.h"

using namespace boost::filesystem;
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    Mat img, img_gray;
    vector< Mat > img_pos;
    vector< Mat > img_neg;
    vector< Mat > gradient;
    vector< int > labels;


    load_images("INRIAPerson/train_64x128_H96/pos/", img_pos);
    labels.assign(img_pos.size(), +1);
    const unsigned long old = (unsigned long) labels.size();
    load_images("INRIAPerson/train_64x128_H96/neg/", img_neg);
    labels.insert(labels.end(), img_neg.size(), -1);
    CV_Assert(old < labels.size());

    cout << "Cantidad de Etiquetas: " << labels.size() << endl;

    get_hogs(img_pos, gradient, Size(64, 128));
    cout << "Gradiente Positivo: " << gradient.size() << endl;

    get_hogs(img_neg, gradient, Size(64, 128));
    cout << "Gradiente Negativo: " << gradient.size() << endl;

    train_svm(gradient, labels);

    // test(Size (64,128));

    return EXIT_SUCCESS;
}