#include "utils.h"

void convert_ml(vector<cv::Mat> & samples, cv::Mat& trainData) {
    cout << "convirtiendo...." << endl;
    const int rows = (int) samples.size();
    const int cols = (int) std::max(samples[0].cols, samples[0].rows);
    cv::Mat tmp(1, cols, CV_32FC1); //usada para transposicion si es necesario

    cout << "rows num" << rows << endl;
    cout << "cols num" << cols << endl;
    trainData = cv::Mat(rows, cols, CV_32FC1);

    cout << "train data" << endl;
    vector< Mat >::const_iterator itr = samples.begin();
    vector< Mat >::const_iterator end = samples.end();

    for (int i = 0; itr != end; ++itr, ++i) {
        CV_Assert(itr->cols == 1 || itr->rows == 1);
        if (itr->cols == 1) {
            //cout<< "entra al if"<<endl;
            transpose(*(itr), tmp);
            tmp.copyTo(trainData.row(i));
        } else if (itr->rows == 1) {
            //cout<< "entra al else"<<endl;   
            itr->copyTo(trainData.row(i));
        }
        samples.erase(samples.begin() + i);
    }
}

Mat get_hogdescriptor_visual_image(Mat& origImg,
        vector< float>& descriptors_values,
        Size winSize,
        Size cellSize,
        int scaleFactor,
        double viz_factor) {

    Mat visual_image;
    resize(origImg, visual_image, Size(origImg.cols*scaleFactor, origImg.rows * scaleFactor));
    cvtColor(visual_image, visual_image, CV_GRAY2BGR);

    int gradientBinSize = 9;
    // dividing 180° into 9 bins, how large (in rad) is one bin?
    float radRangeForOneBin = 3.14 / (float) gradientBinSize;

    // prepare data structure: 9 orientation / gradient strenghts for each cell
    int cells_in_x_dir = winSize.width / cellSize.width;
    int cells_in_y_dir = winSize.height / cellSize.height;

    float*** gradientStrengths = new float**[cells_in_y_dir];
    int** cellUpdateCounter = new int*[cells_in_y_dir];
    for (int y = 0; y < cells_in_y_dir; y++) {
        gradientStrengths[y] = new float*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x = 0; x < cells_in_x_dir; x++) {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;

            for (int bin = 0; bin < gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }

    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;

    // compute gradient strengths per cell
    int descriptorDataIdx = 0;

    for (int blockx = 0; blockx < blocks_in_x_dir; blockx++) {
        for (int blocky = 0; blocky < blocks_in_y_dir; blocky++) {
            // 4 cells per block ...
            for (int cellNr = 0; cellNr < 4; cellNr++) {
                // compute corresponding cell nr
                int cellx = blockx;
                int celly = blocky;
                if (cellNr == 1) celly++;
                if (cellNr == 2) cellx++;
                if (cellNr == 3) {
                    cellx++;
                    celly++;
                }

                for (int bin = 0; bin < gradientBinSize; bin++) {
                    float gradientStrength = descriptors_values[ descriptorDataIdx ];
                    descriptorDataIdx++;

                    gradientStrengths[celly][cellx][bin] += gradientStrength;

                } // for (all bins)


                // note: overlapping blocks lead to multiple updates of this sum!
                // we therefore keep track how often a cell was updated,
                // to compute average gradient strengths
                cellUpdateCounter[celly][cellx]++;

            } // for (all cells)


        } // for (all block x pos)
    } // for (all block y pos)


    // compute average gradient strengths
    for (int celly = 0; celly < cells_in_y_dir; celly++) {
        for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {

            float NrUpdatesForThisCell = (float) cellUpdateCounter[celly][cellx];

            // compute average gradient strenghts for each gradient bin direction
            for (int bin = 0; bin < gradientBinSize; bin++) {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }


    //cout << "descriptorDataIdx = " << descriptorDataIdx << endl;

    // draw cells
    for (int celly = 0; celly < cells_in_y_dir; celly++) {
        for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {
            int drawX = cellx * cellSize.width;
            int drawY = celly * cellSize.height;

            int mx = drawX + cellSize.width / 2;
            int my = drawY + cellSize.height / 2;

            rectangle(visual_image,
                    Point(drawX*scaleFactor, drawY * scaleFactor),
                    Point((drawX + cellSize.width) * scaleFactor,
                    (drawY + cellSize.height) * scaleFactor),
                    CV_RGB(100, 100, 100),
                    1);

            // draw in each cell all 9 gradient strengths
            for (int bin = 0; bin < gradientBinSize; bin++) {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];

                // no line to draw?
                if (currentGradStrength == 0)
                    continue;

                float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

                float dirVecX = cos(currRad);
                float dirVecY = sin(currRad);
                float maxVecLen = cellSize.width / 2;
                float scale = viz_factor; // just a visual_imagealization scale,
                // to see the lines better

                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

                // draw gradient visual_imagealization
                line(visual_image,
                        Point(x1*scaleFactor, y1 * scaleFactor),
                        Point(x2*scaleFactor, y2 * scaleFactor),
                        CV_RGB(0, 0, 255),
                        1);

            } // for (all bins)

        } // for (cellx)
    } // for (celly)


    // don't forget to free memory allocated by helper data structures!
    for (int y = 0; y < cells_in_y_dir; y++) {
        for (int x = 0; x < cells_in_x_dir; x++) {
            delete[] gradientStrengths[y][x];
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;

    return visual_image;

}

void load_images(const string & _path, vector <Mat> & images) {
    string path = _path;
    // Read images
    for (directory_iterator i(path), end_iter; i != end_iter; i++) {
        string filename = path + i->path().filename().string();
        //cout << filename << endl;
        Mat img = imread(filename);
        if (img.empty())
            break;
        //imshow("imagen", img);
        images.push_back(img.clone());
        img.release();
        //img= imread(filename);
    }
}

void get_hogs(vector <Mat> & images, vector <Mat> & gradients, const Size & size) {
    HOGDescriptor hog;

    hog.winSize = size;

    Mat img_gray, showHOG;
    vector<float> descriptors_values;
    vector<Point> locations;

    //    vector< Mat >::iterator img = images.begin();
    //    vector< Mat >::iterator end = images.end();

    cout << "calculando descriptores... " << endl;

    //    for (; img != end; ++img) {
    for (unsigned long i = 0; i < images.size(); i++) {
        Mat imgMat = images.at(i);
        vector< Mat >::iterator img = images.begin() + i;

        //imshow("original", img);
        //waitKey(15);
        //resize(img,img, Size(64,128));
        cvtColor(*img, img_gray, CV_BGR2GRAY);

        //HOGDescriptor hog(Size(64,128), Size(16,16), Size(8,8), Size(8,8), 9);

        hog.compute(img_gray, descriptors_values, Size(8, 8), Size(0, 0), locations);

        gradients.push_back(Mat(descriptors_values).clone());
        Mat(descriptors_values).release();
        images.erase(img);

        //showHOG= get_hogdescriptor_visual_image(img_gray, descriptors_values, size, Size(8,8), 3, 2.5);
        //imshow("hog",showHOG);
        //waitKey(0);
    }
    cout << "descriptores calculados correctamente... " << endl;
}

void train_svm(vector < Mat> & gradient, const vector <int> & labels) {
    PrimalSVM svm;
    Mat train_data;
    cout << "convirtiendo a vector unidimensional (machine learning)" << endl;
    convert_ml(gradient, train_data);

    SVMParams params;
    params.svm_type = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    cout << "comenzando entrenamiento..." << endl;

    svm.train(train_data, Mat(labels), Mat(), Mat(), params);

    svm.save("INRIAPerson/SVM.xml");

    cout << "archivo de entrenamiento guardado correctamente..." << endl;
}

void test(const Size & size) {
    Mat img, img_gray;
    vector<Rect> found, found_filtered;
    vector <float> detector;

    //CvSVM svm;
    PrimalSVM svm;
    HOGDescriptor hog;
    hog.winSize = size;

    svm.load("INRIAPerson/SVM.xml");

    svm.getSupportVector(detector);
    hog.setSVMDetector(detector);

    int k = 0;
    VideoCapture cap1 = VideoCapture(0);
    cap1.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap1.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

    while (1) {
        cap1 >> img;

        if (img.empty())
            break;

        cvtColor(img, img_gray, CV_BGR2GRAY);

        found.clear();
        found_filtered.clear();

        hog.detectMultiScale(img_gray, found);
        //trained_hog.detectMultiScale(img_gray, found, 0, Size(8,8), Size(16,16), 1.05, 2);

        //t = (double)getTickCount() - t;
        //printf("tdetection time = %gms\n", t*1000./cv::getTickFrequency());
        size_t i, j;

        for (i = 0; i < found.size(); i++) {
            Rect r = found[i];
            for (j = 0; j < found.size(); j++)
                if (j != i && (r & found[j]) == r)
                    break;
            if (j == found.size())
                found_filtered.push_back(r);
        }
        for (i = 0; i < found_filtered.size(); i++) {
            Rect r = found_filtered[i];
            //se ajusta el resultado del detector y dibuja un rectangulo que contiene la persona detectada
            r.x += cvRound(r.width * 0.1);
            r.width = cvRound(r.width * 0.8);
            r.y += cvRound(r.height * 0.07);
            r.height = cvRound(r.height * 0.8);
            rectangle(img_gray, r.tl(), r.br(), cv::Scalar(255, 255, 0), 3);
            /*   
               //obtiene centro de masa aproximado 
               circle( img_gray, Point(r.x + r.width/2,  r.y+r.height/2), 5.0, Scalar( 255, 255, 255 ), 1, 8 );
               printf("punto x mc: %d\n", r.x + r.width/2 );
               printf("punto y mc: %d\n", r.y + r.height/2 );*/
        }
        imshow("people detector", img_gray);

        k = waitKey(5);

        if (k == 27) {
            break;
        }
    }

    cap1.release();
}
