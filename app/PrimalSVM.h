#ifndef PRIMALSVM_H
#define PRIMALSVM_H

#include <vector>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

class PrimalSVM : public SVM {
public:
    void getSupportVector(vector<float>& support_vector) const;
};

#endif /* PRIMALSVM_H */

