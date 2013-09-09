#ifndef FALSEPOSITIVES_H
#define FALSEPOSITIVES_H


#include "opencv2/core/core.hpp"

using namespace cv;
using namespace std;

class FalsePositives
{

private:
    bool valid;
public:
    FalsePositives();
    FalsePositives(Point2f source, Point2f trans, Point2f destination, double distance);
    Point2f src;
    Point2f transformedSrc;
    Point2f dst;
    double dist;

    void addPoint(Point2f source, Point2f trans, Point2f destination, double distance);
    bool isValid();
    void inValidate();
};

#endif // FALSEPOSITIVES_H
