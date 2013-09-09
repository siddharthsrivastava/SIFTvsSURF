#include "FalsePositives.h"
#include "opencv2/core/core.hpp"
FalsePositives::FalsePositives()
{
}

FalsePositives::FalsePositives(Point2f source, Point2f trans, Point2f destination, double distance)
{
    src = source;
    transformedSrc = trans;
    dst = destination;
    dist = distance;
    valid = true;
}
 void FalsePositives::addPoint(Point2f source, Point2f trans, Point2f destination, double distance)
 {
    src = source;
    transformedSrc = trans;
    dst = destination;
    dist = distance;
    valid = true;
 }

 void FalsePositives::inValidate()
 {
     valid = false;
 }

 bool FalsePositives::isValid()
 {
     return valid;
 }
