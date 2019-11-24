#include "G2OType.h"

void XIAOC::EdgeProjectPoseOnly::computeError() { 

    const VertexPose *vpose = static_cast<const VertexPose*>(_vertices[0]);
    SE3 Tcr = vpose->estimate();
    Vec3 pc = Tcr * pr_;
    if (pc[2] <= 0) 
        return;

    pc = pc * (1.0 / pc[2]);
    double u = fx_ * pc[0] + cx_;
    double v = fy_ * pc[1] + cy_;

    _error = Vec2(u, v) - _measurement;
}