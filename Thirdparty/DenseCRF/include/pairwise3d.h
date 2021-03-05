#pragma once

#include "densecrf_base.h"
#include "permutohedral_cpu.h"
#include <opencv2/core/core.hpp>
#include <vector>

using namespace std;
using namespace cv;

namespace DenseCRF {

template<int M, int F>
class PottsPotential3D : public PairwisePotential {
protected:
    PermutohedralLatticeCPU lattice_;
    float w_;
    float *norm_;
public:
    PottsPotential3D(const float *features, int N, float w) : PairwisePotential(N), w_(w) {
        lattice_.init(features, F, N);
        norm_ = new float[N];
        std::fill(norm_, norm_ + N, 1.0f);
        lattice_.compute(norm_, norm_, 1);
        for (int i = 0; i < N_; ++i) {
            norm_[i] = 1.0f / (norm_[i] + 1e-20f);
        }
    }

    ~PottsPotential3D() {
        delete[] norm_;
    }

    PottsPotential3D(const PottsPotential3D &o) = delete;   

    // add appearanceKernel
    template<class T = float>
    static PottsPotential3D<M, F> *appearanceKernel(int N, float weight, vector<float>& vobserv, vector<float>& verror, float posdev1, float posdev2) {
        // First assemble features:
        auto *allFeatures = new float[F * N];
        for (int idx = 0; idx < N; ++idx) {
            allFeatures[idx * F + 0] = vobserv[idx] / posdev1;
            allFeatures[idx * F + 1] = verror[idx]  / posdev2;
        }
        auto *pt = new PottsPotential3D<M, F>(allFeatures, N, weight);
        delete[] allFeatures;
        return pt;
    }
    
    // add smoothkernel
    template<class T = float>
    static PottsPotential3D<M, F> *smoothKernel(int N, float weight, vector<Point3f>& points3d, vector<Point2f>& points2d, float posdev1, float posdev2) {
        // First assemble features:
        auto *allFeatures = new float[F * N];
        for (int idx = 0; idx < N; ++idx) {
	    //3d points
	  /*
            Point3f point3d = points3d[idx];
            allFeatures[idx * F + 0] = point3d.x / posdev1;
            allFeatures[idx * F + 1] = point3d.y / posdev1;
            allFeatures[idx * F + 2] = point3d.z / posdev1;*/
	    
	    //2d points
	    Point2f point2d = points2d[idx];
	    allFeatures[idx * F + 0] = point2d.x / posdev2;
            allFeatures[idx * F + 1] = point2d.y / posdev2;
        }
        auto *pt = new PottsPotential3D<M, F>(allFeatures, N, weight);
        delete[] allFeatures;
        return pt;
    }

    void apply(float *out_values, const float *in_values, float *tmp) const {
        lattice_.compute(tmp, in_values, M);
        for (int i = 0, k = 0; i < N_; i++)
            for (int j = 0; j < M; j++, k++)
                out_values[k] += w_ * norm_[i] * tmp[k];
    }
};

}
