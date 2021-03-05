#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>
#include "fundamental_estimator.h"
#include "GCRANSAC.h"
#include "grid_neighborhood_graph.h"
#include "utils.h"
#include "graph.h"
#include "densecrf3d.h"
#include "pairwise3d.h"

using namespace DenseCRF;
using namespace std;
using namespace cv;



//==============================add by dzj 0825======================================================
vector<int> Tracking::GraphCutRansacFilter(Mat& points){
  int cell_number_in_neighborhood_graph_ = 16;   //8
  int cols = 640, rows = 480;
  GridNeighborhoodGraph neighborhood(&points,
		cols / static_cast<double>(cell_number_in_neighborhood_graph_),
		rows / static_cast<double>(cell_number_in_neighborhood_graph_),
		cols / static_cast<double>(cell_number_in_neighborhood_graph_),
		rows / static_cast<double>(cell_number_in_neighborhood_graph_),
		cell_number_in_neighborhood_graph_);
  
  FundamentalMatrixEstimator estimator;
  FundamentalMatrix model;
  GCRANSAC<FundamentalMatrixEstimator, GridNeighborhoodGraph> gcransac;
  
  gcransac.setFPS(-1); // Set the desired FPS (-1 means no limit)
  gcransac.settings.threshold = 0.5; //1.2  2 The inlier-outlier threshold
  gcransac.settings.spatial_coherence_weight = 0.14; //0.14 The weight of the spatial coherence term
  gcransac.settings.confidence = 0.95f; // The required confidence in the results
  gcransac.settings.max_local_optimization_number = 50; // The maximm number of local optimizations
  gcransac.settings.max_iteration_number = 5000; // The maximum number of iterations
  gcransac.settings.min_iteration_number = 100; //50 The minimum number of iterations
  gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball
  gcransac.settings.core_number = 8; // The number of parallel processes
 
  gcransac.run(points,estimator,&neighborhood,model);
  const RANSACStatistics statistics = gcransac.getRansacStatistics();
  if(statistics.inliers.size() == 0){
    vector<int> v(points.rows);
    for(int i = 0; i < points.rows; i++)  v[i] = i;
    return v;
  }
  return statistics.inliers;
}

void Tracking::DrawMatchedFeatures(Mat& points, Mat& imgl, Mat& imgr, string name){
    //return;
    Mat match_image; 
    vector<int> inliers(points.rows);
    for(int i = 0; i < inliers.size(); i++) inliers[i] = i;
    drawMatches(points,inliers,imgl,imgr,match_image);
    if(display)
      showImage(match_image, name,900,675,true);
}

void Tracking::DrawMatchedFeatures(Mat& points, Mat& imgl, Mat& imgr, vector<int>& inliers, string name){
    //return;
    Mat match_image; 
    drawMatches(points,inliers,imgl,imgr,match_image);
    if(display)
      showImage(match_image, name,900,675,true);
}

map<int,int> Tracking::BfMatch(Frame& fl, Frame& fr){
    vector<tuple<double, int, int> > correspondences;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    vector<vector< DMatch > > matches_vector;
    matcher.knnMatch(fl.mDescriptors, fr.mDescriptors, matches_vector, 2);
    for (auto match : matches_vector){
	if (match.size() == 2 && match[0].distance < match[1].distance * 0.65)
	    correspondences.emplace_back(make_tuple<double, int, int>(match[0].distance / match[1].distance, (int)match[0].queryIdx, (int)match[0].trainIdx));
    }
    // Sort the points for PROSAC
    /*
    std::sort(correspondences.begin(), correspondences.end(), [](const std::tuple<double, int, int>& correspondence_1_,
	const std::tuple<double, int, int>& correspondence_2_) -> bool{
	return std::get<0>(correspondence_1_) < std::get<0>(correspondence_2_);
    });
    */
    map<int,int> asso;
    for(int i = 0; i < correspondences.size(); i++){
      int fid1 = get<1>(correspondences[i]);
      int fid2 = get<2>(correspondences[i]);
      asso[fid1] = fid2;
    }
    return asso;
}

Mat Tracking::makeMatchedFeartures2Mat(Frame& fl, Frame& fr, vector<pair<int,int> >& pairs){
    Mat points = Mat(static_cast<int>(pairs.size()), 4, CV_64F);
    double *points_ptr = reinterpret_cast<double*>(points.data);
    for (int i = 0; i < pairs.size(); i++){
      int fid1 = pairs[i].first, fid2 = pairs[i].second;
      cv::Point2d point_1 = (cv::Point2d)fl.mvKeysUn[fid1].pt;
      cv::Point2d point_2 = (cv::Point2d)fr.mvKeysUn[fid2].pt;
      *(points_ptr++) = point_1.x; 
      *(points_ptr++) = point_1.y;
      *(points_ptr++) = point_2.x; 
      *(points_ptr++) = point_2.y;
    }
    return points;
}

Mat Tracking::makeMatchedFeartures2Mat(Frame& fl, Frame& fr, map<int,int>& maps){
    Mat points = Mat(static_cast<int>(maps.size()), 4, CV_64F);
    double *points_ptr = reinterpret_cast<double*>(points.data);
    for (map<int,int>::iterator it = maps.begin(); it != maps.end(); it++){
      int fid1 = it->first, fid2 = it->second;
      cv::Point2d point_1 = (cv::Point2d)fl.mvKeysUn[fid1].pt;
      cv::Point2d point_2 = (cv::Point2d)fr.mvKeysUn[fid2].pt;
      *(points_ptr++) = point_1.x; 
      *(points_ptr++) = point_1.y;
      *(points_ptr++) = point_2.x; 
      *(points_ptr++) = point_2.y;
    }
    return points;
}

Mat Tracking::makeMatchedFeartures2Mat(Frame& f, KeyFrame* fk, map<int,int>& maps){
    Mat points = Mat(static_cast<int>(maps.size()), 4, CV_64F);
    double *points_ptr = reinterpret_cast<double*>(points.data);
    for (map<int,int>::iterator it = maps.begin(); it != maps.end(); it++){
      int fid1 = it->first, fid2 = it->second;
      cv::Point2d point_1 = (cv::Point2d)f.mvKeysUn[fid1].pt;
      cv::Point2d point_2 = (cv::Point2d)fk->mvKeysUn[fid2].pt;
      *(points_ptr++) = point_1.x; 
      *(points_ptr++) = point_1.y;
      *(points_ptr++) = point_2.x; 
      *(points_ptr++) = point_2.y;
    }
    return points;
}

void Tracking::ComputeMapPointsError_single(){
  
  float minErr = 1000000.0, maxErr = 0;
  mvReprojectErrors.resize(mCurrentFrame.N, 0);
  for(int i = 0; i < mCurrentFrame.N; i++){
    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
    if(!pMP || pMP->isBad()) continue;
    
    Mat x3Dw = pMP->GetWorldPos(); 
    Mat Rcw = mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    Mat tcw = mCurrentFrame.mTcw.rowRange(0,3).col(3);
    Mat x3Dc = Rcw*x3Dw+tcw;
    float xc = x3Dc.at<float>(0);
    float yc = x3Dc.at<float>(1);
    float invzc = 1.0/x3Dc.at<float>(2);

    if(invzc<0) continue;

    float u = mCurrentFrame.fx*xc*invzc+mCurrentFrame.cx;
    float v = mCurrentFrame.fy*yc*invzc+mCurrentFrame.cy;

    if(u<mCurrentFrame.mnMinX || u>mCurrentFrame.mnMaxX || v<mCurrentFrame.mnMinY || v>mCurrentFrame.mnMaxY)
      continue;
    Point2d kp = mCurrentFrame.mvKeysUn[i].pt;
    
    float error = (u-kp.x)*(u-kp.x) + (v-kp.y)*(v-kp.y);
    mvReprojectErrors[i] = error;

    if(error < minErr) minErr = error;
    else if(maxErr < error) maxErr = error;
  }
  
  cout<<"maxErr:"<<maxErr<<endl;
  for(int i = 0; i < mCurrentFrame.N; i++)
    mvReprojectErrors[i] /= maxErr;
}

void Tracking::ComputeMapPointsError_multi(){
  vector<float> vobservs,vdepths, verrors, vx,vy,vz;
  
  float minErr = 1000000.0, maxErr = 0;
  mvReprojectErrors.resize(mCurrentFrame.N, 0);
  
  for(int i = 0; i < mCurrentFrame.N; i++){
    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
    if(!pMP || pMP->isBad()) continue;
    
    Mat x3Dw = pMP->GetWorldPos();
    
    float errori = 0;
    map<KeyFrame*,size_t> observations = pMP->GetObservations();
    if(observations.size() < 2) continue;
    for(map<KeyFrame*,size_t>::iterator it = observations.begin(); it != observations.end(); it++){
      KeyFrame* pKF = it->first;
      Mat Rcw = pKF->GetPose().rowRange(0,3).colRange(0,3);
      Mat tcw = pKF->GetPose().rowRange(0,3).col(3);
      Mat x3Dc = Rcw*x3Dw+tcw;
      float xc = x3Dc.at<float>(0);
      float yc = x3Dc.at<float>(1);
      float invzc = 1.0/x3Dc.at<float>(2);

      if(invzc<0) continue;

      float u = pKF->fx*xc*invzc+pKF->cx;
      float v = pKF->fy*yc*invzc+pKF->cy;

      if(u<pKF->mnMinX || u>pKF->mnMaxX || v<pKF->mnMinY || v>pKF->mnMaxY)
         continue;
      
      int fid = it->second;
      Point2d kp = pKF->mvKeysUn[fid].pt;
      float error = sqrt((u-kp.x)*(u-kp.x) + (v-kp.y)*(v-kp.y));
      errori += error;
    }
    
    // error in current frame=============================================================================
    /*
    Mat Rcw = mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    Mat tcw = mCurrentFrame.mTcw.rowRange(0,3).col(3);
    Mat x3Dc = Rcw*x3Dw+tcw;
    float xc = x3Dc.at<float>(0);
    float yc = x3Dc.at<float>(1);
    float invzc = 1.0/x3Dc.at<float>(2);

    if(invzc<0) continue;

    float u = mCurrentFrame.fx*xc*invzc+mCurrentFrame.cx;
    float v = mCurrentFrame.fy*yc*invzc+mCurrentFrame.cy;

    if(u<mCurrentFrame.mnMinX || u>mCurrentFrame.mnMaxX || v<mCurrentFrame.mnMinY || v>mCurrentFrame.mnMaxY)
      continue;
    Point2d kp = mCurrentFrame.mvKeysUn[i].pt;
    
    float error = (u-kp.x)*(u-kp.x) + (v-kp.y)*(v-kp.y);
    
    errori += error;
    */
    // error in current frame=============================================================================
    
    errori /= observations.size();
    float ratio = errori / observations.size();
    //cout<<"observations: "<<observations.size()<<",  errori: "<<errori<<", error/observations: "<<errori/observations.size()<<
    //" ,depth:"<<x3Dw.at<float>(2)<<endl;
    
    //vobservs.push_back(observations.size()); vdepths.push_back(x3Dw.at<float>(2)); verrors.push_back(errori);
    vx.push_back(x3Dw.at<float>(0));
    vy.push_back(x3Dw.at<float>(1));
    vz.push_back(x3Dw.at<float>(2));
    
    if(errori < minErr) minErr = errori;
    else if(maxErr < errori) maxErr = errori;
    //mvReprojectErrors[i] = errori;
    if(ratio > 0.4)
      mvReprojectErrors[i] = ratio;
    else
      mvReprojectErrors[i] = 0;
  }
  
  cout<<"x coord:"<<endl;
  for(int i = 0; i < vx.size(); i++)
 	cout<<vx[i]<<",";
  cout<<endl;
  
  cout<<"y coord:"<<endl;
  for(int i = 0; i < vx.size(); i++)
 	cout<<vy[i]<<",";
  cout<<endl;
  
  cout<<"z coord:"<<endl;
  for(int i = 0; i < vx.size(); i++)
 	cout<<vz[i]<<",";
  cout<<endl;
  
  //cout<<"maxErr:"<<maxErr<<endl;
  //for(int i = 0; i < mCurrentFrame.N; i++)
  //  mvReprojectErrors[i] /= maxErr;
}

void Tracking::ComputeMapPointsErrAndObserv(MapPoint* pMP, int& observs, float& error, float& depth){
    if(!pMP || pMP->isBad()) return;
    map<KeyFrame*,size_t> observations = pMP->GetObservations();
    observs = observations.size();
    if(observs == 0) return;
    
    Mat x3Dw = pMP->GetWorldPos();
    for(map<KeyFrame*,size_t>::iterator it = observations.begin(); it != observations.end(); it++){
      KeyFrame* pKF = it->first;
      Mat Rcw = pKF->GetPose().rowRange(0,3).colRange(0,3);
      Mat tcw = pKF->GetPose().rowRange(0,3).col(3);
      Mat x3Dc = Rcw*x3Dw+tcw;
      float xc = x3Dc.at<float>(0);
      float yc = x3Dc.at<float>(1);
      float invzc = 1.0/x3Dc.at<float>(2);
      
      if(invzc<0) continue;
      float u = pKF->fx*xc*invzc+pKF->cx;
      float v = pKF->fy*yc*invzc+pKF->cy;

      if(u<pKF->mnMinX || u>pKF->mnMaxX || v<pKF->mnMinY || v>pKF->mnMaxY)
         continue;
      
      int fid = it->second;
      Point2d kp = pKF->mvKeysUn[fid].pt;
      float error_ = sqrt((u-kp.x)*(u-kp.x) + (v-kp.y)*(v-kp.y));
      error += error_;
      depth += x3Dc.at<float>(2);
    }
    error /= observs;
    depth /= observs;
}

void Tracking::DrawReprojectionError(){
    Scalar color(0,0,255);
    Mat out_image = mCurrentFrame.mImag.clone();
    for (int i = 0; i < mCurrentFrame.N; i++){
      if(mvReprojectErrors[i] != 0){
	int r = mvReprojectErrors[i] * 15;
	Point2d pt = mCurrentFrame.mvKeysUn[i].pt;
	circle(out_image, pt, r, color, -1);  
      }
    }
    showImage(out_image, "PROJECTION-ERROR",900,675,true);
}

void Tracking::CrfSegmentMapPoints(map<int,int>& maps){
    vector<VMP> vmps;
    vector<Point3f> vpoints;
    vector<float> vobservs,vdepths, verrors;
    vector<Point2f> vcorrd2d;
    for(int i = 0; i < mCurrentFrame.N; i++){
      MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
      if(!pMP) 
	continue;
      int observs = 0;
      float error = 0, depth = 0;

      ComputeMapPointsErrAndObserv(pMP,observs,error,depth);
      if(observs == 0) 
	continue;
      
      vmps.push_back({i,pMP});
      Point2f point2d = (Point2f)mCurrentFrame.mvKeysUn[i].pt;
      vcorrd2d.push_back(point2d);

      Mat point3d = pMP->GetWorldPos();
      vpoints.push_back(Point3f(point3d.at<float>(0),point3d.at<float>(1),point3d.at<float>(2)));
      vobservs.push_back(observs);
      vdepths.push_back(depth);
      verrors.push_back(error);
    }

    short* label = RroughClassify(vobservs,verrors,vdepths);
    for(int i = 0; i < vmps.size(); i++){
      if(mStaticFids[i])
	label[i] = 1;
    }
    
    
    int N = vmps.size();
    //draw init label
    
    Mat out_image = mCurrentFrame.mImag.clone();
    int r = 6;
    Scalar color1(0,0,255);
    Scalar color2(0,255,0);
    for (int i = 0; i < N; i++){
      int fid = vmps[i].fid;
      Point2d pt = mCurrentFrame.mvKeysUn[fid].pt;
      if(label[i] == 0)
	circle(out_image, pt, r, color1, -1);
      else
	circle(out_image, pt, r, color2, -1);
    }
    //showImage(out_image, "BEFORE-CRF-SEGMENT",900,675,true);


    const int M = 2;
    DenseCRF3D<M> crf(N);
    crf.setUnaryEnergyFromLabel(label, mConfidence);
    
    /*
    for(int i = 0; i < vmps.size(); i++){
      if(mStaticFids[i] == 1)
	crf.SetUnaryEnergtForPositiveNode(i,1,0.2);
    }*/

    auto* appearancePairwise = PottsPotential3D<M, 2>::appearanceKernel(N, mW1, vobservs, verrors, mObserv_stdev,  mError_stdev);
    crf.addPairwiseEnergy(appearancePairwise);

    auto* smoothnessPairwise = PottsPotential3D<M, 2>::smoothKernel(N, mW2, vpoints, vcorrd2d, mPoint_stdev,mCoord2d_stdev);
    crf.addPairwiseEnergy(smoothnessPairwise);

    crf.inference(5, true);
    short * map_ = crf.getMap();

    Mat out_image1 = mCurrentFrame.mImag.clone();
    for(int i = 0; i < N; i++){
      int fid = vmps[i].fid;
      Point2d pt = mCurrentFrame.mvKeysUn[fid].pt;
      if(map_[i] == 0)
	circle(out_image1, pt, r, color1, -1);  
      else
	circle(out_image1, pt, r, color2, -1);  
    }
    showImage(out_image1, "AFTER-CRF-SEGMENT",900,675,true);
    
    
    Mat out_image2 = mCurrentFrame.mImag.clone();
    for(int i = 0; i < N; i++){
      int fid = vmps[i].fid;
      if(map_[i] == 0){
	map<int,int>::iterator it = maps.find(fid);
	maps.erase(it);
	vmps[i].pMP->SetBadFlag();
	mCurrentFrame.mvpMapPoints[fid] = static_cast<MapPoint*>(NULL);
      }
    }
    
   /*
   vector<int> flag(mCurrentFrame.N,0);
   for(int i = 0; i < N; i++){
      int fid = vmps[i].fid;
      if(map_[i] == 1)
	flag[fid] = 1;
   }
   for(int i = 0; i < mCurrentFrame.N; i++){
     if(flag[i] == 0){
       mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
       mCurrentFrame.mvpMapPoints[i]->SetBadFlag();
     }
   }
   */   
    
   // Mat points = makeMatchedFeartures2Mat(mCurrentFrame,mLastFrame,maps);
   // DrawMatchedFeatures(points,mCurrentFrame.mImag,mLastFrame.mImag,"MATCH-AFTER-CRF-SEGMENT");
    
    
    delete[] label;
    label = nullptr;
}

short* Tracking::RroughClassify(vector<float>& vobservs, vector<float>& verrors, vector<float>& depths){
    short* res = new short[vobservs.size()];
    float observ_sigma2 = mObserv_stdev*mObserv_stdev;
    float error_sigma2 = mError_stdev*mError_stdev;
    float depth_sigma2 = mPoint_stdev*mPoint_stdev;

    /*
    for(int i = 0; i < vobservs.size(); i++){
    float _logp1 = (vobservs[i]-observ_u)*(vobservs[i]-observ_u)/2*observ_sigma2;
    float _logp2 = (verrors[i]-error_u)*(verrors[i]-error_u)/2*error_sigma2;
    float _logp3 = (depths[i]-depth_u)*(depths[i]-depth_u)/2*depth_sigma2;
    float  sum_logp = _logp1 + _logp2 + _logp3;
    float p1 = 1.0 / (1+exp(-sum_logp));
    cout<<"sum_logp:"<<sum_logp<<", p1:"<<p1<<endl;
    if(p1 > 0.5) 
      res[i] = 1;
    else
      res[i] = 2;
    }
    */

    for(int i = 0; i < vobservs.size(); i++){
      float k1 = (vobservs[i]-mObserv_u)*(vobservs[i]-mObserv_u)/(2*observ_sigma2);
      float k2 = (verrors[i]-mError_u)*(verrors[i]-mError_u)/(2*error_sigma2);
      float k3 = (depths[i]-mDepth_u)*(depths[i]-mDepth_u)/(2*depth_sigma2);
      float p1 = exp(-k1), p2 = exp(-k2), p3 = exp(-k3);
      if(p1+p2+p3 <= mPth) 
	res[i] = 0;  //moving
      else
	res[i] = 1;  //static
    }
    return res;
}

 bool Tracking::MatchCurrFrame2KeyFrame(){
    mCurrentFrame.ComputeBoW();
    ORBmatcher matcher(0.8,true);
    vector<MapPoint*> vpMapPointMatches;
    matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);
    
    Mat points = makeMatchedFeartures2Mat(mCurrentFrame,mpReferenceKF,matcher.mCurr2LastFId);
    DrawMatchedFeatures(points,mCurrentFrame.mImag,mpReferenceKF->mImag,"Curr2RefKeyFrame");
    
    cout<<mCurrentFrame.mnId<<"   "<<mCurrentFrame.N<<endl;

    vector<int> inliers = GraphCutRansacFilter(points);
    vector<int> pointsFids;
    for(map<int,int>::iterator it = matcher.mCurr2LastFId.begin(); it != matcher.mCurr2LastFId.end(); it++)
	pointsFids.push_back(it->first);

    mbInliers = vector<bool>(mCurrentFrame.N, false);
    for(int i = 0; i < inliers.size(); ++i){
	  int idx = inliers[i];
	  int fid = pointsFids[idx];
	  mbInliers[fid] = true;
    }
    DrawMatchedFeatures(points,mCurrentFrame.mImag,mpReferenceKF->mImag, inliers,"GC-Curr2RefKeyFrame");
 }
 
  bool Tracking::MatchCurrent2NFrame(Frame& frame){
    return true;
  }

//==============================add by dzj 0825======================================================
} //namespace ORB_SLAM