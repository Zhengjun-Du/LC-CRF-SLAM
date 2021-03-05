/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Tracking.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "Map.h"
#include "Initializer.h"
#include "Optimizer.h"
#include "PnPsolver.h"
#include <iostream>
#include <mutex>
#include "GCRANSAC.h"
#include "grid_neighborhood_graph.h"
#include "utils.h"
#include "graph.h"
#include "densecrf3d.h"
#include "pairwise3d.h"
#include <unistd.h> 

using namespace DenseCRF;
using namespace std;
using namespace cv;

//ofstream file_("dynapoints_cnt.txt");

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase *pKFDB, const string &strSettingPath, const int sensor) : mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
                                                                                                                                                                                              mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer *>(NULL)), mpSystem(pSys), mpViewer(NULL),                                                                                                                                                                                             mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if (fps == 0)
        fps = 30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl
         << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if (DistCoef.rows == 5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if (mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    if (sensor == System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    if (sensor == System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    cout << endl
         << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if (sensor == System::STEREO || sensor == System::RGBD)
    {
        mThDepth = mbf * (float)fSettings["ThDepth"] / fx;
        cout << endl
             << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if (sensor == System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if (fabs(mDepthMapFactor) < 1e-5)
            mDepthMapFactor = 1;
        else
            mDepthMapFactor = 1.0f / mDepthMapFactor;
    }

    //LCCRF: read crf-prediction parameters from yaml file====================================
    mConf = fSettings["confidence"];
    
    mW1 = fSettings["w1"];
    mW2 = fSettings["w2"];
   
    mRpjErrorMean = fSettings["u_alpha"];
    mRpjErrorStdev = fSettings["stdev_alpha"];
    
    mObservMean = fSettings["u_beta"];
    mObservStdev = fSettings["stdev_beta"];
    
    mGcMean = fSettings["u_gamma"];
    mGcStdev = fSettings["stdev_gamma"];
    
    mPoint3dStdev = fSettings["point3d_stdev"];
    mPoint2dStdev = fSettings["point2d_stdev"];

    mDepthMean = fSettings["u_depth"];
    mPth = fSettings["pth"]; 
    //LCCRF: read crf-prediction parameters from yaml file====================================
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing = pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer = pViewer;
}

cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if (mImGray.channels() == 3)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
        }
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if (mImGray.channels() == 3)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
    }

    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
    }

    if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
        imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

    mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
    cvtColor(mImGray, mImGray, CV_GRAY2BGR);
    mCurrentFrame.mImag = mImGray;

    //LCCRF: added codes============================================================================
    mvStaticFeatures.clear();
    mvFeatureMatchDis.clear();
    mvFeatureMatchProb.clear();

    mImagesQue.push(mImGray);
    mFrameQue.push(mCurrentFrame);

    mStaticFids = vector<int>(mCurrentFrame.N, 0);

    //LCCRF: feature matching on frame i and frame i-15, and get the static feature points in current frame
    if (mFrameQue.size() >= 15)
    {
        Frame refFrame = mFrameQue.front();
        refImg = mImagesQue.front();

        map<int, int> asso = BfMatch(mCurrentFrame, refFrame);

        //LCCRF: since we perform feature match on two distant frame, so the matched feature points can be
        //LCCRF: regard as static points
        for (map<int, int>::iterator it = asso.begin(); it != asso.end(); it++)
            mvStaticFeatures.push_back(it->first);
        
        //LCCRF: draw the matched feature points between current and reference frame
        /*
        Mat points = makeMatchedFeartures2Mat(mCurrentFrame, refFrame, asso);
        int fid = mCurrentFrame.mnId;
        string sid = to_string(fid);
        sid = "imgs/"+sid+"-1.png";
        DrawMatchedFeatures(points,mImGray,refImg,sid);//BF-MATCH
        */

        mFrameQue.pop();
        mImagesQue.pop();
    }
    //LCCRF: added codes============================================================================

    Track();
    return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    if (mImGray.channels() == 3)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
    }

    if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
    else
        mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
    if (mState == NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState = mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if (mState == NOT_INITIALIZED)
    {
        if (mSensor == System::STEREO || mSensor == System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();

        mpFrameDrawer->Update(this);

        if (mState != OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if (!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if (mState == OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame

                CheckReplacedInLastFrame();
                //bOK = TrackReferenceKeyFrame();

                if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                {
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    bOK = TrackWithMotionModel();
                    if (!bOK)
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else
            {
                bOK = Relocalization();
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated

            if (mState == LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                if (!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    if (!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint *> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if (!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if (bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if (mbVO)
                        {
                            for (int i = 0; i < mCurrentFrame.N; i++)
                            {
                                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if (bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.

        if (!mbOnlyTracking)
        {
            if (bOK)
                bOK = TrackLocalMap();
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if (bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if (bOK)
            mState = OK;
        else
            mState = LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if (bOK)
        {
            // Update motion model
            if (!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                mVelocity = mCurrentFrame.mTcw * LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (pMP)
                    if (pMP->Observations() < 1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end(); lit != lend; lit++)
            {
                MapPoint *pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if (NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if (mState == LOST)
        {
            if (mpMap->KeyFramesInMap() <= 5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if (!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
        mLastFrame.mImag = mCurrentFrame.mImag.clone();
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if (!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState == LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState == LOST);
    }
}

void Tracking::StereoInitialization()
{
    if (mCurrentFrame.N > 500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

        // Create KeyFrame
        KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        //LCCRF: add
        pKFini->mImag = mImGray;

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap);
                pNewMP->AddObservation(pKFini, i);
                pKFini->AddMapPoint(pNewMP, i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i] = pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState = OK;
    }
}

void Tracking::MonocularInitialization()
{
    if (!mpInitializer)
    {
        // Set Reference Frame
        if (mCurrentFrame.mvKeys.size() > 100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

            if (mpInitializer)
                delete mpInitializer;

            mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

            return;
        }
    }
    else
    {
        // Try to initialize
        if ((int)mCurrentFrame.mvKeys.size() <= 100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9, true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

        // Check if there are enough correspondences
        if (nmatches < 100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
            return;
        }

        cv::Mat Rcw;                 // Current Camera Rotation
        cv::Mat tcw;                 // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
            {
                if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                {
                    mvIniMatches[i] = -1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
            Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
            tcw.copyTo(Tcw.rowRange(0, 3).col(3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
    KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for (size_t i = 0; i < mvIniMatches.size(); i++)
    {
        if (mvIniMatches[i] < 0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);

        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

    // Set median depth to 1map<int,int>
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f / medianDepth;

    if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
    for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
    {
        if (vpAllMapPoints[iMP])
        {
            MapPoint *pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState = OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for (int i = 0; i < mLastFrame.N; i++)
    {
        MapPoint *pMP = mLastFrame.mvpMapPoints[i];

        if (pMP)
        {
            MapPoint *pRep = pMP->GetReplaced();
            if (pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7, true);
    vector<MapPoint *> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    if (nmatches < 15)
    {   
        cout << "TrackReferenceKeyFrame 1: " << nmatches << endl;
        return false;
    }

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }

    cout << "TrackReferenceKeyFrame 2: " << nmatchesMap << endl;
    return nmatchesMap >= 10; 
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame *pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr * pRef->GetPose());

    if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float, int>> vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for (int i = 0; i < mLastFrame.N; i++)
    {
        float z = mLastFrame.mvDepth[i];
        if (z > 0)
        {
            vDepthIdx.push_back(make_pair(z, i));
        }
    }

    if (vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(), vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for (size_t j = 0; j < vDepthIdx.size(); j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint *pMP = mLastFrame.mvpMapPoints[i];
        if (!pMP)
            bCreateNew = true;
        else if (pMP->Observations() < 1)
        {
            bCreateNew = true;
        }

        if (bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint *pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);

            mLastFrame.mvpMapPoints[i] = pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if (vDepthIdx[j].first > mThDepth && nPoints > 100)
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
    UpdateLastFrame();
    ORBmatcher matcher(0.9, true);
    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

    //LCCRF: get the correspondence of last frame's map points and current frame's feature points
    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
    int th = (mSensor != System::STEREO) ? 15 : 10;
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);
    
    //LCCRF: if the number of association < 10, increase the th and refinding
    if (nmatches < 10)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR);
    }
    if (nmatches < 6)
        return false;

    Optimizer::PoseOptimization(&mCurrentFrame);

    //Mat points1 = makeMatchedFeartures2Mat(mCurrentFrame,mLastFrame,matcher.mCurr2LastFId);
    //DrawMatchedFeatures(points1,mCurrentFrame.mImag,mLastFrame.mImag,"CURR2LAST-1");

    //LCCRF: 1.initial camera pose estimation=========================================================
    //mCurr2LastFId: matched feature points between last frame and current frame
    bool hasGotInitPose = false;
    StaticFeatMatch.clear();
    if (!mvStaticFeatures.empty() && mvStaticFeatures.size() > 20)
    {
        //LCCRF: extract those static points from mCurr2LastFId
        StaticFeatMatch = GetStaticFeatureMatch(matcher.mCurr2LastFId);
        Mat points2 = makeMatchedFeartures2Mat(mCurrentFrame, mLastFrame, StaticFeatMatch);

        //LCCRF: draw the static matched feature points
        /*
        int fid = mCurrentFrame.mnId;
        string sid = to_string(fid);
        sid = "imgs/"+sid+"-2.png";
        DrawMatchedFeatures(points2,mCurrentFrame.mImag,mLastFrame.mImag,sid);//"CURR2LAST-STATIC");
        */

        //LCCRF: compute the initial camera pose
        Mat pose = GetCurrFrameInitPose(StaticFeatMatch);
        mCurrentFrame.SetPose(pose);
        hasGotInitPose = true;

        //LCCRF: compute the fundmental matrix using graph-cut ransac and prior
        if (points2.rows > 14)
        {
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            FundamentalMatrix FundModel = GraphCutRansacFilter(points2);
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            
            if (FundModel.descriptor.data())
            {
                GetFeature2EpipolarDis(FundModel, matcher.mCurr2LastFId);
            }
        }
    }
    if (!hasGotInitPose)
        Optimizer::PoseOptimization(&mCurrentFrame);

    //2.LCCRF: CRF dynamic/static segmantation and pose optimization===============================
    if (mCurrentFrame.mnId > 3)
    {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        DynamicDetectionWithCRF(matcher.mCurr2LastFId);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        Optimizer::PoseOptimization(&mCurrentFrame);
    }

    //LCCRF: discard the dynamic landmarks 
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }

    if (mbOnlyTracking)
    {
        mbVO = nmatchesMap < 10;
        return nmatches > 20;
    }
    

    return nmatchesMap >= 10;
}

bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap();

    SearchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if (!mbOnlyTracking)
                {
                    if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if (mSensor == System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently

    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
    {
        cout << "lost 1: " << mnMatchesInliers << endl;
        return false;
    }

    if (mnMatchesInliers < 30)
    {
        cout << "lost 2: " << mnMatchesInliers << endl;
        return false;
    }
    else
        return true;
}

bool Tracking::NeedNewKeyFrame()
{
    if (mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if (nKFs <= 2)
        nMinObs = 2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose = 0;
    if (mSensor != System::MONOCULAR)
    {
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth)
            {
                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

    // Thresholds
    float thRefRatio = 0.75f;
    if (nKFs < 2)
        thRefRatio = 0.4f;

    if (mSensor == System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c = mSensor != System::MONOCULAR && (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) && mnMatchesInliers > 15);

    if ((c1a || c1b || c1c) && c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if (bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if (mSensor != System::MONOCULAR)
            {
                if (mpLocalMapper->KeyframesInQueue() < 3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if (!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    pKF->mImag = mImGray;

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if (mSensor != System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float, int>> vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0)
            {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        if (!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(), vDepthIdx.end());

            int nPoints = 0;
            for (size_t j = 0; j < vDepthIdx.size(); j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (!pMP)
                    bCreateNew = true;
                else if (pMP->Observations() < 1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }

                if (bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap);
                    pNewMP->AddObservation(pKF, i);
                    pKF->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        if (pMP)
        {
            if (pMP->isBad())
            {
                *vit = static_cast<MapPoint *>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch = 0;

    // Project points in frame and check its visibility
    for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if (pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if (mCurrentFrame.isInFrustum(pMP, 0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if (nToMatch > 0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if (mSensor == System::RGBD)
            th = 3;
        // If the camera has been relocalised recently, perform a coarser search
        if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
            th = 5;
        matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        KeyFrame *pKF = *itKF;
        const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

        for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
        {
            MapPoint *pMP = *itMP;
            if (!pMP)
                continue;
            if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                continue;
            if (!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame *, int> keyframeCounter;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
            if (!pMP->isBad())
            {
                const map<KeyFrame *, size_t> observations = pMP->GetObservations();
                for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i] = NULL;
            }
        }
    }
    if (keyframeCounter.empty())
        return;

    int max = 0;
    KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
    {
        KeyFrame *pKF = it->first;

        if (pKF->isBad())
            continue;

        if (it->second > max)
        {
            max = it->second;
            pKFmax = pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);

        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if (mvpLocalKeyFrames.size() > 80)
            break;

        KeyFrame *pKF = *itKF;

        const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
        {
            KeyFrame *pNeighKF = *itNeighKF;
            if (!pNeighKF->isBad())
            {
                if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame *> spChilds = pKF->GetChilds();
        for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
        {
            KeyFrame *pChildKF = *sit;
            if (!pChildKF->isBad())
            {
                if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame *pParent = pKF->GetParent();
        if (pParent)
        {
            if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                break;
            }
        }
    }
    if (pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if (vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75, true);

    vector<PnPsolver *> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint *>> vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates = 0;

    for (int i = 0; i < nKFs; i++)
    {
        KeyFrame *pKF = vpCandidateKFs[i];
        if (pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
            if (nmatches < 15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver *pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9, true);
    int x = 0;

    while (nCandidates > 0 && !bMatch)
    {
        for (int i = 0; i < nKFs; i++)
        {
            if (vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver *pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if (bNoMore)
            {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if (!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint *> sFound;

                const int np = vbInliers.size();

                for (int j = 0; j < np; j++)
                {
                    if (vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j] = NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if (nGood < 10)
                    continue;

                for (int io = 0; io < mCurrentFrame.N; io++)
                    if (mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if (nGood < 50)
                {
                    int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);
                    if (nadditional + nGood >= 50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if (nGood > 30 && nGood < 50)
                        {
                            sFound.clear();
                            for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                if (mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

                            // Final optimization
                            if (nGood + nadditional >= 50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for (int io = 0; io < mCurrentFrame.N; io++)
                                    if (mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io] = NULL;
                            }
                        }
                    }
                }

                // If the pose is supported by enough inliers stop ransacs and continue

                if (nGood >= 50)
                {
                    x = nGood;
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if (!bMatch)
    {
        //cout<<"nGood:"<<x<<endl;
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }
}

void Tracking::Reset()
{
    cout << "System Reseting" << endl;
    if (mpViewer)
    {
        mpViewer->RequestStop();
        while (!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if (mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer *>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if (mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

FundamentalMatrix Tracking::GraphCutRansacFilter(cv::Mat &points)
{
    int cell_number_in_neighborhood_graph_ = 16; //8
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

    gcransac.setFPS(-1);                                                               // Set the desired FPS (-1 means no limit)
    gcransac.settings.threshold = 0.12;                                                //1.2  2 The inlier-outlier threshold
    gcransac.settings.spatial_coherence_weight = 0.14;                                 //0.14 The weight of the spatial coherence term
    gcransac.settings.confidence = 0.95f;                                              // The required confidence in the results
    gcransac.settings.max_local_optimization_number = 50;                              // The maximm number of local optimizations
    gcransac.settings.max_iteration_number = 5000;                                     // The maximum number of iterations
    gcransac.settings.min_iteration_number = 100;                                      //50 The minimum number of iterations
    gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball
    gcransac.settings.core_number = 8;                                                 // The number of parallel processes

    gcransac.run(points, estimator, &neighborhood, model);
    return model;
}

void Tracking::DrawMatchedFeatures(Mat &points, Mat &imgl, Mat &imgr, string name)
{
    Mat match_image;
    vector<int> inliers(points.rows);
    for (int i = 0; i < inliers.size(); i++)
        inliers[i] = i;
    drawMatches(points, inliers, imgl, imgr, match_image);
    showImage(match_image, name, 1280, 480, true);
   // cv::imwrite(name,match_image);
}

void Tracking::DrawMatchedFeatures(Mat &points, Mat &imgl, Mat &imgr, vector<int> &inliers, string name)
{
    Mat match_image;
    drawMatches(points, inliers, imgl, imgr, match_image);
    showImage(match_image, name, 900, 675, true);
}

//LCCRF: feature match on frame f1 and frame f2 in a brute force way
map<int, int> Tracking::BfMatch(Frame &fl, Frame &fr)
{
    vector<tuple<double, int, int>> correspondences;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    vector<vector<DMatch>> matches_vector;
    matcher.knnMatch(fl.mDescriptors, fr.mDescriptors, matches_vector, 2);
    for (auto match : matches_vector)
    {
        if (match.size() == 2 && match[0].distance < match[1].distance * 0.6)
            correspondences.emplace_back(make_tuple<double, int, int>(match[0].distance / match[1].distance, (int)match[0].queryIdx, (int)match[0].trainIdx));
    }
    map<int, int> asso;
    for (int i = 0; i < correspondences.size(); i++)
    {
        int fid1 = get<1>(correspondences[i]);
        int fid2 = get<2>(correspondences[i]);
        asso[fid1] = fid2;
    }
    return asso;
}

Mat Tracking::makeMatchedFeartures2Mat(Frame &fl, Frame &fr, vector<pair<int, int>> &pairs)
{
    Mat points = Mat(static_cast<int>(pairs.size()), 4, CV_64F);
    double *points_ptr = reinterpret_cast<double *>(points.data);
    for (int i = 0; i < pairs.size(); i++)
    {
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

Mat Tracking::makeMatchedFeartures2Mat(Frame &fl, Frame &fr, map<int, int> &maps)
{
    Mat points = Mat(static_cast<int>(maps.size()), 4, CV_64F);
    double *points_ptr = reinterpret_cast<double *>(points.data);
    for (map<int, int>::iterator it = maps.begin(); it != maps.end(); it++)
    {
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


void Tracking::ComputeMapPointErrAndObserv(MapPoint *pMP, int &observs, float &error, float &depth)
{
    if (!pMP || pMP->isBad())
        return;
    map<KeyFrame *, size_t> observations = pMP->GetObservations();
    observs = observations.size();
    if (observs == 0)
        return;

    Mat x3Dw = pMP->GetWorldPos();
    for (map<KeyFrame *, size_t>::iterator it = observations.begin(); it != observations.end(); it++)
    {
        KeyFrame *pKF = it->first;
        Mat Rcw = pKF->GetPose().rowRange(0, 3).colRange(0, 3);
        Mat tcw = pKF->GetPose().rowRange(0, 3).col(3);
        Mat x3Dc = Rcw * x3Dw + tcw;
        float xc = x3Dc.at<float>(0);
        float yc = x3Dc.at<float>(1);
        float invzc = 1.0 / x3Dc.at<float>(2);

        if (invzc < 0)
            continue;
        float u = pKF->fx * xc * invzc + pKF->cx;
        float v = pKF->fy * yc * invzc + pKF->cy;

        if (u < pKF->mnMinX || u > pKF->mnMaxX || v < pKF->mnMinY || v > pKF->mnMaxY)
            continue;

        int fid = it->second;
        Point2d kp = pKF->mvKeysUn[fid].pt;
        float error_ = sqrt((u - kp.x) * (u - kp.x) + (v - kp.y) * (v - kp.y));
        error += error_;
        depth += x3Dc.at<float>(2);
    }
    error /= observs;
    depth /= observs;
}

void Tracking::DynamicDetectionWithCRF(map<int, int> &maps)
{
    vector<FeatureMapAsso> featureMapAssos;
    vector<Point3f> vpoints;
    vector<float> vobservs, vdepths, verrors;
    vector<Point2f> vcorrd2d;

    //LCCRF: initial label the feature points with Unary potential 
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
        if (!pMP)
            continue;
        int observs = 0;
        float error = 0, depth = 0;

        ComputeMapPointErrAndObserv(pMP, observs, error, depth);
        if (observs == 0)
            continue;

        featureMapAssos.push_back({i, pMP});
        Point2f point2d = (Point2f)mCurrentFrame.mvKeysUn[i].pt;
        vcorrd2d.push_back(point2d);

        Mat point3d = pMP->GetWorldPos();
        vpoints.push_back(Point3f(point3d.at<float>(0), point3d.at<float>(1), point3d.at<float>(2)));
        vobservs.push_back(observs);
        vdepths.push_back(depth);
        verrors.push_back(error);
    }
    short *init_label = RroughClassify(vobservs, verrors, vdepths, featureMapAssos);

    /*
    for(int i = 0; i < featureMapAssos.size(); i++){
        if(mStaticFids[i])
	        init_label[i] = 1;
    }
    
    for(int i = 0; i < featureMapAssos.size(); i++){
        int fid = featureMapAssos[i].fid;
        if(StaticFeatMatch.find(fid) != StaticFeatMatch.end())
	    init_label[i] = 1;
      
        if(!mvFeatureMatchProb.empty()){
            double dis = mvFeatureMatchDis[fid];
            double p4 = exp(- (dis-1.2)*(dis-1.2)/(2*1.1*1.1));
       
            if(dis >= 10){
	            init_label[i] = 0;
            }
        } 
    }
    */

    int N = featureMapAssos.size();

    int r = 4;
    Scalar red_(0, 0, 255);
    Scalar green_(0, 255, 0);

    //LCCRF: draw the initial lable of feature points labeled by RroughClassify
    /*
    Mat out_image = mCurrentFrame.mImag.clone();
    for (int i = 0; i < N; i++)
    {
        int fid = featureMapAssos[i].fid;
        Point2d pt = mCurrentFrame.mvKeysUn[fid].pt;
        if (init_label[i] == 0)
            circle(out_image, pt, r, red_, -1);
        else
            circle(out_image, pt, r, green_, -1);
    }
    showImage(out_image, "Initial Labeling", 900, 675, false);
    waitKey();
    */

    //LCCRF: further label the feature points with Pairwise potential
    int fid = mCurrentFrame.mnId;
    const int M = 2;
    DenseCRF3D<M> crf(N);
    crf.setUnaryEnergyFromLabel(init_label, mConf);

    auto *appearancePairwise = PottsPotential3D<M, 2>::appearanceKernel(N, mW1, vobservs, verrors, mObservStdev, mRpjErrorStdev);
    crf.addPairwiseEnergy(appearancePairwise);

    auto *smoothnessPairwise = PottsPotential3D<M, 2>::smoothKernel(N, mW2, vpoints, vcorrd2d, mPoint3dStdev, mPoint2dStdev);
    crf.addPairwiseEnergy(smoothnessPairwise);

    crf.inference(5, true);
    short *res_label = crf.getMap();

    Mat out_image1 = mCurrentFrame.mImag.clone();
    for (int i = 0; i < N; i++)
    {
        int id = featureMapAssos[i].fid;
        Point2d pt = mCurrentFrame.mvKeysUn[id].pt;
        if (res_label[i] == 0)
            circle(out_image1, pt, r, red_, -1);
        else
            circle(out_image1, pt, r, green_, -1);
    }
    showImage(out_image1, "Accurate Labeling", 900, 675, true);

    //LCCRF: mark the dynamic map points as BadFlag, which will be removed from the point cloud
    for (int i = 0; i < N; i++)
    {
        int fid = featureMapAssos[i].fid;
        if (res_label[i] == 0)
        {
            map<int, int>::iterator it = maps.find(fid);
            maps.erase(it);
            featureMapAssos[i].pMP->SetBadFlag();
            mCurrentFrame.mvpMapPoints[fid] = static_cast<MapPoint *>(NULL);
        }
    }
    delete[] init_label;
    init_label = nullptr;
}

//LCCRF: initial lebeling by unary potential of CRF 
short *Tracking::RroughClassify(vector<float> &vobservs, vector<float> &verrors, vector<float> &depths, vector<FeatureMapAsso> &vmp)
{
    short *init_label = new short[vobservs.size()];
    float observ_sigma2 = mObservStdev * mObservStdev;
    float rpjerror_sigma2 = mRpjErrorStdev * mRpjErrorStdev;
    float depth_sigma2 = mPoint3dStdev * mPoint3dStdev;
    float mprior_u = mGcMean, prior_sigma = mGcStdev;
    float prior_sigma2 = prior_sigma * prior_sigma;

    for (int i = 0; i < vobservs.size(); i++)
    {
        float k1 = (vobservs[i] - mObservMean) * (vobservs[i] - mObservMean) / (2 * observ_sigma2);
        float k2 = (verrors[i] - mRpjErrorMean) * (verrors[i] - mRpjErrorMean) / (2 * rpjerror_sigma2);
        float k3 = (depths[i] - mDepthMean) * (depths[i] - mDepthMean) / (2 * depth_sigma2);
        float p1 = exp(-k1), p2 = exp(-k2), p3 = exp(-k3);
        int fid = vmp[i].fid;

        /*
        if(mvFeatureMatchProb.empty()){
	        if(p1+p2 <= 0.36)
	            init_label[i] = 0;  //moving
	        else
	            init_label[i] = 1;  //static
        }
        else{
	        double p4 = mvFeatureMatchProb[fid]; 
	        if(p1+p2+p4 <= 0.6)
	            init_label[i] = 0;  //moving
	        else
	            init_label[i] = 1;  //static
        } 
        */

        if (mvFeatureMatchProb.empty())
        {
            if (p1 + p2 + p3 <= mPth)
                init_label[i] = 0; //moving
            else
                init_label[i] = 1; //static
        }
        else
        {
            double p4 = mvFeatureMatchProb[fid];
            if (p1 + p2 + p3 + p4 <= mPth + 0.2){
                init_label[i] = 0; //moving
            }
            else{
                init_label[i] = 1; //static
            }
        }
    }
    return init_label;
}

//LCCRF: mvStaticFeatures: static feature points generated by feature matching(frame i and frame i-15)
//LCCRF: reference the function GrabImageRGBD
map<int, int> Tracking::GetStaticFeatureMatch(map<int, int> &asso)
{
    map<int, int> staticFeaMatch;
    for (map<int, int>::iterator it = asso.begin(); it != asso.end(); it++)
    {
        int currFid = it->first, lastFid = it->second;
        if (find(mvStaticFeatures.begin(), mvStaticFeatures.end(), currFid) != mvStaticFeatures.end())
            staticFeaMatch[currFid] = lastFid;
    }
    return staticFeaMatch;
}

//LCCRF: compue the symmetric epipolar distance of matched feature points and its likelihood
void Tracking::GetFeature2EpipolarDis(FundamentalMatrix &fundModel, map<int, int> &asso)
{
    int idx = 0;
    FundamentalMatrixEstimator estimator;
    for (map<int, int>::iterator it = asso.begin(); it != asso.end(); it++)
    {
        int fid1 = it->first, fid2 = it->second;
        cv::Point2d point_1 = (cv::Point2d)mCurrentFrame.mvKeysUn[fid1].pt;
        cv::Point2d point_2 = (cv::Point2d)mLastFrame.mvKeysUn[fid2].pt;
        double p[4] = {point_1.x, point_1.y, point_2.x, point_2.y};
        Mat points = Mat(1, 4, CV_64F, p);
        double dis = estimator.symmetricEpipolarDistance(points, fundModel.descriptor);

        double prob = exp(- (dis-mGcMean)*(dis-mGcMean)/(2*mGcStdev*mGcStdev));
        mvFeatureMatchDis[fid1] = dis;
        mvFeatureMatchProb[fid1] = prob;
    }
}

//LCCRF: get current frame's pose through pose optimization
Mat Tracking::GetCurrFrameInitPose(map<int, int> &StaticFeatMatch)
{
    Frame currFrameCpy = Frame(mCurrentFrame);
    fill(currFrameCpy.mvpMapPoints.begin(), currFrameCpy.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
    for (map<int, int>::iterator it = StaticFeatMatch.begin(); it != StaticFeatMatch.end(); it++)
    {
        int fid1 = it->first, fid2 = it->second;
        currFrameCpy.mvpMapPoints[fid1] = mLastFrame.mvpMapPoints[fid2];
    }
    Optimizer::PoseOptimization(&currFrameCpy);
    return currFrameCpy.mTcw;
}
} // namespace ORB_SLAM2