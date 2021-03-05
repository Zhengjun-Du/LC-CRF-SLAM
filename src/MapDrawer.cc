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

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>
#include "Converter.h"
#include <algorithm>
using namespace std;

namespace ORB_SLAM2
{
MapDrawer::MapDrawer(Map *pMap, const string &strSettingPath) : mpMap(pMap)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
    mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
    mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
    mPointSize = fSettings["Viewer.PointSize"];
    mCameraSize = fSettings["Viewer.CameraSize"];
    mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];
    drawLengthThre = fSettings["drawLengthThre"];

    // checkpoint = fSettings["checkpoint"];
    // checkfile = (string)fSettings["checkfile"];

    // if (checkpoint != -1) {
    //     FILE *fp0 = fopen(checkfile.c_str(), "w");
    //     fclose(fp0);
    // }

    bool typeMat = int(fSettings["typeMat"]);

    string file = fSettings["Tcw_gt"];
    FILE *fp = fopen(file.c_str(), "r");
    if (!fp)
    {
        return;
    }
    int cnt = 0;
    double r11, r12, r13, r14;
    double r21, r22, r23, r24;
    double r31, r32, r33, r34;
    double tp;
    int frameId;
    double tx, ty, tz;
    double qx, qy, qz, qw;

    cout << typeMat << endl;
    if (typeMat)
    {

        while (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                      &r11, &r12, &r13, &r14, &r21, &r22, &r23, &r24,
                      &r31, &r32, &r33, &r34) != EOF)
        {
            Eigen::Matrix3d R;
            R << r11, r12, r13,
                r21, r22, r23,
                r31, r32, r33;
            Eigen::Vector3d tt;
            tt << r14, r24, r34;
            Twc_gts.push_back(Converter::toCvSE3(R, tt));
            cnt++;
        }
    }
    else
    {
        double _;
        while (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf", &_, &tx,
                      &ty, &tz, &qx, &qy, &qz, &qw) != EOF)
        {
            Eigen::Quaterniond q1(qw, qx, qy, qz);
            Eigen::Matrix3d R = q1.toRotationMatrix();
            Eigen::Vector3d t;
            t << tx, ty, tz;
            Twc_gts.push_back(Converter::toCvSE3(R, t));
            cnt++;
        }
    }

    fclose(fp);
    cout << "load Twc gt: " << cnt << endl;
}

void MapDrawer::DrawMapPoints()
{
    int n = mpMap->KeyFramesInMap();
    bool check = false;
    // (checkpoint != -1 && n % checkpoint == 0 && !checks.count(n));

    FILE *fp;
    // if (check)  {
    //     fp = fopen(checkfile.c_str(), "a");
    //     fprintf(fp, "points at checkpoint %d\n", n);
    // }

    const vector<MapPoint *> &vpMPs = mpMap->GetAllMapPoints();
    const vector<MapPoint *> &vpRefMPs = mpMap->GetReferenceMapPoints();

    set<MapPoint *> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if (vpMPs.empty())
    {
        if (check)
            fclose(fp);
        return;
    }

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0, 0.0, 0.0);

    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    {
        if (vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;
        cv::Mat pos = vpMPs[i]->GetWorldPos();
        glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));

        if (check)
        {
            fprintf(fp, "%lf %lf %lf\n", pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
        }
    }
    glEnd();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0, 0.0, 0.0);

    for (set<MapPoint *>::iterator sit = spRefMPs.begin(), send = spRefMPs.end(); sit != send; sit++)
    {
        if ((*sit)->isBad())
            continue;
        cv::Mat pos = (*sit)->GetWorldPos();
        glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));

        if (check)
        {
            fprintf(fp, "%lf %lf %lf\n", pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
        }
    }

    glEnd();

    if (check)
        fclose(fp);
}

bool kfcmp(const KeyFrame *k1, const KeyFrame *k2)
{
    return k1->mnFrameId < k2->mnFrameId;
}

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
{

    int n = mpMap->KeyFramesInMap();

    bool check = false;
    // (checkpoint != -1 && n % checkpoint == 0 && !checks.count(n));

    FILE *fp;
    // if (check)  {
    //     fp = fopen(checkfile.c_str(), "a");
    //     fprintf(fp, "keyframes at checkpoint %d\n", n);
    // }

    const float &w = mKeyFrameSize / 2;
    const float h = w * 0.75 / 2;
    const float z = w * 0.6 / 2;

    vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();

    if (bDrawKF)
    {

        int maxId = 0, minId = 99999;
        KeyFrame *maxFrame = NULL, *minFrame = NULL;
        set<int> ids;

        sort(vpKFs.begin(), vpKFs.end(), kfcmp);
        for (size_t i = 0; i < int(vpKFs.size()) - 1; i++)
        {
            cv::Mat Twc1 = vpKFs[i]->GetPoseInverse().clone();
            cv::Mat Twc2 = vpKFs[i + 1]->GetPoseInverse().clone();
            double x1 = Twc1.at<float>(0, 3);
            double y1 = Twc1.at<float>(1, 3);
            double z1 = Twc1.at<float>(2, 3);
            double x2 = Twc2.at<float>(0, 3);
            double y2 = Twc2.at<float>(1, 3);
            double z2 = Twc2.at<float>(2, 3);

            glLineWidth(mKeyFrameLineWidth * 3);
            glColor3f(0.0f, 0.0f, 1.0f);
            glBegin(GL_LINES);
            glVertex3f(x1, y1, z1);
            glVertex3f(x2, y2, z2);
            glEnd();
            //  cout << i << endl;
        }

        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame *pKF = vpKFs[i];

            ids.insert(pKF->mnFrameId);
            if (pKF->mnFrameId >= maxId)
            {
                maxId = pKF->mnFrameId;
                maxFrame = pKF;
            }
            if (pKF->mnFrameId <= minId)
            {
                minId = pKF->mnFrameId;
                minFrame = pKF;
            }

            cv::Mat Twc = pKF->GetPoseInverse().t();

            if (check)
            {

                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                        fprintf(fp, "%lf ", Twc.at<float>(i, j));
                fprintf(fp, "\n");
            }

            // glPushMatrix();

            // glMultMatrixf(Twc.ptr<GLfloat>(0));

            // glLineWidth(mKeyFrameLineWidth);
            // glColor3f(0.0f,0.0f,1.0f);
            // glBegin(GL_LINES);
            // glVertex3f(0,0,0);
            // glVertex3f(w,h,z);
            // glVertex3f(0,0,0);
            // glVertex3f(w,-h,z);
            // glVertex3f(0,0,0);
            // glVertex3f(-w,-h,z);
            // glVertex3f(0,0,0);
            // glVertex3f(-w,h,z);

            // glVertex3f(w,h,z);
            // glVertex3f(w,-h,z);

            // glVertex3f(-w,h,z);
            // glVertex3f(-w,-h,z);

            // glVertex3f(-w,h,z);
            // glVertex3f(w,h,z);

            // glVertex3f(-w,-h,z);
            // glVertex3f(w,-h,z);
            // glEnd();

            // glPopMatrix();
        }
        double tscale = 933/928.0; 


        if (Twc_gts.size() != 0 && maxFrame != NULL and minFrame != maxFrame)
        {
            KeyFrame *k1 = minFrame, *k2 = maxFrame;
            cv::Mat rel = k2->GetPose() * k1->GetPoseInverse();
            // cout << "11111111" << endl;
            // cout << k1->mnFrameId << " " << k2->mnFrameId << endl;
            // cout << Twc_gts.size() << endl;
            cv::Mat Twc_gt1 = Twc_gts[int(tscale * k1->mnFrameId)], Twc_gt2 = Twc_gts[int(tscale * k2->mnFrameId)];
            cv::Mat rel_gt = Twc_gt2.inv() * Twc_gt1;
            // cout << "222222222" << endl;
            Eigen::Vector3f t1, t2;
            // for (int i = 0; i < 3; i++) {
            t1 << rel.at<float>(0, 3), rel.at<float>(1, 3), rel.at<float>(2, 3);
            t2 << rel_gt.at<float>(0, 3), rel_gt.at<float>(1, 3), rel_gt.at<float>(2, 3);
            // }

            if (check)
            {
                fprintf(fp, "ground truth keyframes at checkpoint %d\n", n);
            }

            double scale = t2.norm() / t1.norm();
            // cout << "scale: " << scale << endl;
            // cout << "k1pose: "<< endl;
            // cout << k1->GetPose() << endl;
            //  cout << "minid: " << minId << endl;
            //  cout << "maxid: " << maxId << endl;

            #if SHOWGT // 0 not show gt 1 show gt
            for (size_t i = 0; i < maxId * tscale; i++)
            {

                cv::Mat Twc1 = Twc_gts[i].clone();
                Twc1 = k1->GetPose() * Twc_gts[int(tscale * k1->mnFrameId)].inv() * Twc1;
                cv::Mat Twc2 = Twc_gts[i + 1].clone();
                Twc2 = k1->GetPose() * Twc_gts[int(tscale * k1->mnFrameId)].inv() * Twc2;
                double x1 = Twc1.at<float>(0, 3);
                double y1 = Twc1.at<float>(1, 3);
                double z1 = Twc1.at<float>(2, 3);
                double x2 = Twc2.at<float>(0, 3);
                double y2 = Twc2.at<float>(1, 3);
                double z2 = Twc2.at<float>(2, 3);

                glLineWidth(mKeyFrameLineWidth * 3);
                //glColor3f(0.0f,0.0f,0.0f);
                glColor3f(1.0, 20.0 / 255, 147.0 / 255);
                glBegin(GL_LINES);
                glVertex3f(x1, y1, z1);
                glVertex3f(x2, y2, z2);
                glEnd();
                // cout << i << endl;
            }
            #endif

            // for(size_t i=0; i<Twc_gts.size(); i++)
            // {
            //     if (!ids.count(i)) continue;
            //     cv::Mat Twc = Twc_gts[int(tscale * i)].clone();

            //     Twc = k1->GetPose() * Twc_gts[int(tscale * k1->mnFrameId)].inv() * Twc;

            //     Twc = Twc.t();
            //     for (int j = 0; j < 3; j++) Twc.at<float>(3, j) /= scale;

            //     if (check) {
            //         for (int i = 0; i < 4; i++)
            //             for (int j = 0; j < 4; j++)
            //                 fprintf(fp, "%lf ", Twc.at<float>(i, j));
            //         fprintf(fp, "\n");
            //     }

            //     glPushMatrix();
            //     glMultMatrixf(Twc.ptr<GLfloat>(0));

            //     glLineWidth(mKeyFrameLineWidth);
            //     glColor3f(1.0f,0.0f,0.0f);
            //     glBegin(GL_LINES);
            //     glVertex3f(0,0,0);
            //     glVertex3f(w,h,z);
            //     glVertex3f(0,0,0);
            //     glVertex3f(w,-h,z);
            //     glVertex3f(0,0,0);
            //     glVertex3f(-w,-h,z);
            //     glVertex3f(0,0,0);
            //     glVertex3f(-w,h,z);

            //     glVertex3f(w,h,z);
            //     glVertex3f(w,-h,z);

            //     glVertex3f(-w,h,z);
            //     glVertex3f(-w,-h,z);

            //     glVertex3f(-w,h,z);
            //     glVertex3f(w,h,z);

            //     glVertex3f(-w,-h,z);
            //     glVertex3f(w,-h,z);
            //     glEnd();

            //     glPopMatrix();
            // }
        }
    }

    if (check)
        fclose(fp);

    if (bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f, 1.0f, 0.0f, 0.6f);
        glBegin(GL_LINES);

        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            // Covisibility Graph
            const vector<KeyFrame *> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
            cv::Mat Ow = vpKFs[i]->GetCameraCenter();
            if (!vCovKFs.empty())
            {
                for (vector<KeyFrame *>::const_iterator vit = vCovKFs.begin(), vend = vCovKFs.end(); vit != vend; vit++)
                {
                    if ((*vit)->mnId < vpKFs[i]->mnId)
                        continue;
                    cv::Mat Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
                    glVertex3f(Ow2.at<float>(0), Ow2.at<float>(1), Ow2.at<float>(2));
                }
            }

            // Spanning tree
            KeyFrame *pParent = vpKFs[i]->GetParent();
            if (pParent)
            {
                cv::Mat Owp = pParent->GetCameraCenter();
                glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
                glVertex3f(Owp.at<float>(0), Owp.at<float>(1), Owp.at<float>(2));
            }

            // Loops
            set<KeyFrame *> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for (set<KeyFrame *>::iterator sit = sLoopKFs.begin(), send = sLoopKFs.end(); sit != send; sit++)
            {
                if ((*sit)->mnId < vpKFs[i]->mnId)
                    continue;
                cv::Mat Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
                glVertex3f(Owl.at<float>(0), Owl.at<float>(1), Owl.at<float>(2));
            }
        }

        glEnd();
    }
}

void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;
    const float h = w * 0.75;
    const float z = w * 0.6;

    glPushMatrix();

#ifdef HAVE_GLES
    glMultMatrixf(Twc.m);
#else
    glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);

    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);

    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);
    glEnd();

    glPopMatrix();
}

void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.clone();
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
    if (!mCameraPose.empty())
    {
        cv::Mat Rwc(3, 3, CV_32F);
        cv::Mat twc(3, 1, CV_32F);
        {
            unique_lock<mutex> lock(mMutexCamera);
            Rwc = mCameraPose.rowRange(0, 3).colRange(0, 3).t();
            twc = -Rwc * mCameraPose.rowRange(0, 3).col(3);
        }

        M.m[0] = Rwc.at<float>(0, 0);
        M.m[1] = Rwc.at<float>(1, 0);
        M.m[2] = Rwc.at<float>(2, 0);
        M.m[3] = 0.0;

        M.m[4] = Rwc.at<float>(0, 1);
        M.m[5] = Rwc.at<float>(1, 1);
        M.m[6] = Rwc.at<float>(2, 1);
        M.m[7] = 0.0;

        M.m[8] = Rwc.at<float>(0, 2);
        M.m[9] = Rwc.at<float>(1, 2);
        M.m[10] = Rwc.at<float>(2, 2);
        M.m[11] = 0.0;

        M.m[12] = twc.at<float>(0);
        M.m[13] = twc.at<float>(1);
        M.m[14] = twc.at<float>(2);
        M.m[15] = 1.0;
    }
    else
        M.SetIdentity();
}

} // namespace ORB_SLAM2
