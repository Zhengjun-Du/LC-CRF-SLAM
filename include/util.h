#ifndef UTIL_H
#define UTIL_H

#include<iostream>
#include<iomanip>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<vector>
#include<opencv2/core/core.hpp>
using namespace std;

void GenerateRGBtxt(int start, int end){
  char* rgbpath	= "/home/dzj/projects/orbslam-dynamic/ORB_SLAM2/data/207-3/rgb.txt";
  ofstream of(rgbpath);
  double stime = 1341846313.553992;
  for(int i = start; i <= end; i++){
    string s = to_string(stime);
    of<<s<<"  rgb/"<<setw(7)<<setfill('0')<<i<<".png"<<endl;
    stime += 0.03;
  }
}

void GenerateDepthtxt(int start, int end){
  char* deppath	= "/home/dzj/projects/orbslam-dynamic/ORB_SLAM2/data/207-3/depth.txt";
  ofstream of(deppath);
  double stime = 1341846313.553992;
  for(int i = start; i <= end; i++){
    string s = to_string(stime);
    of<<s<<" depth/"<<setw(7)<<setfill('0')<<i<<".png"<<endl;
    stime += 0.03;
  }
}

void GenerateAssotxt(int start, int end){
  char* assopath= "/home/dzj/projects/orbslam-dynamic/ORB_SLAM2/data/207-3/associate.txt";
  ofstream of(assopath);
  double stime = 1341846313.553992;
  for(int i = start; i <= end; i++){
    string s = to_string(stime);
    of<<s<<" rgb/"<<setw(7)<<setfill('0')<<i<<".png "<<s<<" depth/"<<setw(7)<<setfill('0')<<i<<".png "<<endl;
    stime += 0.03;
  }
}

void GenaretaCustomData(int start, int end){
  GenerateRGBtxt(start, end);
  GenerateDepthtxt(start, end);
  GenerateAssotxt(start, end);
}


#endif