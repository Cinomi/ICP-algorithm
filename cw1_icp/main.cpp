#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <math.h>
#include <stdio.h>
#include <random>
#include <vector>
#include <igl/fit_plane.h>
#include <igl/viewer/Viewer.h>
#include <igl/per_vertex_normals.h>
#include <igl/copyleft/marching_cubes.h>

#include "nanoflann.hpp"
#include "tutorial_shared_path.h"
#include "nanogui/formhelper.h"
#include "nanogui/screen.h"
#include "icp.hpp"

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cstdlib>

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;
using namespace nanoflann;

Eigen::MatrixXd V1, V2, V3, V4, V5, matched, matched12, matched123, matched1234, matched12345, normal;
Eigen::MatrixXi F1, F2, F3, F4, F5;
Eigen::VectorXd dist;

int main(int argc, char** argv)
{
    
    igl::viewer::Viewer viewer;
    {
        viewer.core.show_lines=false;
    }
    
    int sample_gap = 1;           // gap between sub sampling in two mesh alignment
    int sample_gap_multi=1;       // sample gap in multiple mesh alignment
    int iteration = 15;
    int iteration_plane = 10;
    int rotation_angle = 0;
    double noise_level = 0.01;
    int iteration_multi = 50;
    
    igl::readOFF(TUTORIAL_SHARED_PATH "/bun000.off", V1, F1);     //m1
    igl::readOFF(TUTORIAL_SHARED_PATH "/bun045.off", V2, F2);     //m2
    
    // transform mesh 2 to show them with different positions
    Eigen::Vector3d initial_T;
    initial_T(0)=-0.05;
    initial_T(1)=0.0;
    initial_T(2)=0.0;
    V2 = V2 - initial_T.replicate(1, V2.rows()).transpose();
    
    // display meshes
    Eigen::MatrixXd V(V1.rows()+V2.rows(),V1.cols());
    V<<V1,V2;
    Eigen::MatrixXi F(F1.rows()+F2.rows(),F1.cols());
    F<<F1,(F2.array()+V1.rows());
    
    // yellow color for faces of first mesh, blue for second
    Eigen::MatrixXd C(F.rows(),3);
    C<<
    Eigen::RowVector3d(0.9,0.775,0.25).replicate(F1.rows(),1),
    Eigen::RowVector3d(0.2,0.2,1.0).replicate(F2.rows(),1);
    
    
    viewer.data.set_mesh(V,F);
    viewer.data.set_colors(C);
    
    
    viewer.callback_init = [&sample_gap, &iteration, &iteration_plane, &rotation_angle, &noise_level, &iteration_multi, &sample_gap_multi](igl::viewer::Viewer& viewer){
        viewer.ngui->addWindow(Eigen::Vector2i(900,10), "ICP Algorithm");
        viewer.ngui->addGroup("Coursework 1");
        
        ///////////////////////////////////////////////////////////////////////////////////
        /////////////////////////// Restart (Initialization) //////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////
        viewer.ngui->addButton("Initialization", [&](){
            viewer.data.clear();
            
            igl::readOFF(TUTORIAL_SHARED_PATH "/bun000.off", V1, F1);     //m1
            igl::readOFF(TUTORIAL_SHARED_PATH "/bun045.off", V2, F2);     //m2
            
            Eigen::Vector3d initial_T;
            initial_T(0)=-0.05;
            initial_T(1)=0.0;
            initial_T(2)=0.0;
            V2 = V2 - initial_T.replicate(1, V2.rows()).transpose();
            
            // display meshes
            Eigen::MatrixXd V(V1.rows()+V2.rows(),V1.cols());
            V<<V1,V2;
            Eigen::MatrixXi F(F1.rows()+F2.rows(),F1.cols());
            F<<F1,(F2.array()+V1.rows());
            
            // yellow color for faces of first mesh, blue for second
            Eigen::MatrixXd C(F.rows(),3);
            C<<
            Eigen::RowVector3d(0.9,0.775,0.25).replicate(F1.rows(),1),
            Eigen::RowVector3d(0.2,0.2,1.0).replicate(F2.rows(),1);
            
            viewer.data.set_mesh(V,F);
            viewer.data.set_colors(C);
        });
        
        ////////////////////////////////////////////////////////////////////////////////
        ///////////////////// Task 1 & 4 : point-to-point ICP //////////////////////////
        ////////////////////////////////////////////////////////////////////////////////
        viewer.ngui->addGroup("Task 1 & Task 4");
        viewer.ngui->addVariable<int>("Sample Gap", [&](int val){
            sample_gap = val;
        },[&](){
            return sample_gap;
        });
        viewer.ngui->addVariable<int>("Iteration", [&](int val){
            iteration = val;
        },[&](){
            return iteration;
        });
        viewer.ngui->addButton("Point-to-point ICP (two meshes alignment)", [&](){
            // viewer.data.clear();
            double time_start = clock();

            for(int i=0;i<iteration;i++){
                // get sampled source
                Eigen::MatrixXd sampled_V2;
                Eigen::MatrixXd tem_V2;
                
                sampled_V2 = icp::getSampleSource(V2, sample_gap);
                
                // get matched points set
                matched = icp::getMatchedPoints(sampled_V2, V1);
                
                // rigid transformation calculation
                pair<Matrix3d, Vector3d> RT = icp::getRigidTransform(sampled_V2, matched);

                // apply rigid transform on source mesh
                if(icp::detectError(V2, V1, RT)){
                    V2 =(V2 - RT.second.replicate(1, V2.rows()).transpose())* RT.first;
                }

            }
            
            double time_cost = (clock() - time_start) / CLOCKS_PER_SEC * 1.0;
            std::cout << "Rotation angle: "<< rotation_angle << endl;
            std::cout << "Iteration: "<<iteration << endl;
            std::cout << "Noise level: "<<noise_level << endl;
            std::cout << "Time Cost: " << time_cost << endl;
            
            // display meshes
            Eigen::MatrixXd V(V1.rows()+V2.rows(),V1.cols());
            V<<V1,V2;
            Eigen::MatrixXi F(F1.rows()+F2.rows(),F1.cols());
            F<<F1,(F2.array()+V1.rows());
            
            // yellow color for faces of first mesh, blue for second
            Eigen::MatrixXd C(F.rows(),3);
            C<<
            Eigen::RowVector3d(0.9,0.775,0.25).replicate(F1.rows(),1),
            Eigen::RowVector3d(0.2,0.2,1.0).replicate(F2.rows(),1);
            
            viewer.data.set_mesh(V,F);
            viewer.data.set_colors(C);
        });
        
        ////////////////////////////////////////////////////////////////////////////////
        /////////////////////////////// Task 2 : rotation //////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////
        viewer.ngui->addGroup("Task 2");
        viewer.ngui->addVariable<int>("Rotation angle (clockwise)", [&](int val){
            rotation_angle = val;
        },[&](){
            return rotation_angle;
        });
        viewer.ngui->addButton("Initialize M2 with above angle",[&](){
            viewer.data.clear();
            
            // rotation matrix (around z axis)
            Eigen::Matrix3d R;
            R << AngleAxisd(rotation_angle * M_PI/180, Eigen::Vector3d(0,0,1)).toRotationMatrix();
            
            // get new source mesh 2
            igl::readOFF(TUTORIAL_SHARED_PATH "/bun000.off", V1, F1);     //m1
            V2 = V1 * R;
            F2 = F1;
            
            // display meshes
            Eigen::MatrixXd V(V1.rows()+V2.rows(),V1.cols());
            V<<V1,V2;
            Eigen::MatrixXi F(F1.rows()+F2.rows(),F1.cols());
            F<<F1,(F2.array()+V1.rows());
            
            // yellow color for faces of first mesh, blue for second
            Eigen::MatrixXd C(F.rows(),3);
            C<<
            Eigen::RowVector3d(0.9,0.775,0.25).replicate(F1.rows(),1),
            Eigen::RowVector3d(0.2,0.2,1.0).replicate(F2.rows(),1);
            
            viewer.data.set_mesh(V,F);
            viewer.data.set_colors(C);
        });
        
        ///////////////////////////////////////////////////////////////////////////////
        //////////////////////////////// Task 3 : noise ///////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////
        viewer.ngui->addGroup("Task 3");
        viewer.ngui->addVariable<double>("Noise level", [&](double val){
            noise_level = val;
        },[&](){
            return noise_level;
        });
        viewer.ngui->addButton("Preturb M2 with noise", [&](){
            viewer.data.clear();
            
            igl::readOFF(TUTORIAL_SHARED_PATH "/bun000.off", V1, F1);    //mesh 1
            
            V2 = icp::addNoise(V1, noise_level);
            
            // rotation matrix
            Eigen::Matrix3d R;
            R << AngleAxisd(-45 * M_PI/180, Eigen::Vector3d(0,1,0)).toRotationMatrix();
            V2 = V2 * R;
            
            F2 = F1;
            
            // display meshes
            Eigen::MatrixXd V(V1.rows()+V2.rows(),V1.cols());
            V<<V1,V2;
            Eigen::MatrixXi F(F1.rows()+F2.rows(),F1.cols());
            F<<F1,(F2.array()+V1.rows());
            
            // yellow color for faces of first mesh, blue for second
            Eigen::MatrixXd C(F.rows(),3);
            C<<
            Eigen::RowVector3d(0.9,0.775,0.25).replicate(F1.rows(),1),
            Eigen::RowVector3d(0.2,0.2,1.0).replicate(F2.rows(),1);
            
            viewer.data.set_mesh(V,F);
            viewer.data.set_colors(C);
        });
        
        ///////////////////////////////////////////////////////////////////////////////////
        //////////////////////// Task 5 : Multi-mesh Alignment ////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////
        viewer.ngui->addGroup("Task 5");
        viewer.ngui->addVariable<int>("Sample Gap", [&](int val){
            sample_gap_multi = val;
        },[&](){
            return sample_gap_multi;
        });
        viewer.ngui->addVariable<int>("Iteration", [&](int val){
            iteration_multi = val;
        },[&](){
            return iteration_multi;
        });
        viewer.ngui->addButton("Initialize 5 meshes", [&](){
            viewer.data.clear();
            
            igl::readOFF(TUTORIAL_SHARED_PATH "/bun270.off", V1, F1);
            igl::readOFF(TUTORIAL_SHARED_PATH "/bun315.off", V2, F2);
            igl::readOFF(TUTORIAL_SHARED_PATH "/bun000.off", V3, F3);
            igl::readOFF(TUTORIAL_SHARED_PATH "/bun045.off", V4, F4);
            igl::readOFF(TUTORIAL_SHARED_PATH "/bun090.off", V5, F5);
            
            // display meshes
            Eigen::MatrixXd V(V1.rows()+V2.rows()+V3.rows()+V4.rows()+V5.rows(), V1.cols());
            V<<V1, V2, V3, V4, V5;
            Eigen::MatrixXi F(F1.rows()+F2.rows()+F3.rows()+F4.rows()+F5.rows(), F1.cols());
            F<<F1, (F2.array()+V1.rows()), (F3.array()+V2.rows()+V1.rows()), (F4.array()+V3.rows()+V2.rows()+V1.rows()), (F5.array()+V4.rows()+V3.rows()+V2.rows()+V1.rows());
            
            
            Eigen::MatrixXd C(F.rows(),3);
            C<<
            Eigen::RowVector3d(0.51,0.51,0.51).replicate(F1.rows(),1),    // gray
            Eigen::RowVector3d(0.227,0.37,0.804).replicate(F2.rows(),1), // blue
            Eigen::RowVector3d(0.93,0.0,0.0).replicate(F3.rows(),1),   // red
            Eigen::RowVector3d(0.18,0.545,0.34).replicate(F4.rows(),1),  // green
            Eigen::RowVector3d(1.0,1.0,0.0).replicate(F5.rows(),1);      // yellow
            
            viewer.data.set_mesh(V,F);
            viewer.data.set_colors(C);
            
        });
        viewer.ngui->addButton("Multimesh Alignment", [&](){
            
            for(int i1=0;i1<iteration_multi;i1++){
                Eigen::MatrixXd sample_src1;
                sample_src1=icp::getSampleSource(V1, sample_gap_multi);
                matched12 = icp::getMatchedPoints(sample_src1, V2);
                pair<Matrix3d, Vector3d> RT1 = icp::getRigidTransform(sample_src1, matched12);
                V1 =(V1 - RT1.second.replicate(1, V1.rows()).transpose())* RT1.first;
            }
            Eigen::MatrixXd V12(V1.rows()+V2.rows(), V1.cols());
            V12<<V1, V2;
            Eigen::MatrixXi F12(F1.rows()+F2.rows(), F1.cols());
            F12<<F1, (F2.array()+V1.rows());
            
            
            for(int i2=0; i2<iteration_multi; i2++){
                Eigen::MatrixXd sample_src2;
                sample_src2=icp::getSampleSource(V3, sample_gap_multi);
                matched123 = icp::getMatchedPoints(sample_src2, V12);
                pair<Matrix3d, Vector3d> RT2 = icp::getRigidTransform(sample_src2, matched123);
                V3 = (V3 - RT2.second.replicate(1, V3.rows()).transpose())*RT2.first;
            }
            Eigen::MatrixXd V123(V12.rows()+V3.rows(), V1.cols());
            V123<<V12, V3;
            Eigen::MatrixXi F123(F12.rows()+F3.rows(), F1.cols());
            F123<<F12, (F3.array()+V12.rows());
            
            for(int i3=0; i3<iteration_multi; i3++){
                Eigen::MatrixXd sample_src3;
                sample_src3=icp::getSampleSource(V4, sample_gap_multi);
                matched1234 = icp::getMatchedPoints(sample_src3, V123);
                pair<Matrix3d, Vector3d> RT3 = icp::getRigidTransform(sample_src3, matched1234);
                V4 = (V4 - RT3.second.replicate(1, V4.rows()).transpose())*RT3.first;
            }
            Eigen::MatrixXd V1234(V123.rows()+V4.rows(), V1.cols());
            V1234<<V123, V4;
            Eigen::MatrixXi F1234(F123.rows()+F4.rows(), F1.cols());
            F1234<<F123,(F4.array()+V123.rows());
            
            for(int i4=0; i4<iteration_multi; i4++){
                Eigen::MatrixXd sample_src4;
                sample_src4=icp::getSampleSource(V5, sample_gap_multi);
                matched12345 = icp::getMatchedPoints(sample_src4, V1234);
                pair<Matrix3d, Vector3d> RT4 = icp::getRigidTransform(sample_src4, matched12345);
                V5 = (V5 - RT4.second.replicate(1, V5.rows()).transpose())*RT4.first;
            }
            Eigen::MatrixXd V12345(V1234.rows()+V5.rows(), V1.cols());
            V12345<<V1234, V5;
            Eigen::MatrixXi F12345(F1234.rows()+F5.rows(), F1.cols());
            F12345<<F1234,(F5.array()+V1234.rows());
            
            Eigen::MatrixXd C(F12345.rows(),3);
            C<<
            Eigen::RowVector3d(0.51,0.51,0.51).replicate(F1.rows(),1),    // gray
            Eigen::RowVector3d(0.227,0.37,0.804).replicate(F2.rows(),1), // blue
            Eigen::RowVector3d(0.93,0.0,0.0).replicate(F3.rows(),1),   // red
            Eigen::RowVector3d(0.18,0.545,0.34).replicate(F4.rows(),1),  // green
            Eigen::RowVector3d(1.0,1.0,0.0).replicate(F5.rows(),1);      // yellow
            
            viewer.data.set_mesh(V12345,F12345);
            viewer.data.set_colors(C);
        });
        
        //////////////////////////////////////////////////////////////////////////
        ////////////////////// Task 6: point-to-plane ICP ////////////////////////
        //////////////////////////////////////////////////////////////////////////
        viewer.ngui->addGroup("Task 6");
        viewer.ngui->addVariable<int>("Iteration", [&](int val){
            iteration_plane = val;
        },[&](){
            return iteration_plane;
        });
        viewer.ngui->addButton("Point-to-plane ICP", [&](){
            for(int i=0;i<iteration_plane;i++){
            // get normal
            normal=icp::getNormal(V2, V1);
            // get matched point set
            matched = icp::getMatchedPoints(V2, V1);
            pair<Matrix3d, Vector3d> RT = icp::point2planeICP(V2, matched, normal);
                if(icp::detectError(V2, V1, RT)){
                    V2 =(V2 - RT.second.replicate(1, V2.rows()).transpose())* RT.first;
                }
            }
            
            // display meshes
            Eigen::MatrixXd V(V1.rows()+V2.rows(),V1.cols());
            V<<V1,V2;
            Eigen::MatrixXi F(F1.rows()+F2.rows(),F1.cols());
            F<<F1,(F2.array()+V1.rows());
            
            // yellow color for faces of first mesh, blue for second
            Eigen::MatrixXd C(F.rows(),3);
            C<<
            Eigen::RowVector3d(0.9,0.775,0.25).replicate(F1.rows(),1),
            Eigen::RowVector3d(0.2,0.2,1.0).replicate(F2.rows(),1);
            
            viewer.data.set_mesh(V,F);
            viewer.data.set_colors(C);
            
        });
        
        viewer.screen->performLayout();
        return false;
    };

    viewer.launch();
    return 0;
}


