/*
 * Copyright 2018 University of Pennsylvania
 * Author: Jason Owens
 * 
 * This file is part of the pico_flexx_driver.
 *
 * pico_flexx_driver is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * pico_flexx_driver is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with pico_flexx_driver.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cassert>

#include <ros/ros.h>
#include <ros/console.h>
#include <nodelet/nodelet.h>

#include <std_msgs/Header.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h> 
#include <image_geometry/pinhole_camera_model.h>
#include <message_filters/subscriber.h>
#include <message_filters/cache.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <yaml-cpp/yaml.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/calib3d/calib3d.hpp>



/**
 * This nodelet republishes the pico flexx point cloud with color
 * derived from an extrinsically calibrated external camera.
 */
class ColorCloudNodelet : public nodelet::Nodelet
{
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                          sensor_msgs::PointCloud2> FrameSyncPolicy;
  
public:    

  ColorCloudNodelet()
    : tfListener(tfBuffer)
  {
  }
  
  virtual ~ColorCloudNodelet()
  {
    delete sync_;
    delete policy_;
  }
  
  virtual void onInit()
  {
    nh = getMTNodeHandle();
    ros::NodeHandle cfg = getMTPrivateNodeHandle();

    cfg.param("queue_size", queue_size_, 10);

    // subscribe to the color image camera info
    have_info_ = false;
    color_info_sub_ = nh.subscribe("camera_info", 1, &ColorCloudNodelet::info_cb, this);

    // configure the synchronization between the cloud and the image
    policy_ = new FrameSyncPolicy(queue_size_);
    sync_ = new message_filters::Synchronizer<FrameSyncPolicy>(*policy_);
    color_sub_.subscribe(nh, "image", 1);
    cloud_sub_.subscribe(nh, "cloud", 1);
    sync_->connectInput(color_sub_, cloud_sub_);
    sync_->registerCallback(boost::bind(&ColorCloudNodelet::process, this, _1, _2));

    pub_ = nh.advertise<sensor_msgs::PointCloud2>("color_cloud", 1);

  }

  void computeDistortMap(const sensor_msgs::CameraInfo& camInfo,
                         const image_geometry::PinholeCameraModel& cameraModel_,
                         std::vector<cv::Point2i>& distortedPoints_)
  {
    cv::Point2i bogus(0,0);
    distortedPoints_.resize(camInfo.height * camInfo.width);
    for (int v = 0; v < camInfo.height; v++) {
      for (int u = 0; u < camInfo.width; u++) {
        // raw <- rect
        cv::Point2d uv_raw = cameraModel_.unrectifyPoint(cv::Point2d(u,v));
        if (uv_raw.x < 0 || uv_raw.x >= (camInfo.width - 0.5) ||
            uv_raw.y < 0 || uv_raw.y >= (camInfo.height - 0.5)) {
          distortedPoints_[v * camInfo.width + u] = bogus;
        } else {
          cv::Point2i uvi((int)std::round(uv_raw.x),(int)std::round(uv_raw.y));
          distortedPoints_[v * camInfo.width + u] = uvi;
        }
      }
    }  
  }
  
  void computeDistortMapFisheye(const sensor_msgs::CameraInfo& camInfo,
                                std::vector<cv::Point2i>& distortedPoints_)
  {
    cv::Point2i bogus(0,0);

    // compute the new height and width of the undistorted image
    //double new_width, new_height;
    //estimate_resolution(camInfo, new_width, new_height);
    
    distortedPoints_.resize(camInfo.height * camInfo.width);
    for (int v = 0; v < camInfo.height; v++) {
      for (int u = 0; u < camInfo.width; u++) {
        // raw <- rect
        double uu = (u - K.at<float>(0,2))/K.at<float>(0,0);
        double vv = (v - K.at<float>(1,2))/K.at<float>(1,1);
        cv::Point2f p(uu,vv);
        std::vector<cv::Point2f> out(1);
        cv::fisheye::distortPoints(std::vector<cv::Point2f>(1,p), out, K, D);
        cv::Point2d uv_raw(out[0].x,out[0].y);
        if (uv_raw.x < 0 || uv_raw.x >= (camInfo.width - 0.9) ||
            uv_raw.y < 0 || uv_raw.y >= (camInfo.height - 0.9)) {
          distortedPoints_[v * camInfo.width + u] = bogus;
        } else {
          cv::Point2i uvi((int)std::round(uv_raw.x),(int)std::round(uv_raw.y));
          distortedPoints_[v * camInfo.width + u] = uvi;
        }
      }
    }    
  }

  
  // Point, upper left
  inline void pul(cv::Point2i& p, double u, double v)
  {
    p.x = (int)std::max(floor(u),0.0);
    p.y = (int)std::max(floor(v),0.0);
  }
  inline void pur(cv::Point2i& p, double u, double v)
  {
    p.x = (int)std::min(ceil(u),ci_.width-1.0);
    p.y = (int)std::max(floor(v),0.0);
  }
  inline void plr(cv::Point2i& p, double u, double v)
  {
    p.x = (int)std::min(ceil(u),ci_.width-1.0);
    p.y = (int)std::min(ceil(v),ci_.height-1.0);
  }
  inline void pll(cv::Point2i& p, double u, double v)
  {
    p.x = (int)std::max(floor(u),0.0);
    p.y = (int)std::min(ceil(v),ci_.height-1.0);
  }
  
  inline bool project(const Eigen::Vector3d& p, Eigen::Vector2d& ip)
  {
    // project point to color image plane
    if (!std::isfinite(p[0])) return false;
    ip[0] = ci_.K[0] * p[0] / p[2] + ci_.K[2];
    ip[1] = ci_.K[4] * p[1] / p[2] + ci_.K[5];
    //return true;
    if (ip[0] >= 0 && ip[0] < ci_.width &&
        ip[1] >= 0 && ip[1] < ci_.height) return true;
    return false;      
  }

  template<typename T>
  inline bool in_frustum(const T& ip)
  {
    if (ip.x >= 0 && ip.x < ci_.width &&
        ip.y >= 0 && ip.y < ci_.height) return true;
    return false;      
  }
  
  inline int idx(const cv::Point2i& p)
  {
    int i = p.y * ci_.width + p.x;
    assert(i >= 0 && i < (ci_.width * ci_.height));
    return i;
  }

  inline float get_color(const Eigen::Vector3d& pt,
                         const Eigen::Affine3d& T,
                         const cv::Mat& img)
  {
    Eigen::Vector3d tpt = T * pt;
    Eigen::Vector2d uv;
    if (project(tpt, uv)) {
      cv::Point2i ul, ur, ll, lr;
      double x = uv[0];
      double y = uv[1];
      int u = std::round(x);
      int v = std::round(y);
      cv::Point2i p(u,v);
      cv::Point2i& dul = distorted_points_[idx(p)];
      if (!in_frustum(dul)) return 0;
      cv::Vec3b vcolor =
        img.at<cv::Vec3b>(dul);
      
      // interpolation of the point color
      // pul(ul, x, y);
      // pur(ur, x, y);
      // pll(ll, x, y);
      // plr(lr, x, y);
      // cv::Point2i& dul = distorted_points_[idx(ul)];
      // cv::Point2i& dur = distorted_points_[idx(ur)];
      // cv::Point2i& dll = distorted_points_[idx(ll)];
      // cv::Point2i& dlr = distorted_points_[idx(lr)];
      // float ulw = (lr.x - x)*(lr.y - y);
      // float urw = (ll.y - y)*(x - ll.x);
      // float llw = (y - ur.y)*(ur.x - x);
      // float lrw = (x - ul.x)*(y - ul.y);
      // cv::Vec3b vcolor =
      //   img.at<cv::Vec3b>(dul)*ulw +
      //   img.at<cv::Vec3b>(dur)*urw +
      //   img.at<cv::Vec3b>(dll)*llw +
      //   img.at<cv::Vec3b>(dlr)*lrw;
      uint32_t color =
        (vcolor[0] << 16) +
        (vcolor[1] << 8) +
        vcolor[2];
      return *reinterpret_cast<float*>(&color);
    }
    return 0; // return black for pixels outside the camera frustum
  }
  
  void process(const sensor_msgs::Image::ConstPtr& color,
               const sensor_msgs::PointCloud2::ConstPtr& cloud)
  {
    if (!have_info_) return;
    cv_bridge::CvImageConstPtr imgptr = cv_bridge::toCvShare(color, "rgb8");
    const cv::Mat& img = imgptr->image;

    // create a new pointcloud message
    sensor_msgs::PointCloud2::Ptr msgCloud(new sensor_msgs::PointCloud2);
    // create cloud message for built-in cloud
    msgCloud->header = cloud->header;
    msgCloud->height = cloud->height;
    msgCloud->width = cloud->width;
    msgCloud->is_bigendian = false;
    msgCloud->is_dense = false;
    msgCloud->point_step = (uint32_t)(4 * sizeof(float) + sizeof(u_int16_t) + sizeof(u_int8_t) + sizeof(float));
    msgCloud->row_step = (uint32_t)(msgCloud->point_step * cloud->width);
    msgCloud->fields.resize(7);
    msgCloud->fields[0].name = "x";
    msgCloud->fields[0].offset = 0;
    msgCloud->fields[0].datatype = sensor_msgs::PointField::FLOAT32;
    msgCloud->fields[0].count = 1;
    msgCloud->fields[1].name = "y";
    msgCloud->fields[1].offset = msgCloud->fields[0].offset + (uint32_t)sizeof(float);
    msgCloud->fields[1].datatype = sensor_msgs::PointField::FLOAT32;
    msgCloud->fields[1].count = 1;
    msgCloud->fields[2].name = "z";
    msgCloud->fields[2].offset = msgCloud->fields[1].offset + (uint32_t)sizeof(float);
    msgCloud->fields[2].datatype = sensor_msgs::PointField::FLOAT32;
    msgCloud->fields[2].count = 1;
    msgCloud->fields[3].name = "noise";
    msgCloud->fields[3].offset = msgCloud->fields[2].offset + (uint32_t)sizeof(float);
    msgCloud->fields[3].datatype = sensor_msgs::PointField::FLOAT32;
    msgCloud->fields[3].count = 1;
    msgCloud->fields[4].name = "intensity";
    msgCloud->fields[4].offset = msgCloud->fields[3].offset + (uint32_t)sizeof(float);
    msgCloud->fields[4].datatype = sensor_msgs::PointField::UINT16;
    msgCloud->fields[4].count = 1;
    msgCloud->fields[5].name = "gray";
    msgCloud->fields[5].offset = msgCloud->fields[4].offset + (uint32_t)sizeof(uint16_t);
    msgCloud->fields[5].datatype = sensor_msgs::PointField::UINT8;
    msgCloud->fields[5].count = 1;
    // ADD the rgb field
    msgCloud->fields[6].name = "rgb";
    msgCloud->fields[6].offset = msgCloud->fields[5].offset + (uint32_t)sizeof(uint8_t);
    msgCloud->fields[6].datatype = sensor_msgs::PointField::FLOAT32;
    msgCloud->fields[6].count = 1;
    size_t N = msgCloud->width * msgCloud->height;
    msgCloud->data.resize(msgCloud->point_step * N);

    // lookup the transform
    geometry_msgs::TransformStamped tf;
    try {
      tf = tfBuffer.lookupTransform(color->header.frame_id,
                                    cloud->header.frame_id,
                                    color->header.stamp);
      ROS_DEBUG_THROTTLE(5, "Transform (t) from cloud -> color: [%f,%f,%f]",
                         tf.transform.translation.x,
                         tf.transform.translation.y,
                         tf.transform.translation.z);
    } catch (tf2::TransformException& ex) {
      NODELET_WARN("TF exception: %s", ex.what());
      return;
    }
    Eigen::Affine3d T;
    T = tf2::transformToEigen(tf);
    cv::Mat cvT = cv::Mat::eye(4,4,CV_64F);
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x)
        cvT.at<double>(x,y) = T(x,y);
    cv::Affine3d A(cvT);
    
    uint32_t rgb_offset = msgCloud->fields[6].offset;
    for (size_t i = 0; i < N; ++i) {
      const float* itR = reinterpret_cast<const float*>(&cloud->data[i * cloud->point_step]);
      float* itC = reinterpret_cast<float*>(&msgCloud->data[i * msgCloud->point_step]);
      float* itrgb = reinterpret_cast<float*>(&msgCloud->data[i * msgCloud->point_step] + rgb_offset);
      // copy the existing point into itC
      memcpy(itC, itR, cloud->point_step); 
      // set the color
      Eigen::Vector3d pt(*(itC),*(itC+1),*(itC+2));      
      *itrgb = get_color(pt, T, img);
    }

    pub_.publish(msgCloud);
  }
  
  void info_cb(const sensor_msgs::CameraInfo::ConstPtr& info)
  {
    if (have_info_) return;
    
    ci_ = *info;
    color_info_sub_.shutdown();

    // this is probably unnecessary
    K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = ci_.K[0];
    K.at<float>(1,1) = ci_.K[4];
    K.at<float>(0,2) = ci_.K[2];
    K.at<float>(1,2) = ci_.K[5];
    D = cv::Mat(4,1,CV_32F);
    for (size_t i = 0; i < 4; ++i)
      D.at<float>(i) = ci_.D[i];    

    if (ci_.distortion_model == "equidistant") {
      computeDistortMapFisheye(ci_, distorted_points_);
    } else {
      camera_model_.fromCameraInfo(ci_);
      computeDistortMap(ci_, camera_model_, distorted_points_);
    }
    have_info_ = true;
  }
  
private:
  ros::NodeHandle nh;
  ros::Publisher pub_;
  ros::Subscriber color_info_sub_;
  message_filters::Subscriber<sensor_msgs::Image> color_sub_;
  message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;
  FrameSyncPolicy* policy_;
  message_filters::Synchronizer<FrameSyncPolicy>* sync_;

  int queue_size_;
  bool have_info_;
  sensor_msgs::CameraInfo ci_;
  image_geometry::PinholeCameraModel camera_model_;

  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener;
  // pre-compute the distortion map
  std::vector<cv::Point2i> distorted_points_;

  cv::Mat K, D;
};

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(ColorCloudNodelet, nodelet::Nodelet)
