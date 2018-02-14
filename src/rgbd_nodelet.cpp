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

#include <string>
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

typedef union
{
  float f;
  struct {
    uint8_t b;
    uint8_t g;
    uint8_t r;
    uint8_t a;
  };
} FColor;

class RGBDNodelet : public nodelet::Nodelet
{
public:
  RGBDNodelet()
    : tfListener(tfBuffer), have_info_(false)
  {
  }

  virtual ~RGBDNodelet()
  {
  }

  virtual
  void onInit()
  {
    nh_ = getNodeHandle();
    ros::NodeHandle cfg = getPrivateNodeHandle();
    // if image_width/height is > 0, then use it to generate a different size synthetic image
    cfg.param("image_height", height_, 0);
    cfg.param("image_width", width_, 0);
    cfg.param("frame_id", frame_id_, std::string("INVALID"));

    // set up the listeners
    cloud_sub_ = nh_.subscribe("colored_cloud", 1, &RGBDNodelet::cloud_cb, this);
    info_sub_ = nh_.subscribe("camera_info", 1, &RGBDNodelet::info_cb, this);

    // set up the publishers
    rgb_pub_ = nh_.advertise<sensor_msgs::Image>("rgb", 1);
    depth_pub_ = nh_.advertise<sensor_msgs::Image>("depth", 1);
  }

  bool project(const Eigen::Vector3d& wp, Eigen::Vector2d& ip)
  {
    // project point to color image plane
    if (!std::isfinite(wp[0])) {
      ip[0] = ip[1] = -2;
      return false;
    }
    ip[0] = ci_.K[0] * wp[0] / wp[2] + ci_.K[2];
    ip[1] = ci_.K[4] * wp[1] / wp[2] + ci_.K[5];
    if (ip[0] >= 0 && ip[0] < ci_.width &&
        ip[1] >= 0 && ip[1] < ci_.height) return true;
    return false;
  }

  inline
  int idx(int u, int v)
  {
    return v * ci_.width + u;
  }
  
  void cloud_cb(const sensor_msgs::PointCloud2::ConstPtr& cloud)
  {
    if (!have_info_) return;
    NODELET_DEBUG("rgbd cloud cb");

    // lookup the transform
    geometry_msgs::TransformStamped tf;
    try {
      tf = tfBuffer.lookupTransform(frame_id_,
                                    cloud->header.frame_id,
                                    cloud->header.stamp);
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

    rgb = cv::Mat::zeros(ci_.height, ci_.width, CV_8UC3);
    depth = cv::Mat::zeros(ci_.height, ci_.width, CV_16U);
    
    uint32_t rgb_offset = cloud->fields[6].offset;
    size_t N = cloud->width * cloud->height;
    NODELET_DEBUG("N: %lu, %d", N, rgb_offset);
    for (size_t i = 0; i < N; ++i) {
      const float* itC = reinterpret_cast<const float*>(&cloud->data[i * cloud->point_step]);
      Eigen::Vector3d pt(*(itC), *(itC+1), *(itC+2));
      Eigen::Vector3d tpt = T * pt;

      uint16_t D = (uint16_t)floor(tpt[2] * 1000.0);
      Eigen::Vector2d ip;
      if (project(tpt, ip)) {
        int u = (int)std::round(ip[0]-0.5);
        int v = (int)std::round(ip[1]-0.5);
        //int i = idx(u,v);
        if (depth.at<uint16_t>(v,u) > 0 && depth.at<uint16_t>(v,u) < D) continue;
        // set depth
        depth.at<uint16_t>(v,u) = D;
        FColor color;
        color.f = *reinterpret_cast<const float*>(&cloud->data[i * cloud->point_step] + rgb_offset);
        // set color
        rgb.at<cv::Vec3b>(v,u) = cv::Vec3b(color.b, color.g, color.r);
      }
    }

    // cv::namedWindow("rgb");
    // cv::namedWindow("depth");
    // cv::imshow("rgb", rgb);
    // cv::imshow("depth", depth);
    // cv::waitKey(50);
    
    // publish the images
    std_msgs::Header hdr = cloud->header;
    hdr.frame_id = frame_id_;
    cv_bridge::CvImage crgb(hdr, "bgr8", rgb);
    cv_bridge::CvImage cdepth(hdr, "mono16", depth);
    sensor_msgs::Image mrgb, mdepth;
    crgb.toImageMsg(mrgb);
    cdepth.toImageMsg(mdepth);
    rgb_pub_.publish(mrgb);
    depth_pub_.publish(mdepth);
    
    // transform the cloud into the target frame
    // create the synthetic images
    // project the points into the images
    // smaller depth overrides larger depth
  }

  void info_cb(const sensor_msgs::CameraInfo::ConstPtr& info)
  {
    ci_ = *info;

    // change the size of the image
    if (width_ > 0 && height_ > 0) {
      ci_.width = width_;
      ci_.height = height_;
    }
    
    // create an idealized camerainfo
    ci_.K[2] = ci_.width / 2.0;
    ci_.K[5] = ci_.height / 2.0;
    double f = (ci_.K[0] + ci_.K[4]) / 2.0;
    ci_.K[0] = ci_.K[4] = f;

    ci_.P[2] = ci_.K[2];
    ci_.P[6] = ci_.K[5];
    ci_.P[0] = ci_.K[0];
    ci_.P[5] = ci_.K[4];

    NODELET_DEBUG("Info: %dx%d, %f %f %f %f", ci_.width, ci_.height,
                  ci_.K[0], ci_.K[4], ci_.K[2], ci_.K[5]);

    info_sub_.shutdown();
    have_info_ = true;    
  }

private:
  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener;
  ros::NodeHandle nh_;
  ros::Subscriber cloud_sub_;
  ros::Subscriber info_sub_;
  ros::Publisher rgb_pub_;
  ros::Publisher depth_pub_;

  bool have_info_;
  sensor_msgs::CameraInfo ci_;
  int width_;
  int height_;
  std::string frame_id_;

  cv::Mat rgb, depth;
};

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(RGBDNodelet, nodelet::Nodelet)
