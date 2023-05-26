#ifndef _TRACKER_HPP_
#define _TRACKER_HPP_

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/video/tracking.hpp>
#include <yolov8_trt.hpp>


/* bounding box kalman filter with const-velocity model
 * assume box size (w & h) was not changed in a short time
 * state: box center, width and height, (x, y, w, h, vx, vy)
 * measure: (x, y, w, h)
 * x(k+1) = x(k) + vx * dt
 * y(k+1) = y(k) + vy * dt
 * w(k+1) = w(k)
 * h(k+1) = h(k)
 */
class BoxXYWHKalmanFilter
{
public:
    BoxXYWHKalmanFilter(double dt, double pNoise=1e-5, double qNoise=1e-5):
        dt_(dt), pNoise_(pNoise), qNoise_(qNoise)
    {
        kf_ = std::make_shared<cv::KalmanFilter>();
        init();
    }

    ~BoxXYWHKalmanFilter(){}

    void init()
    {
        kf_->init(stateSize_, measSize_, 0, CV_32F);
        cv::setIdentity(kf_->transitionMatrix);
        cv::setIdentity(kf_->transitionMatrix.rowRange(0, measSize_).
            colRange(measSize_, stateSize_), cv::Scalar(dt_));
        kf_->measurementMatrix = cv::Mat::zeros(measSize_, stateSize_, CV_32F);
        cv::setIdentity(kf_->measurementMatrix);
        cv::setIdentity(kf_->processNoiseCov, cv::Scalar(pNoise_));
        cv::setIdentity(kf_->measurementNoiseCov, cv::Scalar(qNoise_));
    }

    void initStatus(const cv::Rect& bbox)
    {
        kf_->statePre = cv::Mat::zeros(stateSize_, 1, CV_32F);
        kf_->statePost = cv::Mat::zeros(stateSize_, 1, CV_32F);
        float cx = bbox.x + bbox.width / 2;
        float cy = bbox.y + bbox.height / 2;
        kf_->statePre.at<float>(0, 0) = cx;
        kf_->statePre.at<float>(1, 0) = cy;
        kf_->statePre.at<float>(2, 0) = bbox.width;
        kf_->statePre.at<float>(3, 0) = bbox.height;
        kf_->statePost.at<float>(0, 0) = cx;
        kf_->statePost.at<float>(1, 0) = cy;
        kf_->statePost.at<float>(2, 0) = bbox.width;
        kf_->statePost.at<float>(3, 0) = bbox.height;
    }

    const cv::Mat& predict()
    {
        return kf_->predict();
    }

    const cv::Mat& predict(double dt)
    {
        cv::setIdentity(kf_->transitionMatrix.rowRange(0, measSize_).
            colRange(measSize_, stateSize_), cv::Scalar(dt));
        return kf_->predict();
    }

    const cv::Mat& correct(const cv::Mat& measurement)
    {
        return kf_->correct(measurement);
    }

    cv::Mat setMeasure(const cv::Rect& bbox) const
    {
        cv::Mat measure = cv::Mat::zeros(measSize_, 1, CV_32F);
        measure.at<float>(0, 0) = bbox.x + bbox.width / 2;
        measure.at<float>(1, 0) = bbox.y + bbox.height / 2;
        measure.at<float>(2, 0) = bbox.width;
        measure.at<float>(3, 0) = bbox.height;
        return measure;
    }

    cv::Rect getRectPost() const
    {
        const float& x = kf_->statePost.at<float>(0, 0);
        const float& y = kf_->statePost.at<float>(1, 0);
        const float& w = kf_->statePost.at<float>(2, 0);
        const float& h = kf_->statePost.at<float>(3, 0);
        return cv::Rect(x-w/2, y-h/2, w, h);
    }

    cv::Rect getRectPre() const
    {
        const float& x = kf_->statePre.at<float>(0, 0);
        const float& y = kf_->statePre.at<float>(1, 0);
        const float& w = kf_->statePre.at<float>(2, 0);
        const float& h = kf_->statePre.at<float>(3, 0);
        return cv::Rect(x-w/2, y-h/2, w, h);
    }

private:
    double dt_;
    const int stateSize_ = 6;
    const int measSize_ = 4;
    double pNoise_, qNoise_;
    std::shared_ptr<cv::KalmanFilter> kf_;
};


/* simplified bounding box tracking with kalman filter and iou
 * step 0: select a bounding box for tracking
 * step 1: init tracker status
 * step 2: predict next position with kalman filter
 * step 3: detect objects in the next frame
 * step 4: select the best match based on the IOU between detection and prediction
 * step 5: correct bounding box position with the best match object
 * step 6: update tracking status
 */
class Tracker
{
public:
    static double computeIou(const cv::Rect& box1, const cv::Rect& box2)
    {
        int iArea = (box1 & box2).area();
        int uArea = box1.area() + box2.area() - iArea;
        return 1.0 * iArea / uArea;
    }

    Tracker(double dt=0.03, size_t keep=30, double iouThr=0.5):
        keep_(keep), iouThr_(iouThr)
    {
        bboxKF_ = std::make_shared<BoxXYWHKalmanFilter>(dt);
        initTracker();
    }
    
    ~Tracker() {}

    bool selectTarget(const cv::Point& p, const std::vector<Object>& objects);
    void step(const std::vector<Object>& objects);
    void draw(cv::Mat& img);
    
    void initTracker()
    {
        bboxKF_->init();
        lostCount = 0;
        target_.clear();
    }

    bool withTarget() const
    {
        return target_.size() > 0;
    }

    int getLabel() const
    {
        return label_;
    }

private:
    size_t keep_;
    std::vector<cv::Rect> target_;
    int label_;
    double iouThr_;
    int lostCount = 0;
    const int lostCountMax = 30;
    std::shared_ptr<BoxXYWHKalmanFilter> bboxKF_;
};


#endif