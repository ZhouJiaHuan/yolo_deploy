#ifndef _BOX_KF_HPP_
#define _BOX_KF_HPP_

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/video/tracking.hpp>


class BoxKalmanFilter
{
public:
    BoxKalmanFilter()
    {
        kf_ = std::make_shared<cv::KalmanFilter>();
    }
    BoxKalmanFilter(double dt, double pNoise, double qNoise)
    {
        kf_ = std::make_shared<cv::KalmanFilter>();
        initKf(dt, pNoise, qNoise);
    }
    virtual ~BoxKalmanFilter() {}

    virtual void initKf(double dt=0.033, double pNoise=1e-5, double qNoise=1e-5) {}
    virtual void initStatus(const cv::Rect& bbox) = 0;

    const cv::Mat& predict()
    {
        return kf_->predict();
    }

    virtual const cv::Mat& predict(double dt) = 0;

    const cv::Mat& correct(const cv::Mat& measurement)
    {
        return kf_->correct(measurement);
    }

    virtual cv::Mat setMeasure(const cv::Rect& bbox) const = 0;
    virtual cv::Rect getRectPre() const = 0;
    virtual cv::Rect getRectPost() const = 0;

public:
    std::shared_ptr<cv::KalmanFilter> kf_;
    const float minSize_ = 5;
};


/* bounding box kalman filter with const-velocity model
 * assume box size (w & h) was not changed in a short time
 * state: box center, width and height, (x, y, w, h, vx, vy)
 * measure: (x, y, w, h)
 * x(k+1) = x(k) + vx * dt
 * y(k+1) = y(k) + vy * dt
 * w(k+1) = w(k)
 * h(k+1) = h(k)
 */
class BoxXYWHKalmanFilter: public BoxKalmanFilter
{
public:
    BoxXYWHKalmanFilter()
    {

    }
    BoxXYWHKalmanFilter(double dt, double pNoise, double qNoise):
        BoxKalmanFilter(dt, pNoise, qNoise)
    {
        initKf(dt, pNoise, qNoise);
    }

    ~BoxXYWHKalmanFilter(){}

    void initKf(double dt=0.033, double pNoise=1e-5, double qNoise=1e-5)
    {
        kf_->init(stateSize_, measSize_, 0, CV_32F);
        cv::setIdentity(kf_->transitionMatrix);
        cv::setIdentity(kf_->transitionMatrix.rowRange(0, measSize_).
            colRange(measSize_, stateSize_), cv::Scalar(dt));
        kf_->measurementMatrix = cv::Mat::zeros(measSize_, stateSize_, CV_32F);
        cv::setIdentity(kf_->measurementMatrix);
        cv::setIdentity(kf_->processNoiseCov, cv::Scalar(pNoise));
        cv::setIdentity(kf_->measurementNoiseCov, cv::Scalar(qNoise));       
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

    const cv::Mat& predict(double dt)
    {
        cv::setIdentity(kf_->transitionMatrix.rowRange(0, measSize_).
            colRange(measSize_, stateSize_), cv::Scalar(dt));
        return kf_->predict();
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
        float w = fmax(minSize_, kf_->statePost.at<float>(2, 0));
        float h = fmax(minSize_, kf_->statePost.at<float>(3, 0));
        return cv::Rect(x-w/2, y-h/2, w, h);
    }

    cv::Rect getRectPre() const
    {
        const float& x = kf_->statePre.at<float>(0, 0);
        const float& y = kf_->statePre.at<float>(1, 0);
        float w = fmax(minSize_, kf_->statePost.at<float>(2, 0));
        float h = fmax(minSize_, kf_->statePost.at<float>(3, 0));
        return cv::Rect(x-w/2, y-h/2, w, h);
    }

private:
    const int stateSize_ = 6;
    const int measSize_ = 4;
};


/* bounding box kalman filter with const-velocity model
 * state: box top-left and bottom-right, (x1, y1, x2, y2, vx1, vy1, vx2, vy2)
 * measure: (x1, y1, x2, y2)
 * x1(k+1) = x1(k) + vx1 * dt
 * y1(k+1) = y1(k) + vy1 * dt
 * x2(k+1) = x2(k) + vx2 * dt
 * y2(k+1) = y2(k) + vy2 * dt
 */
class BoxXYXYKalmanFilter: public BoxKalmanFilter
{
public:
    BoxXYXYKalmanFilter()
    {

    }

    BoxXYXYKalmanFilter(double dt, double pNoise, double qNoise):
        BoxKalmanFilter(dt, pNoise, qNoise)
    {
        initKf(dt, pNoise, qNoise);
    }

    ~BoxXYXYKalmanFilter(){}

    void initKf(double dt=0.033, double pNoise=1e-5, double qNoise=1e-5)
    {
        kf_->init(stateSize_, measSize_, 0, CV_32F);
        cv::setIdentity(kf_->transitionMatrix);
        cv::setIdentity(kf_->transitionMatrix.rowRange(0, measSize_).
            colRange(measSize_, stateSize_), cv::Scalar(dt));
        kf_->measurementMatrix = cv::Mat::zeros(measSize_, stateSize_, CV_32F);
        cv::setIdentity(kf_->measurementMatrix);
        cv::setIdentity(kf_->processNoiseCov, cv::Scalar(pNoise));
        cv::setIdentity(kf_->measurementNoiseCov, cv::Scalar(qNoise));       
    }

    void initStatus(const cv::Rect& bbox)
    {
        kf_->statePre = cv::Mat::zeros(stateSize_, 1, CV_32F);
        kf_->statePost = cv::Mat::zeros(stateSize_, 1, CV_32F);
        float cx = bbox.x + bbox.width / 2;
        float cy = bbox.y + bbox.height / 2;
        kf_->statePre.at<float>(0, 0) = bbox.x;
        kf_->statePre.at<float>(1, 0) = bbox.y;
        kf_->statePre.at<float>(2, 0) = bbox.x + bbox.width;
        kf_->statePre.at<float>(3, 0) = bbox.y + bbox.height;
        kf_->statePost.at<float>(0, 0) = bbox.x;
        kf_->statePost.at<float>(1, 0) = bbox.y;
        kf_->statePost.at<float>(2, 0) = bbox.x + bbox.width;
        kf_->statePost.at<float>(3, 0) = bbox.y + bbox.height;
    }

    const cv::Mat& predict(double dt)
    {
        cv::setIdentity(kf_->transitionMatrix.rowRange(0, measSize_).
            colRange(measSize_, stateSize_), cv::Scalar(dt));
        return kf_->predict();
    }

    cv::Mat setMeasure(const cv::Rect& bbox) const
    {
        cv::Mat measure = cv::Mat::zeros(measSize_, 1, CV_32F);
        measure.at<float>(0, 0) = bbox.x;
        measure.at<float>(1, 0) = bbox.y;
        measure.at<float>(2, 0) = bbox.x + bbox.width;
        measure.at<float>(3, 0) = bbox.y + bbox.height;
        return measure;
    }

    cv::Rect getRectPost() const
    {
        const float& x1 = kf_->statePost.at<float>(0, 0);
        const float& y1 = kf_->statePost.at<float>(1, 0);
        const float& x2 = kf_->statePost.at<float>(2, 0);
        const float& y2 = kf_->statePost.at<float>(3, 0);
        return cv::Rect(x1, y1, fmax(minSize_, x2-x1), fmax(minSize_, y2-y1));
    }

    cv::Rect getRectPre() const
    {
        const float& x1 = kf_->statePre.at<float>(0, 0);
        const float& y1 = kf_->statePre.at<float>(1, 0);
        const float& x2 = kf_->statePre.at<float>(2, 0);
        const float& y2 = kf_->statePre.at<float>(3, 0);
        return cv::Rect(x1, y1, fmax(minSize_, x2-x1), fmax(minSize_, y2-y1));
    }

private:
    const int stateSize_ = 8;
    const int measSize_ = 4;
};

#endif