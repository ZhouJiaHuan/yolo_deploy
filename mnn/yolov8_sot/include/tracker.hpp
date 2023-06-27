#ifndef _TRACKER_HPP_
#define _TRACKER_HPP_

#include "box_kf.hpp"
#include "yolov8_mnn.hpp"


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
    enum KalmanType {KF_BBOX_XYWH, KF_BBOX_XYXY};
    static double computeIou(const cv::Rect& box1, const cv::Rect& box2)
    {
        int iArea = (box1 & box2).area();
        int uArea = box1.area() + box2.area() - iArea;
        return 1.0 * iArea / uArea;
    }

    Tracker(double dt=0.03, size_t keep=30, double iouThr=0.5, KalmanType kf=KF_BBOX_XYWH):
        dt_(dt), keep_(keep), iouThr_(iouThr)
    {
        if (kf == KF_BBOX_XYWH)
        {
            bboxKF_ = std::make_shared<BoxXYWHKalmanFilter>();
        }
        else if (kf == KF_BBOX_XYXY)
        {
            bboxKF_ = std::make_shared<BoxXYXYKalmanFilter>();
        }
        else
        {
            std::cerr << "invalid kalman filter type!" << std::endl;
            exit(EXIT_FAILURE);
        }
        initTracker();
    }
    
    ~Tracker() {}

    bool selectTarget(const cv::Point& p, const std::vector<Object>& objects);
    void step(const std::vector<Object>& objects);
    void draw(cv::Mat& img);
    
    void initTracker()
    {
        bboxKF_->initKf(dt_, 1e-5, 1e-5);
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
    double dt_;
    size_t keep_;
    std::vector<cv::Rect> target_;
    int label_;
    double iouThr_;
    int lostCount = 0;
    const int lostCountMax = 30;
    std::shared_ptr<BoxKalmanFilter> bboxKF_;
};


#endif