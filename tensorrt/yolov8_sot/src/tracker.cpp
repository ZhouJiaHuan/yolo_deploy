#include "tracker.hpp"


bool Tracker::selectTarget(const cv::Point& p, const std::vector<Object>& objects)
{
    Object currentTarget;
    float minDis = 10000.0f;
    for (const auto& object: objects)
    {
        const auto& bbox = object.rect;
        if (bbox.contains(p))
        {
            float cx = bbox.x + bbox.width / 2;
            float cy = bbox.y + bbox.height / 2;
            float currentDis = sqrt(pow(cx-p.x, 2) + pow(cy-p.y, 2));
            if (currentDis < minDis)
            {
                currentTarget = object;
                minDis = currentDis; 
            }
        }
    }
    
    target_.clear();
    bboxKF_->init();
    if (minDis < 10000)
    {  
        target_.push_back(currentTarget.rect);
        label_ = currentTarget.label;
        bboxKF_->initStatus(currentTarget.rect);
        return true;
    }
    return false;
}


void Tracker::step(const std::vector<Object>& objects)
{
    // predict
    auto xPre = bboxKF_->predict();
    cv::Point tl(xPre.at<float>(0, 0), xPre.at<float>(1, 0));
    cv::Point br(xPre.at<float>(2, 0), xPre.at<float>(3, 0));
    cv::Rect boxPre(tl, br);

    double maxIou = 0.0;
    cv::Rect matchedBox;
    for (const auto& object: objects)
    {
        if (object.label == label_)
        {
            const auto& bbox = object.rect;
            // get bbox with maximum iou
            double currentIou = computeIou(boxPre, bbox);
            if (currentIou > maxIou)
            {
                matchedBox = bbox;
                maxIou = currentIou;
            }
        }
    }

    // iou threshold
    if (maxIou > iouThr_)
    {
        // update target_
        cv::Mat measure = cv::Mat::zeros(4, 1, CV_32F);
        measure.at<float>(0, 0) = matchedBox.x;
        measure.at<float>(1, 0) = matchedBox.y;
        measure.at<float>(2, 0) = matchedBox.x + matchedBox.width;
        measure.at<float>(3, 0) = matchedBox.y + matchedBox.height;
        auto xEst = bboxKF_->correct(measure);

        if (target_.size() > keep_)
        {
            target_.pop_back();
        }
        cv::Point tl(xEst.at<float>(0, 0), xEst.at<float>(1, 0));
        cv::Point br(xEst.at<float>(2, 0), xEst.at<float>(3, 0));
        target_.insert(target_.begin(), cv::Rect(tl, br));
        lostCount = 0;
    }
    else if (++lostCount < lostCountMax)
    {
        target_.insert(target_.begin(), boxPre);
    }
    else
    {
        // reset
        initTracker();
    }
}

void Tracker::draw(cv::Mat& img)
{
    if (target_.size() > 0)
    {
        cv::rectangle(img, target_[0], cv::Scalar(0, 0, 255), 2);
    }
    for (const auto& box: target_)
    {
        int cx = box.x + box.width / 2;
        int cy = box.y + box.height / 2;
        cv::circle(img, cv::Point(cx, cy), 5, cv::Scalar(0, 0, 255));
    }
}

