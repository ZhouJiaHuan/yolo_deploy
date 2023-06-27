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
    
    initTracker();
    if (minDis < 10000.0f)
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
    cv::Rect boxPre = bboxKF_->getRectPre();

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
        cv::Mat measure = bboxKF_->setMeasure(matchedBox);
        bboxKF_->correct(measure);

        if (target_.size() > keep_)
        {
            target_.pop_back();
        }
        target_.insert(target_.begin(), bboxKF_->getRectPost());
        lostCount = 0;
    }
    else if (lostCount < lostCountMax)
    {
        lostCount += 1;
        target_.insert(target_.begin(), boxPre);
    }
    else
    {
        initTracker();
        std::cout << "tracking target lost!" << std::endl;
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
        cv::circle(img, cv::Point(cx, cy), 2, cv::Scalar(0, 0, 255), -1);
    }
}

