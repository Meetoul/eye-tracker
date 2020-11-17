#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <cstdint>
#include <numeric>

void log(const std::string& msg) {
    std::cout << msg << std::endl;
}

class TimeLogger {
public:
    void start() {
        mBegin = std::chrono::steady_clock::now();
    }

    void stop(const std::string action) {
        using namespace std::chrono;
        const auto end = steady_clock::now();
        const auto duration = duration_cast<milliseconds>(end - mBegin).count();

        std::cout << action << " took " << duration << "ms" << std::endl;
    }

private:
    std::chrono::steady_clock::time_point mBegin;
};

TimeLogger tl;

static const std::string WIN = "tracker";

static const std::string PREDICTOR_FILE = "shape_predictor_68_face_landmarks.dat";

static const std::vector<unsigned long> LEFT_POINTS = {36, 37, 38, 39, 40, 41};
static const std::vector<unsigned long> RIGHT_POINTS = {42, 43, 44, 45, 46, 47};

static const double ACCEPTABLE_DIFF = 40.0;

void apply_eye_mask(const cv::Mat& mask,
    const dlib::full_object_detection& shape_points,
    const std::vector<unsigned long> eye_points) {

    std::vector<cv::Point> points;

    for(const auto& ep : eye_points) {
        const auto p = shape_points.part(ep);
        points.emplace_back(p.x(), p.y());
    }

    cv::fillConvexPoly(mask, points, cv::Scalar(255, 255, 255));
}

int find_max_area(const std::vector<std::vector<cv::Point>> areas) {
    int maxAreaIdx = -1;
    double maxArea = 0;

    for(int i = 0; i < areas.size(); ++i) {
        const double area = cv::contourArea(areas[i]);
        if(area > maxArea) {
            maxArea = area;
            maxAreaIdx = i;
        }
    }

    return maxAreaIdx;
}

std::vector<cv::Point> get_max_area_contour(const cv::Mat& frame) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(frame, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    const auto maxContourIndex = find_max_area(contours);
    std::vector<cv::Point> contour;

    if (maxContourIndex >= 0) {
        contour = contours[maxContourIndex];
    }

    return contour;
}

cv::Point find_gaze(const cv::Mat& frame) {
    const auto contour = get_max_area_contour(frame);
    const auto moments = cv::moments(contour, true);

    double gaze_x = moments.m10 / moments.m00;
    double gaze_y = moments.m01 / moments.m00 ;

    return cv::Point{gaze_x, gaze_y};
}

class Calibrator {
public:
    Calibrator(double acceptableDiff, int iterations)
        : mAcceptableDiff{acceptableDiff}
        , mInterations{iterations} {
    }

    bool calibrate(const cv::Mat& frame) {
        if (mThreshold > 0) {
            return false;
        }

        cv::Point2f center;
        float radius;

        cv::Mat tempFrame;
        int calibratedThreshold = 0;

        for(int i = 100; i > 0; --i) {
            cv::threshold(frame, tempFrame, i, 255, cv::THRESH_BINARY);

            cv::erode(tempFrame, tempFrame, cv::Mat{}, cv::Point{-1, -1}, 2);
            cv::dilate(tempFrame, tempFrame, cv::Mat{}, cv::Point{-1, -1}, 4);
            cv::medianBlur(tempFrame, tempFrame, 3);

            cv::bitwise_not(tempFrame, tempFrame);

            const auto contour = get_max_area_contour(tempFrame);

            auto currentDiff = std::numeric_limits<double>::max();
            if (contour.size() > 3) {
                cv::minEnclosingCircle(contour, center, radius);

                const auto circleArea = CV_PI * radius * radius;
                const auto contourArea = cv::contourArea(contour);

                const auto areaDiff = circleArea - contourArea;
                const auto normalizedDiff = areaDiff / circleArea * 100;

                std::cout << "threshold: " << i
                    << ", circ: " << circleArea
                    << ", countour: " << contourArea
                    << ", diff: " << areaDiff
                    << ", normalized: " << normalizedDiff
                    << std::endl;

                if(normalizedDiff < mAcceptableDiff) {
                    calibratedThreshold = i;
                    break;
                }
            }
        }

        if (calibratedThreshold != 0) {
            mThresholds.emplace_back(calibratedThreshold);
            mInterations--;

            if (mInterations < 0) {
                mThreshold = std::accumulate(std::cbegin(mThresholds), std::cend(mThresholds), 0) / mThresholds.size();
                return false;
            }
        }

        return true;
    }

    int getThreshold() const {
        return mThreshold;
    }

private:
    double mAcceptableDiff;
    int mInterations;
    int mThreshold = -1;
    std::vector<int> mThresholds;
};

int main() {
    log("Started");

    cv::VideoCapture camera(0);
    if (!camera.isOpened()) {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return 1;
    }

    int threshold = 0;

    // create a window to display the images from the webcam
    cv::namedWindow(WIN, cv::WINDOW_NORMAL);
//    cv::createTrackbar("threshold", WIN, &threshold, 254);

    auto ffd = dlib::get_frontal_face_detector();

    dlib::shape_predictor sp;
    dlib::deserialize(PREDICTOR_FILE) >> sp;

    // this will contain the image from the webcam
    cv::Mat colorFrame;
    cv::Mat frame;
    cv::Mat mask;

    int dilation_size = 1;
    int dilation_type = cv::MORPH_RECT;
//    int dilation_type = MORPH_CROSS;
//    int dilation_type = MORPH_ELLIPSE;
    cv::Mat kernel = cv::getStructuringElement( dilation_type,
                         cv::Size(2*dilation_size + 1, 2*dilation_size+1),
                         cv::Point(dilation_size, dilation_size));

    Calibrator leftCalibrator(ACCEPTABLE_DIFF, 10);
    // display the frame until you press a key
    while (1) {
        // capture the next frame from the webcam
        camera >> colorFrame;

        cv::flip(colorFrame, colorFrame, 1);

        cv::cvtColor(colorFrame, frame, cv::COLOR_BGR2GRAY);

        dlib::cv_image<unsigned char> dlibImage{frame};

//        tl.start();
        const auto faces = ffd(dlibImage);
//        tl.stop("face detection");

        if(faces.size() == 1) {
//            log("Processing detected face");
//            tl.start();
            const auto faceShape = sp(dlibImage, faces[0]);
//            tl.stop("face shaping");

            if(faceShape.num_parts() == 68) {
//                log("Processing points");
                mask = cv::Mat::zeros(frame.size(),
                    frame.type());

                apply_eye_mask(mask, faceShape, LEFT_POINTS);
                apply_eye_mask(mask, faceShape, RIGHT_POINTS);

                cv::dilate(mask, mask, kernel, cv::Point{-1, -1}, 2);
                cv::bitwise_and(frame, mask, frame);

                // convert all 0,0,0 pixels of the frame to 255,255,255,
                // so eyeball is the only dark part left
                mask = (frame == 0);
                frame.setTo(255, mask);

                const auto mid = (faceShape.part(39).x() + faceShape.part(42).x()) / 2;

                if(!leftCalibrator.calibrate(frame)) {
                    threshold = leftCalibrator.getThreshold();
                    std::cout << "threshold is: " << threshold << std::endl;
                    cv::threshold(frame, frame, threshold, 255, cv::THRESH_BINARY);

                    cv::erode(frame, frame, cv::Mat{}, cv::Point{-1, -1}, 2);
                    cv::dilate(frame, frame, cv::Mat{}, cv::Point{-1, -1}, 4);
                    cv::medianBlur(frame, frame, 3);

                    cv::bitwise_not(frame, frame);
                }

//
//                const auto leftGaze = find_gaze(frame(cv::Rect{0, 0, mid, frame.rows}));
//                auto rightGaze = find_gaze(frame(cv::Rect{mid, 0, frame.cols - mid, frame.rows}));

 //               rightGaze += cv::Point{mid, 0};

//                cv::circle(colorFrame, leftGaze, 2, cv::Scalar(0,0,255),cv::FILLED);
//                cv::circle(colorFrame, rightGaze, 2, cv::Scalar(0,0,255),cv::FILLED);

                colorFrame.setTo(cv::Scalar(0, 255, 0), frame);
            } else {
                log("Face shape is not complete");
            }

        } else if (faces.empty()) {
            log("No faces detected");
        } else {
            log("More than one faces detected");
        }

        // show the image on the window
        cv::imshow(WIN, colorFrame);

        // wait (10ms) for a key to be pressed
        if (cv::waitKey(33) == 'q')
            break;
    }

    return 0;
}