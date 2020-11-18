#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>

#include <cstdint>
#include <future>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

void log(const std::string& msg) { std::cout << msg << std::endl; }

class TimeLogger {
public:
    void start() { mBegin = std::chrono::steady_clock::now(); }

    void stop(const std::string action) {
        using namespace std::chrono;
        const auto end = steady_clock::now();
        const auto duration = duration_cast<milliseconds>(end - mBegin).count();
        const auto mduration = duration_cast<microseconds>(end - mBegin).count();

        // std::cout << action << " took " << duration << "ms" << std::endl;
        std::cout << action << " took " << mduration << "micros" << std::endl;
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

void apply_eye_mask(const cv::Mat& mask, const dlib::full_object_detection& shape_points,
                    const std::vector<unsigned long> eye_points) {
    std::vector<cv::Point> points;

    for (const auto& ep : eye_points) {
        const auto p = shape_points.part(ep);
        points.emplace_back(p.x(), p.y());
    }

    cv::fillConvexPoly(mask, points, cv::Scalar(255, 255, 255));
}

int find_max_area(const std::vector<std::vector<cv::Point>> areas) {
    int maxAreaIdx = -1;
    double maxArea = 0;

    for (int i = 0; i < areas.size(); ++i) {
        const double area = cv::contourArea(areas[i]);
        if (area > maxArea) {
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

void select_pupil(const cv::Mat& in, cv::Mat& out, int threshold) {
    cv::threshold(in, out, threshold, 255, cv::THRESH_BINARY);

    cv::erode(out, out, cv::Mat{}, cv::Point{-1, -1}, 2);
    cv::dilate(out, out, cv::Mat{}, cv::Point{-1, -1}, 4);
    cv::medianBlur(out, out, 3);

    cv::bitwise_not(out, out);
}

cv::Point find_gaze(cv::Mat& frame, int threshold) {
    select_pupil(frame, frame, threshold);
    const auto contour = get_max_area_contour(frame);
    const auto moments = cv::moments(contour, true);

    double gaze_x = moments.m10 / moments.m00;
    double gaze_y = moments.m01 / moments.m00;

    return cv::Point{gaze_x, gaze_y};
}

class Calibrator {
public:
    Calibrator(double acceptableDiff, int iterations)
        : mAcceptableDiff{acceptableDiff}, mInterations{iterations} {}

    bool calibrate(const cv::Mat& frame) {
        if (mThreshold > 0) {
            return true;
        }

        cv::Point2f center;
        float radius;

        cv::Mat tempFrame;
        int calibratedThreshold = 0;

        for (int i = 100; i > 0; --i) {
            select_pupil(frame, tempFrame, i);

            const auto contour = get_max_area_contour(tempFrame);

            auto currentDiff = std::numeric_limits<double>::max();
            if (contour.size() > 3) {
                cv::minEnclosingCircle(contour, center, radius);

                const auto circleArea = CV_PI * radius * radius;
                const auto contourArea = cv::contourArea(contour);

                const auto areaDiff = circleArea - contourArea;
                const auto normalizedDiff = areaDiff / circleArea * 100;

                if (normalizedDiff < mAcceptableDiff) {
                    calibratedThreshold = i;
                    break;
                }
            }
        }

        if (calibratedThreshold != 0) {
            mThresholds.emplace_back(calibratedThreshold);
            mInterations--;

            if (mInterations < 0) {
                mThreshold = std::accumulate(std::cbegin(mThresholds), std::cend(mThresholds), 0) /
                             mThresholds.size();
                return true;
            }
        }

        return false;
    }

    int getThreshold() const { return mThreshold; }

private:
    double mAcceptableDiff;
    int mInterations;
    int mThreshold = -1;
    std::vector<int> mThresholds;
};

class TailDrawer {
public:
    TailDrawer(int tailSize, cv::Scalar color) : mTailSize{tailSize}, mColor{color} {}

    void draw(cv::Mat& img, const cv::Point& point) {
        if (point.x > 0 && point.y > 0) {
            addPoint(point);
        }

        if (mPoints.size() > 1) {
            for(int i = 1; i < mPoints.size(); ++i) {
                cv::line(img, mPoints[i - 1], mPoints[i], mColor, 2);
            }
        }
    }

private:
    void addPoint(const cv::Point& point) {
        if (mPoints.size() == mTailSize) {
            mPoints.erase(std::begin(mPoints));
        }
        mPoints.emplace_back(point);
    }

    const int mTailSize;
    cv::Scalar mColor;
    std::vector<cv::Point> mPoints;
};

int main() {
    cv::VideoCapture camera(0);
    if (!camera.isOpened()) {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return 1;
    }

    int threshold = 0;

    cv::namedWindow(WIN, cv::WINDOW_NORMAL);

    auto ffd = dlib::get_frontal_face_detector();

    dlib::shape_predictor sp;
    dlib::deserialize(PREDICTOR_FILE) >> sp;

    // this will contain the image from the webcam
    cv::Mat colorFrame;
    cv::Mat frame;
    cv::Mat mask;

    int dilation_size = 1;
    int dilation_type = cv::MORPH_RECT;
    // int dilation_type = MORPH_CROSS;
    // int dilation_type = MORPH_ELLIPSE;
    cv::Mat kernel = cv::getStructuringElement(
        dilation_type, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        cv::Point(dilation_size, dilation_size));

    Calibrator leftCalibrator(ACCEPTABLE_DIFF, 10);
    Calibrator rightCalibrator(ACCEPTABLE_DIFF, 10);

    bool calibrationComplete = false;

    int leftThreshold = -1;
    int rightThreshold = -1;

    const int tailSize = 200;

    TailDrawer rightEyeDrawer{tailSize, cv::Scalar{0, 0, 255}};
    TailDrawer leftEyeDrawer{tailSize, cv::Scalar{255, 0, 0}};
    // display the frame until you press a key
    while (1) {
        // capture the next frame from the webcam
        camera >> colorFrame;

        cv::flip(colorFrame, colorFrame, 1);

        cv::cvtColor(colorFrame, frame, cv::COLOR_BGR2GRAY);

        dlib::cv_image<unsigned char> dlibImage{frame};

        const auto faces = ffd(dlibImage);

        if (faces.size() == 1) {
            const auto faceShape = sp(dlibImage, faces[0]);

            if (faceShape.num_parts() == 68) {
                mask = cv::Mat::zeros(frame.size(), frame.type());

                apply_eye_mask(mask, faceShape, LEFT_POINTS);
                apply_eye_mask(mask, faceShape, RIGHT_POINTS);

                cv::dilate(mask, mask, kernel, cv::Point{-1, -1}, 2);
                cv::bitwise_and(frame, mask, frame);

                // convert all 0,0,0 pixels of the frame to 255,255,255,
                // so eyeball is the only dark part left
                mask = (frame == 0);
                frame.setTo(255, mask);

                const auto mid = (faceShape.part(39).x() + faceShape.part(42).x()) / 2;

                auto rightRoi = frame(cv::Rect{mid, 0, frame.cols - mid, frame.rows});
                auto leftRoi = frame(cv::Rect{0, 0, mid, frame.rows});

                if (!calibrationComplete) {
                    tl.start();
                    std::future<bool> rightFuture = std::async(
                        std::launch::async, &Calibrator::calibrate, &rightCalibrator, rightRoi);

                    const auto leftComplete = leftCalibrator.calibrate(leftRoi);

                    if (rightFuture.get() && leftComplete) {
                        calibrationComplete = true;

                        leftThreshold = leftCalibrator.getThreshold();
                        rightThreshold = rightCalibrator.getThreshold();

                        std::cout << "############### Calibration complete #################"
                                  << std::endl;
                        std::cout << "left threshold is: " << leftThreshold << std::endl;
                        std::cout << "right threshold is: " << rightThreshold << std::endl;
                    }
                    tl.stop("calibration");
                } else {
                    std::future<cv::Point> rightFuture = std::async(
                        std::launch::async, &find_gaze, std::ref(rightRoi), rightThreshold);

                    const auto leftGaze = find_gaze(leftRoi, leftThreshold);
                    const auto rightGaze = rightFuture.get() + cv::Point{mid, 0};

                    cv::circle(colorFrame, leftGaze, 2, cv::Scalar(255, 0, 0), cv::FILLED);
                    cv::circle(colorFrame, rightGaze, 2, cv::Scalar(0, 0, 255), cv::FILLED);

                    rightEyeDrawer.draw(colorFrame, rightGaze);
                    leftEyeDrawer.draw(colorFrame, leftGaze);
                }
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
        if (cv::waitKey(33) == 'q') break;
    }

    return 0;
}
