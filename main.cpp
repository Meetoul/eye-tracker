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
#include <tuple>
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

static const int GAZE_RANGE = 500;

void appplyEyeMask(const cv::Mat& mask, const dlib::full_object_detection& shapePoints,
                   const std::vector<unsigned long> eyePoints) {
    std::vector<cv::Point> points;

    for (const auto& ep : eyePoints) {
        const auto p = shapePoints.part(ep);
        points.emplace_back(p.x(), p.y());
    }

    cv::fillConvexPoly(mask, points, cv::Scalar(255, 255, 255));
}

void printMask(const cv::Mat& mask, const dlib::full_object_detection& shapePoints) {
    std::vector<cv::Point> points;

    for (int i = 0; i < 68; ++i) {
        const auto p = shapePoints.part(i);
        cv::circle(mask, cv::Point{p.x(), p.y()}, 2, cv::Scalar(255, 0, 0), 1, cv::LINE_8);
    }
}

int findMaxArea(const std::vector<std::vector<cv::Point>> areas) {
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

std::vector<cv::Point> getMaxAreaContour(const cv::Mat& frame) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(frame, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    const auto maxContourIndex = findMaxArea(contours);
    std::vector<cv::Point> contour;

    if (maxContourIndex >= 0) {
        contour = contours[maxContourIndex];
    }

    return contour;
}

void selectPupil(const cv::Mat& in, cv::Mat& out, int threshold) {
    cv::threshold(in, out, threshold, 255, cv::THRESH_BINARY);

    cv::erode(out, out, cv::Mat{}, cv::Point{-1, -1}, 2);
    cv::dilate(out, out, cv::Mat{}, cv::Point{-1, -1}, 4);
    cv::medianBlur(out, out, 3);

    cv::bitwise_not(out, out);
}

cv::Point2f findCenter(const cv::Mat& frame) {
    const auto contour = getMaxAreaContour(frame);
    const auto moments = cv::moments(contour, true);

    const float gaze_x = moments.m10 / moments.m00;
    const float gaze_y = moments.m01 / moments.m00;

    return cv::Point2f{gaze_x, gaze_y};
}

std::tuple<cv::Point2f, int> findEyeCenter(const cv::Mat& frame) {
    const auto contour = getMaxAreaContour(frame);
    const auto rect = cv::boundingRect(contour);

    //    std::cout << "Eye height is " << rect.height << std::endl;

    return std::make_tuple(cv::Point2f{rect.x + rect.width / 2, rect.y + rect.height / 2},
                           rect.height);
}

cv::Point2f findGaze(cv::Mat& frame, int threshold) {
    selectPupil(frame, frame, threshold);
    return findCenter(frame);
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
            selectPupil(frame, tempFrame, i);

            const auto contour = getMaxAreaContour(tempFrame);

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
            for (int i = 1; i < mPoints.size(); ++i) {
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

class MovementDetector {
public:
    cv::Point detect(const cv::Point& point) {
        // input point is invalid, return zero vector
        if (point.x < 0 && point.y < 0) {
            return cv::Point{};
        }

        // input point is a first point, return zero vector
        if (mPrevPoint.x == INVAL && mPrevPoint.y == INVAL) {
            mPrevPoint = point;
            return cv::Point{};
        }

        const auto vec = cv::Point{point.x - mPrevPoint.x, point.y - mPrevPoint.y};

        const auto abs_x = abs(vec.x);
        const auto abs_y = abs(vec.y);

        if ((abs_x < STAB_BOT && abs_y < STAB_BOT) || (abs_x > STAB_TOP && abs_y > STAB_TOP)) {
            // probably shaking or jumping, return zero vector
            return cv::Point{};
        }

        mPrevPoint = point;

        return vec;
    }

private:
    static constexpr int INVAL = -1;
    static constexpr int STAB_BOT = 2;
    static constexpr int STAB_TOP = 70;

    cv::Point mPrevPoint{INVAL, INVAL};
};

bool inRange(int num, int range) { return abs(num) < range; }

float lowerBy(float d) {
    const float a4 = 0.0554691;
    const float a3 = -0.49279;
    const float a2 = 1.25006;
    const float a1 = -1.81274;
    const float a0 = 2;

    return a4 * pow(d, 4) + a3 * pow(d, 3) + a2 * pow(d, 2) + a1 * d + a0;
}

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

    const int circSize = 50;
    const cv::Point2f circCenter{circSize / 2, circSize / 2};

    cv::Mat axis{circSize, circSize, CV_8UC3, cv::Scalar{255, 255, 255}};

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
                const auto leftEyePoint =
                    cv::Point2f{faceShape.part(39).x(), faceShape.part(39).y()};
                const auto rightEyePoint =
                    cv::Point2f{faceShape.part(42).x(), faceShape.part(42).y()};

                const auto midPoint = (leftEyePoint + rightEyePoint) / 2;

                const auto leftRoiRect =
                    cv::Rect{midPoint.x, 0, frame.cols - midPoint.x, frame.rows};
                const auto rightRoiRect = cv::Rect{0, 0, midPoint.x, frame.rows};

                mask = cv::Mat::zeros(frame.size(), frame.type());

                appplyEyeMask(mask, faceShape, LEFT_POINTS);
                appplyEyeMask(mask, faceShape, RIGHT_POINTS);

                cv::dilate(mask, mask, kernel, cv::Point{-1, -1}, 2);

                // apply obtained mask to the image
                cv::bitwise_and(frame, mask, frame);

                // convert all 0,0,0 pixels of the frame to 255,255,255,
                // so eyeball is the only dark part left
                mask = (frame == 0);
                frame.setTo(255, mask);

                auto rightRoi = frame(rightRoiRect);
                auto leftRoi = frame(leftRoiRect);

                if (!calibrationComplete) {
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
                } else {
                    std::future<cv::Point2f> rightFuture = std::async(
                        std::launch::async, &findGaze, std::ref(rightRoi), rightThreshold);

                    const auto leftGaze = findGaze(leftRoi, leftThreshold);
                    const auto rightGaze = rightFuture.get() + cv::Point2f{midPoint.x, 0};

                    const auto gaze = (leftGaze + rightGaze) * 0.5;

                    cv::circle(colorFrame, gaze, 2, cv::Scalar(255, 0, 0), 1, cv::LINE_8);
                    cv::circle(colorFrame, midPoint, 2, cv::Scalar(0, 255, 0), 1, cv::LINE_8);
                    //                    cv::line(colorFrame, leftGaze, rightGaze, cv::Scalar(0, 0,
                    //                    255)); cv::line(colorFrame, leftEyePoint, rightEyePoint,
                    //                    cv::Scalar(255, 0, 0));

                    const auto diff = gaze - midPoint;

                    printMask(colorFrame, faceShape);

                    if (inRange(diff.x, GAZE_RANGE) && inRange(diff.y, GAZE_RANGE)) {
                        axis = cv::Scalar{255, 255, 255};

                        cv::circle(axis, circCenter, circSize / 2, cv::Scalar(0, 0, 0), 1,
                                   cv::LINE_8);
                        cv::drawMarker(axis, circCenter, cv::Scalar(0, 0, 0), cv::MARKER_CROSS,
                                       circSize);
                        cv::circle(axis, diff + circCenter, 1, cv::Scalar(0, 0, 255), 1,
                                   cv::LINE_8);

                        std::cout << "eye diff is " << diff.x << ", " << diff.y << std::endl;
                    }
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
        cv::imshow(WIN, axis);

        // wait (10ms) for a key to be pressed
        if (cv::waitKey(33) == 'q') break;
    }

    return 0;
}
