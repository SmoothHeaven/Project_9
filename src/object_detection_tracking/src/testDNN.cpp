#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp> 
#include <vector>


int main() {
    // Open the default camera
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    // Set the camera properties
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);
    if (!cap.isOpened()) {
        std::cout << "Failed to open camera!" << std::endl;
        return -1;
    }

    // Load the pre-trained face detection model
    cv::dnn::Net net = cv::dnn::readNetFromCaffe("DNN/deploy.prototxt",
                                                "DNN/res10_300x300_ssd_iter_140000.caffemodel");
    if (net.empty()) {
        std::cout << "Failed to load face detection model!" << std::endl;
        return -1;
    }

    // Create a tracker
    cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();

    // Main loop
    cv::Mat frame;
    bool tracking = false;
    cv::Rect bbox;
    while (cap.read(frame)) {
        // Check if frame is empty
        if (frame.empty()) {
            std::cout << "Frame is empty!" << std::endl;
            continue;
        }

        // If not tracking, perform face detection
        if (!tracking) {
            // Create a blob from the input frame
            cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(104, 177, 123));

            // Set the blob as input to the network
            net.setInput(blob);

            // Forward pass through the network
            cv::Mat detections = net.forward();

            // Process the detections
            for (int i = 0; i < detections.size[2]; i++) {
                float confidence = detections.at<float>(cv::Vec<int, 4>(0, 0, i, 2));

                // Draw bounding box if confidence is above a threshold
                if (confidence > 0.5) {
                    int x1 = static_cast<int>(detections.at<float>(cv::Vec<int, 4>(0, 0, i, 3)) * frame.cols);
                    int y1 = static_cast<int>(detections.at<float>(cv::Vec<int, 4>(0, 0, i, 4)) * frame.rows);
                    int x2 = static_cast<int>(detections.at<float>(cv::Vec<int, 4>(0, 0, i, 5)) * frame.cols);
                    int y2 = static_cast<int>(detections.at<float>(cv::Vec<int, 4>(0, 0, i, 6)) * frame.rows);

                    bbox = cv::Rect(x1, y1, x2 - x1, y2 - y1);
                    tracking = true;
                    tracker->init(frame, bbox);
                    break;
                }
            }
        } else {
            // Check if bbox is initialized
            if (bbox.width <= 0 || bbox.height <= 0) {
                std::cout << "Bounding box is not initialized!" << std::endl;
                tracking = false;
                continue;
            }
            // Update the tracker
            bool success = tracker->update(frame, bbox);

            // If tracking is successful, draw bounding box
            if (success) {
                cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);
            } else {
                tracker = cv::TrackerKCF::create();
                tracking = false;
            }
        }

        // Display the frame with bounding boxes
        cv::imshow("Face Detection and Tracking", frame);

        // Exit if 'q' is pressed
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release the camera and destroy windows
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
