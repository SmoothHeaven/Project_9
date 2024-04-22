#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>


#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"

#include <geometry_msgs/msg/point32.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>


class DetectionAndTracking {
public:
    DetectionAndTracking(const std::string& videoPath, const std::string& modelPath, const std::string& configPath)
        : videoPath(videoPath)
    {
        // Initialize ROS2
        rclcpp::init(0, nullptr);
        node = rclcpp::Node::make_shared("detection_and_tracking_node");
        publisher = node->create_publisher<sensor_msgs::msg::Image>("image", 30);
        // Initialize the PointCloud2 publisher
        pointCloudPublisher = node->create_publisher<sensor_msgs::msg::PointCloud2>("object_location", 10);
        // Load the DNN model
        net = cv::dnn::readNetFromCaffe(configPath, modelPath);
        // Set video resolution to 640x480 (480p)
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        // Set video frame rate to 30 fps
        cap.set(cv::CAP_PROP_FPS, 30);
    }

    void run() {
        std::thread captureThread(&DetectionAndTracking::videoCapture, this);
        std::thread preprocessingThread(&DetectionAndTracking::preprocessingDNN, this);
        std::thread detectionThread(&DetectionAndTracking::faceDetectionDNN, this);
        std::thread outputThread(&DetectionAndTracking::videoOutput, this);

        captureThread.join();
        preprocessingThread.join();
        detectionThread.join();
        outputThread.join();
    }

private:
    std::mutex frameQueueMutex; // Protects frameQueue
    std::mutex detectedQueueMutex; // Protects detectedQueue
    std::mutex frameBlobQueueMutex; // Protects frameBlobQueue

    std::condition_variable frameReady;
    std::condition_variable processedFrameReady;
    std::condition_variable detectedFrameReady;

    std::queue<cv::Mat> frameQueue;
    std::queue<cv::Mat> detectedQueue;
    std::queue<std::pair<cv::Mat, cv::Mat>> frameBlobQueue;

    cv::dnn::Net net;
    cv::VideoCapture cap;
    std::string videoPath;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointCloudPublisher;
    std::shared_ptr<rclcpp::Node> node;

    cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
    bool stopProcessing = false;
    bool tracking = false;
    cv::Rect bbox;

    // Set the orientation
    tf2::Quaternion q;


    void videoCapture() {
        cap.open(videoPath, cv::CAP_V4L2);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open video file" << std::endl;
            return;
        }

        cv::Mat frame;
        while (rclcpp::ok() && cap.read(frame)) {
            cv::Mat D = (cv::Mat_<double>(5,1) << 0.07414285932675879, -0.2324638202268596, 0.007229804062587446, 0.0108458569817934, 0.0);
            cv::Mat K = (cv::Mat_<double>(3,3) << 634.7598716003646, 0.0, 327.8325317106525, 0.0, 638.956081473938, 241.97666841657622, 0.0, 0.0, 1.0);
            cv::Mat undistortedFrame;

            cv::undistort(frame, undistortedFrame, K, D);
            std::lock_guard<std::mutex> lock(frameQueueMutex);
            frameQueue.push(undistortedFrame);
            frameReady.notify_one();
            publishTransform();
        }
    }

    void preprocessingDNN() {
        while (rclcpp::ok()) {
            std::unique_lock<std::mutex> lock(frameQueueMutex);
            frameReady.wait(lock, [this] { return !frameQueue.empty(); });

            cv::Mat frame = frameQueue.front();
            frameQueue.pop();

            // DNN preprocessing
            cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(104, 177, 123));
            {
                std::lock_guard<std::mutex> lock(frameBlobQueueMutex);
                frameBlobQueue.push({blob, frame});
            }
            processedFrameReady.notify_one();
        }
    }

    void faceDetectionDNN() {
        while (rclcpp::ok()) {
            std::unique_lock<std::mutex> lock(frameBlobQueueMutex);
            processedFrameReady.wait(lock, [this] { return !frameBlobQueue.empty(); });
            std::pair<cv::Mat, cv::Mat> frameBlob = frameBlobQueue.front();
            frameBlobQueue.pop();

            cv::Mat blob = frameBlob.first;
            cv::Mat originalFrame = frameBlob.second;
            if (!tracking) {
                net.setInput(blob);
                cv::Mat detections = net.forward();

                for (int i = 0; i < detections.size[2]; i++) {
                    float confidence = detections.at<float>(cv::Vec<int, 4>(0, 0, i, 2));

                    // Draw bounding box if confidence is above a threshold
                    if (confidence > 0.5) {
                        int x1 = static_cast<int>(detections.at<float>(cv::Vec<int, 4>(0, 0, i, 3)) * originalFrame.cols);
                        int y1 = static_cast<int>(detections.at<float>(cv::Vec<int, 4>(0, 0, i, 4)) * originalFrame.rows);
                        int x2 = static_cast<int>(detections.at<float>(cv::Vec<int, 4>(0, 0, i, 5)) * originalFrame.cols);
                        int y2 = static_cast<int>(detections.at<float>(cv::Vec<int, 4>(0, 0, i, 6)) * originalFrame.rows);

                        bbox = cv::Rect(x1, y1, x2 - x1, y2 - y1);
                        tracking = true;
                        tracker->init(originalFrame, bbox);
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
                bool success = tracker->update(originalFrame, bbox);

                // If tracking is successful, draw bounding box
                if (success) {
                    cv::rectangle(originalFrame, bbox, cv::Scalar(0, 255, 0), 2);
                    // Convert the bounding box center to a Point32 message
                    geometry_msgs::msg::Point32 location;
                    location.x = bbox.x + bbox.width / 2.0;
                    location.y = bbox.y + bbox.height / 2.0;
                    // Visualize the object location
                    visualizeObjectLocation(location);
                } else {
                    tracker = cv::TrackerKCF::create();
                    tracking = false;
                }
            }
            // Add the frame to the detected queue regardless of face detection result

            std::lock_guard<std::mutex> lock3(detectedQueueMutex);
            detectedQueue.push(originalFrame);
            detectedFrameReady.notify_one();
        }
    }

    void videoOutput() {
        while (rclcpp::ok()) {
            std::unique_lock<std::mutex> lock(detectedQueueMutex);
            detectedFrameReady.wait(lock, [this] { return !detectedQueue.empty(); });

            cv::Mat frame = detectedQueue.front();
            detectedQueue.pop();

            // Convert the frame to a ROS2 message
            auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();

            // Publish the message
            publisher->publish(*msg);

            // Display the frame
            cv::imshow("Video Output", frame);

            rclcpp::spin_some(node);
        }
        std::cout << "Video output ended." << std::endl;
    }

    void visualizeObjectLocation(const geometry_msgs::msg::Point32& location) {
        // Create a PointCloud2 message
        auto cloud_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
        // Set the header
        cloud_msg->header.stamp = node->now();
        cloud_msg->header.frame_id = "camera";  // Replace with your frame ID

        // Set the fields
        cloud_msg->height = 1;
        cloud_msg->width = 1;
        sensor_msgs::PointCloud2Modifier pcd_modifier(*cloud_msg);
        pcd_modifier.setPointCloud2FieldsByString(1, "xyz");

        // Set the data
        sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");

        *iter_x = (location.x-320)*0.001;
        *iter_y = -(location.y-240)*0.001;
        *iter_z = 1;

        // Publish the message
        pointCloudPublisher->publish(std::move(cloud_msg));
    }

    void publishTransform() {
        // Create a TransformStamped message
        geometry_msgs::msg::TransformStamped transform_stamped;

        // Set the header
        transform_stamped.header.stamp = node->now();
        transform_stamped.header.frame_id = "vx300s/gripper_link";  // Replace with your target frame ID
        transform_stamped.child_frame_id = "camera";  // Replace with your source frame ID

        // Set the position
        transform_stamped.transform.translation.x = 0.0;
        transform_stamped.transform.translation.y = 0.0;
        transform_stamped.transform.translation.z = 0.0;

        q.setRPY(M_PI/2, 0, M_PI/2);  // Replace with your desired roll, pitch, yaw
        transform_stamped.transform.rotation.x = q.x();
        transform_stamped.transform.rotation.y = q.y();
        transform_stamped.transform.rotation.z = q.z();
        transform_stamped.transform.rotation.w = q.w();

        // Create a TransformBroadcaster
        static tf2_ros::TransformBroadcaster br(node);

        // Publish the transform
        br.sendTransform(transform_stamped);
    }




};

int main() {
    std::string videoPath = "/dev/video0";
    std::string modelPath = "DNN/res10_300x300_ssd_iter_140000.caffemodel";
    std::string configPath = "DNN/deploy.prototxt";

    DetectionAndTracking detectionAndTracking(videoPath, modelPath, configPath);
    detectionAndTracking.run();

    // Shutdown ROS2
    rclcpp::shutdown();

    return 0;
}
