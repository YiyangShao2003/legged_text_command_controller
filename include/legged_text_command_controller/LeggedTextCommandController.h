//
// Created by qiayuanl on 9/1/24.
//

#pragma once
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <realtime_tools/realtime_box.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <deque>
#include <memory>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "legged_controllers/ControllerBase.h"

namespace legged {

struct Observations {
    vector_t observations;
    vector_t currentProprioception;
};

struct ModelReturns {
    vector_t actions;
    vector_t latents;
};

class LeggedTextCommandController : public ControllerBase {
  using tensor_element_t = float;
  using Twist = geometry_msgs::msg::TwistStamped;

 public:
  ~LeggedTextCommandController();

  controller_interface::return_type update(const rclcpp::Time& time, const rclcpp::Duration& period) override;

  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 protected:
  virtual Observations getObservations();
  virtual vector_t updateObsBuffer(const vector_t& observations);
  virtual ModelReturns playModel(const Observations& obsStructure) const;
  virtual vector_t remapJointOrder(const vector_t & policy_vector) const;

  // Helper functions
  void updateLatestObservation(const Observations& obs);
  bool retrieveLatestAction(ModelReturns& model_returns_struct);

  // Function executed by the model thread
  void modelThreadFunction();

  // Thread for model inference
  std::thread modelThread_;

  // Synchronization primitives
  std::mutex modelMutex_;
  std::condition_variable modelCv_;
  std::atomic<bool> modelThreadRunning_{false};

  // **Single Shared Observation Variable**
  Observations latestObservation_;
  bool newObservationAvailable_{false};
  // **Single Shared Action Variable**
  // Eigen::VectorXd latestAction_;
  ModelReturns latestModelReturns_;
  std::mutex actionMutex_;
  bool actionAvailable_{false};

  // Obs recording
  bool logObs_;
  std::ofstream obsJsonFile_;
  std::string obsFilePath_;
  std::mutex obsFileMutex_;

  // Latent vector recording
  bool logLatent_;
  std::ofstream latentJsonFile_;
  std::string latentFilePath_;
  std::mutex latentFileMutex_;
  size_t latentSize_;
  scalar_t latentRecordTime_;
  rclcpp::Time latentLoggingStartTime_;  
  bool latentLoggingActive_ = false;     


  // Onnx
  std::shared_ptr<Ort::Env> onnxEnvPrt_;
  std::unique_ptr<Ort::Session> sessionPtr_;
  std::vector<const char*> inputNames_;
  std::vector<const char*> outputNames_;
  std::vector<std::vector<int64_t>> inputShapes_;
  std::vector<std::vector<int64_t>> outputShapes_;

  vector_t lastActions_;
  std::vector<std::string> jointNameInPolicy_;

  // Observation
  std::vector<std::string> obsWithHistoryNames_;
  std::vector<std::string> obsWithoutHistoryNames_;
  size_t observationSize_{0};
  std::vector<size_t> obsIndexMap_;

  // Observation History Buffer
  size_t obsHistoryLength_;
  size_t obsBufferLength_;
  size_t obsWithHistorySize_;
  std::deque<vector_t> observationBuffer_;

  // Command
  vector_t command_;
  std::vector<vector_t> commandList_;
  std::vector<scalar_t> commandDurationList_;
  rclcpp::Time lastCommandTime_;
  size_t commandIndex_{0};

  // Action
  scalar_t actionScale_{0};
  std::string actionType_{"position_absolute"};
  vector_t desiredPosition_;

  // Time
  bool firstUpdate_{false};
  scalar_t policyFrequency_{50.};
  rclcpp::Time lastPlayTime_;

  // Command Interface
  rclcpp::Subscription<Twist>::SharedPtr velocitySubscriber_;
  realtime_tools::RealtimeBox<std::shared_ptr<Twist>> receivedVelocityMsg_;

  std::shared_ptr<rclcpp::Publisher<std_msgs::msg::Float64MultiArray>> publisher_;
  std::shared_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>> publisherRealtime_;
};

}  // namespace legged
