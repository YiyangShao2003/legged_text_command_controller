//
// Created by qiayuanl on 9/1/24.
//

#include <memory>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>

#include "legged_text_command_controller/LeggedTextCommandController.h"

using json = nlohmann::json;
namespace nlohmann {
    template <>
    struct adl_serializer<Eigen::VectorXd> {
        static void from_json(const json& j, Eigen::VectorXd& vec) {
            std::vector<double> tmp = j.get<std::vector<double>>();
            vec = Eigen::Map<Eigen::VectorXd>(tmp.data(), tmp.size());
        }

        static void to_json(json& j, const Eigen::VectorXd& vec) {
            std::vector<double> tmp(vec.data(), vec.data() + vec.size());
            j = tmp;
        }
    };
}  // namespace nlohmann

namespace legged {

// Enqueue new observations by updating the latestObservation_
void LeggedTextCommandController::updateLatestObservation(const Observations& obs) {
  std::lock_guard<std::mutex> lock(modelMutex_);
  latestObservation_ = obs;
  newObservationAvailable_ = true;
  modelCv_.notify_one();
}

// Retrieve the latest action produced by the model thread
bool LeggedTextCommandController::retrieveLatestAction(Eigen::VectorXd& action) {
  std::lock_guard<std::mutex> lock(actionMutex_);
  if (actionAvailable_) {
    action = latestAction_;
    actionAvailable_ = false;
    return true;
  }
  return false;
}
void LeggedTextCommandController::modelThreadFunction() {
  while (modelThreadRunning_) {
    Observations currentObs;
    {
      std::unique_lock<std::mutex> lock(modelMutex_);
      modelCv_.wait(lock, [&]() { return newObservationAvailable_ || !modelThreadRunning_; });
      if (!modelThreadRunning_) {
        break;
      }
      if (newObservationAvailable_) {
        currentObs = latestObservation_;
        newObservationAvailable_ = false;
      } else {
        continue; // Spurious wake-up or shutdown signal
      }
    }

    // Run model inference
    Eigen::VectorXd action;
    try {
      action = playModel(currentObs);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_node()->get_logger(), "Model inference failed: %s", e.what());
      continue; // Skip to the next iteration
    }

    {
      std::lock_guard<std::mutex> lock(actionMutex_);
      latestAction_ = action;
      actionAvailable_ = true;
    }
  }

  RCLCPP_INFO(get_node()->get_logger(), "Model thread has stopped.");
}

LeggedTextCommandController::~LeggedTextCommandController() {
  modelThreadRunning_ = false;
  modelCv_.notify_one();
  if (modelThread_.joinable()) {
    modelThread_.join();
  }

  if (logObs_ == true && obsJsonFile_.is_open()) {
    {
      std::lock_guard<std::mutex> lock(obsFileMutex_);
      obsJsonFile_ << std::endl << "  ]" << std::endl << "}" << std::endl;
      obsJsonFile_.close();
    }
  }
}

void printInputsOutputs(const std::vector<const char*>& inputNames, const std::vector<std::vector<int64_t>>& inputShapes,
                        const std::vector<const char*>& outputNames, const std::vector<std::vector<int64_t>>& outputShapes) {
  std::cout << "Inputs:" << std::endl;
  for (size_t i = 0; i < inputNames.size(); ++i) {
    std::cout << "  Input " << i << ": " << inputNames[i] << " - Shape: [";
    for (size_t j = 0; j < inputShapes[i].size(); ++j) {
      std::cout << inputShapes[i][j];
      if (j < inputShapes[i].size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }

  std::cout << "Outputs:" << std::endl;
  for (size_t i = 0; i < outputNames.size(); ++i) {
    std::cout << "  Output " << i << ": " << outputNames[i] << " - Shape: [";
    for (size_t j = 0; j < outputShapes[i].size(); ++j) {
      std::cout << outputShapes[i][j];
      if (j < outputShapes[i].size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
}

controller_interface::return_type LeggedTextCommandController::update(const rclcpp::Time& time, const rclcpp::Duration& period) {
  if (ControllerBase::update(time, period) != controller_interface::return_type::OK) {
    return controller_interface::return_type::ERROR;
  }

  std::shared_ptr<Twist> lastCommandMsg;
  receivedVelocityMsg_.get(lastCommandMsg);
  if (time - lastCommandMsg->header.stamp > std::chrono::milliseconds{static_cast<int>(0.5 * 1000.0)}) {
    lastCommandMsg->twist.linear.x = 0.0;
    lastCommandMsg->twist.linear.y = 0.0;
    lastCommandMsg->twist.angular.z = 0.0;
  }
  
  if (commandIndex_ < commandList_.size()) {
    double elapsed_seconds = (time - lastCommandTime_).seconds();
    // Check if the current command duration has elapsed
    if (elapsed_seconds >= commandDurationList_[commandIndex_]) {
      commandIndex_++; // Move obsJsonFile_to the next command
      if (commandIndex_ < commandList_.size()) {
        command_ = commandList_[commandIndex_]; // Update to the next command
        lastCommandTime_ = time; // Reset the timer for the new command
      } else {
        // Optionally reset to the first command or stop updating commands
        commandIndex_ = 0;
        command_ = commandList_[commandIndex_];
        lastCommandTime_ = time;
      }
      RCLCPP_INFO(get_node()->get_logger(), "Switched to command index %zu", commandIndex_);
    }
  }

  if (firstUpdate_ || (time - lastPlayTime_).seconds() >= 1. / policyFrequency_) {
    auto obs_struct = getObservations();
    updateLatestObservation(obs_struct);
      
    // Serialize and write obs_struct to JSON
    if (logObs_ == true) {
      std::lock_guard<std::mutex> lock(obsFileMutex_);
      json j_obs;
      j_obs["timestamp"] = time.seconds(); // Assuming you want to record the timestamp
      j_obs["currentProprioception"] = obs_struct.currentProprioception;
      
      // Add comma if not the first entry
      if (obsJsonFile_.tellp() > 0) {
        obsJsonFile_ << "," << std::endl;
      }
      
      obsJsonFile_ << "    " << j_obs.dump(4); // Pretty print with 4-space indentation
    }
  }

  // Retrieve actions from the model thread if available
  Eigen::VectorXd latestAction;
  if (retrieveLatestAction(latestAction)) {
    lastActions_ = latestAction;

    if (actionType_ == "position_absolute") {
      for (Eigen::Index i = 0; i < lastActions_.size(); ++i) {
        size_t hardwareIndex = jointIndexMap_[jointNameInPolicy_[i]];
        desiredPosition_[hardwareIndex] = lastActions_[i] * actionScale_ + defaultPosition_[hardwareIndex];
      }
    } else if (actionType_ == "position_relative") {
      for (Eigen::Index i = 0; i < lastActions_.size(); ++i) {
        size_t hardwareIndex = jointIndexMap_[jointNameInPolicy_[i]];
        desiredPosition_[hardwareIndex] += lastActions_[i] * actionScale_;
      }
    } else if (actionType_ == "position_delta") {
      const vector_t currentPosition = leggedModel_->getLeggedModel()->getGeneralizedPosition().tail(lastActions_.size());
      for (Eigen::Index i = 0; i < lastActions_.size(); ++i) {
        size_t hardwareIndex = jointIndexMap_[jointNameInPolicy_[i]];
        desiredPosition_[hardwareIndex] = currentPosition[hardwareIndex] + lastActions_[i] * actionScale_;
      }
    }
    setPositions(desiredPosition_);

    firstUpdate_ = false;
    lastPlayTime_ = time;

    if (publisherRealtime_->trylock()) {
      auto& msg = publisherRealtime_->msg_;
      msg.data.clear();
      // for (const double i : obs_struct.currentProprioception) {
      //   msg.data.push_back(i);
      // }
      for (const double i : lastActions_) {
        msg.data.push_back(i);
      }
      publisherRealtime_->unlockAndPublish();
    }
  }

  return controller_interface::return_type::OK;
}

controller_interface::CallbackReturn LeggedTextCommandController::on_configure(const rclcpp_lifecycle::State& previous_state) {
  if (ControllerBase::on_configure(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }

  // Onnx
  std::string policyPath{};
  get_node()->get_parameter("policy.path", policyPath);
  get_node()->get_parameter("policy.frequency", policyFrequency_);
  onnxEnvPrt_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "LeggedTextCommandController");
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetInterOpNumThreads(1);
  sessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPrt_, policyPath.c_str(), sessionOptions);
  inputNames_.clear();
  outputNames_.clear();
  inputShapes_.clear();
  outputShapes_.clear();
  Ort::AllocatorWithDefaultOptions allocator;
  for (size_t i = 0; i < sessionPtr_->GetInputCount(); i++) {
    inputNames_.push_back(sessionPtr_->GetInputName(i, allocator));
    inputShapes_.push_back(sessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < sessionPtr_->GetOutputCount(); i++) {
    outputNames_.push_back(sessionPtr_->GetOutputName(i, allocator));
    outputShapes_.push_back(sessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  printInputsOutputs(inputNames_, inputShapes_, outputNames_, outputShapes_);

  const size_t numJoints = leggedModel_->getLeggedModel()->getJointNames().size();

  jointNameInPolicy_ = get_node()->get_parameter("policy.joint_names").as_string_array();
  if (jointNameInPolicy_.size() != numJoints) {
    RCLCPP_ERROR(get_node()->get_logger(), "joint_names size is not equal to joint size.");
    return controller_interface::CallbackReturn::ERROR;
  }
  auto jointNameObsOrder_ = get_node()->get_parameter("policy.observations.obs_order_joint_names").as_string_array();
  if (jointNameObsOrder_.size() != jointNameInPolicy_.size()) {
    RCLCPP_ERROR(get_node()->get_logger(), "obs_order_joint_names size (%zu) does not match policy joint_names size (%zu).",
                  jointNameObsOrder_.size(), jointNameInPolicy_.size());
    return controller_interface::CallbackReturn::ERROR;
  }
  std::unordered_map<std::string, size_t> policyIndexMap_;
  for (size_t i = 0; i < jointNameInPolicy_.size(); ++i) {
    policyIndexMap_[jointNameInPolicy_[i]] = i;
  }
  obsIndexMap_.reserve(jointNameObsOrder_.size());
  for (const auto & joint_name : jointNameObsOrder_) {
    auto it = policyIndexMap_.find(joint_name);
    if (it != policyIndexMap_.end()) {
      obsIndexMap_.push_back(it->second);
    } else {
      RCLCPP_ERROR(get_node()->get_logger(), "Joint name '%s' in obs_order_joint_names not found in policy joint_names.",
                    joint_name.c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
  }


  lastActions_.setZero(numJoints);
  command_ = vector_t::Zero(512);
  RCLCPP_INFO_STREAM(rclcpp::get_logger("LeggedTextCommandController"), "Load Onnx model from" << policyPath << " successfully !");

  // Command
  const auto command_file_path = get_node()->get_parameter("command.command_file_path").as_string();
  std::ifstream commandFile(command_file_path);
  if (!commandFile) {
      RCLCPP_ERROR(get_node()->get_logger(), "Failed to open command file: %s", command_file_path.c_str());
      return controller_interface::CallbackReturn::ERROR;
  }

  json dataSet;
  commandFile >> dataSet;

  // For the observations JSON file
  // Determine the directory of the command file
  logObs_ = get_node()->get_parameter("log.log_obs").as_bool();
  if (logObs_ == true) {
    std::filesystem::path cmdPath(command_file_path);
    std::filesystem::path dirPath = cmdPath.parent_path();
    
    // Set the observations JSON file path
    obsFilePath_ = (dirPath / "observations.json").string();

    // Open the JSON file for writing
    obsJsonFile_.open(obsFilePath_, std::ios::out | std::ios::trunc);
    if (!obsJsonFile_.is_open()) {
      RCLCPP_ERROR(get_node()->get_logger(), "Failed to open observations JSON file: %s", obsFilePath_.c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
    std::cout << "Observations JSON file path: " << obsFilePath_ << std::endl;
    obsJsonFile_ << "{" << std::endl;
    obsJsonFile_ << "  \"observations\": [" << std::endl;
  }

  // Retrieve tasks and durations
  auto tasks = get_node()->get_parameter("command.task").as_string_array();
  auto task_durations = get_node()->get_parameter("command.task_duration").as_double_array();

  // Helper lambda to trim whitespace
  auto trim = [](std::string& s) -> void {
      const std::string whitespace = " \t\n\r";
      size_t start = s.find_first_not_of(whitespace);
      size_t end = s.find_last_not_of(whitespace);
      if (start != std::string::npos && end != std::string::npos)
          s = s.substr(start, end - start + 1);
      else
          s.clear();
  };

  // Process each task
  for (size_t i = 0; i < tasks.size(); ++i) {
      const std::string& task_str = tasks[i];
      double duration = (i < task_durations.size()) ? task_durations[i] : 0.0;

      // Split task string into sub-tasks
      std::vector<std::string> sub_tasks;
      std::stringstream ss(task_str);
      std::string sub_task;
      while (std::getline(ss, sub_task, ';')) {
          trim(sub_task);
          if (!sub_task.empty()) {
              sub_tasks.push_back(sub_task);
          }
      }

      if (sub_tasks.empty()) {
          RCLCPP_ERROR(get_node()->get_logger(), "No valid sub-tasks in: '%s'", task_str.c_str());
          return controller_interface::CallbackReturn::ERROR;
      }

      vector_t combined_embedding;
      double total_weight = 0.0;
      size_t embedding_size = 0;

      for (const auto& st : sub_tasks) {
          // Split sub-task into task name and weight
          size_t colon_pos = st.find(':');
          if (colon_pos == std::string::npos) {
              RCLCPP_ERROR(get_node()->get_logger(), "Invalid sub-task format: '%s'. Expected 'task_name:weight'.", st.c_str());
              return controller_interface::CallbackReturn::ERROR;
          }

          std::string task_name = st.substr(0, colon_pos);
          std::string weight_str = st.substr(colon_pos + 1);
          trim(task_name);
          trim(weight_str);

          // Convert weight to double with default value 1.0
          double weight = 1.0;
          if (!weight_str.empty()) {
              try {
                  weight = std::stod(weight_str);
              } catch (...) {
                  RCLCPP_ERROR(get_node()->get_logger(), "Invalid weight '%s' for task '%s'.", weight_str.c_str(), task_name.c_str());
                  return controller_interface::CallbackReturn::ERROR;
              }
          }

          if (!dataSet.contains(task_name) || !dataSet[task_name].contains("embedding") || !dataSet[task_name]["embedding"].is_array() || dataSet[task_name]["embedding"].empty()) {
              RCLCPP_ERROR(get_node()->get_logger(), "Invalid or missing embedding for task '%s'.", task_name.c_str());
              return controller_interface::CallbackReturn::ERROR;
          }

          std::vector<double> embedding_tmp = dataSet[task_name]["embedding"][0].get<std::vector<double>>();
          if (embedding_tmp.empty()) {
              RCLCPP_ERROR(get_node()->get_logger(), "Empty embedding for task '%s'.", task_name.c_str());
              return controller_interface::CallbackReturn::ERROR;
          }

          if (combined_embedding.size() == 0) {
              embedding_size = embedding_tmp.size();
              combined_embedding = vector_t::Zero(embedding_size);
          } else if (embedding_tmp.size() != embedding_size) {
              RCLCPP_ERROR(get_node()->get_logger(), "Embedding size mismatch for task '%s'. Expected %zu but got %zu.", task_name.c_str(), embedding_size, embedding_tmp.size());
              return controller_interface::CallbackReturn::ERROR;
          }

          Eigen::Map<Eigen::VectorXd> embedding(embedding_tmp.data(), embedding_tmp.size());
          combined_embedding += embedding * weight;
          total_weight += weight;

          std::string caption = dataSet[task_name].value("caption", "No Caption");
          RCLCPP_INFO(get_node()->get_logger(), "Command: %s (Weight: %.2f)", caption.c_str(), weight);
      }

      if (total_weight != 0.0 && std::abs(total_weight - 1.0) > 1e-6) {
          combined_embedding /= total_weight;
          RCLCPP_INFO(get_node()->get_logger(), "Normalized combined embedding by total weight: %.6f", total_weight);
      }

      commandList_.emplace_back(combined_embedding);
      commandDurationList_.emplace_back(duration);
  }

  // Validate command lists
  if (commandList_.size() != commandDurationList_.size()) {
      RCLCPP_ERROR(get_node()->get_logger(), "commandList_ and commandDurationList_ size mismatch.");
      return controller_interface::CallbackReturn::ERROR;
  }

  RCLCPP_INFO(get_node()->get_logger(), "Loaded %zu commands successfully.", commandList_.size());
  // Observation
  obsWithHistoryNames_ = get_node()->get_parameter("policy.observations.with_history").as_string_array();
  obsWithoutHistoryNames_ = get_node()->get_parameter("policy.observations.without_history").as_string_array();
  auto obsHistoryTime_ = get_node()->get_parameter("policy.observations.history_time").as_double();
  obsHistoryLength_ = get_node()->get_parameter("policy.observations.history_length").as_int();

  obsBufferLength_ = obsHistoryTime_ * int(policyFrequency_);

  observationSize_ = 0;

  for (const auto& name : obsWithHistoryNames_) {
    if (name == "base_lin_vel") {
      observationSize_ += 3;
    } else if (name == "base_ang_vel") {
      observationSize_ += 3;
    } else if (name == "projected_gravity") {
      observationSize_ += 3;
    } else if (name == "joint_positions") {
      observationSize_ += numJoints;
    } else if (name == "joint_velocities") {
      observationSize_ += numJoints;
    } else if (name == "last_action") {
      observationSize_ += numJoints;
    }
  }
  obsWithHistorySize_ = observationSize_;

  observationSize_ *= obsHistoryLength_;

  for (const auto& name : obsWithoutHistoryNames_) {
    if (name == "text_command") {
      observationSize_ += 512;
    }
  }

  // Action
  get_node()->get_parameter("policy.action_scale", actionScale_);
  get_node()->get_parameter("policy.action_type", actionType_);
  if (actionType_ != "position_absolute" && actionType_ != "position_relative" && actionType_ != "position_delta") {
    RCLCPP_ERROR(get_node()->get_logger(), "Unknown action type: %s", actionType_.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  desiredPosition_.setZero(numJoints);

  // ROS Interface
  velocitySubscriber_ =
      get_node()->create_subscription<Twist>("/cmd_vel", rclcpp::SystemDefaultsQoS(), [this](const std::shared_ptr<Twist> msg) -> void {
        if ((msg->header.stamp.sec == 0) && (msg->header.stamp.nanosec == 0)) {
          RCLCPP_WARN_ONCE(get_node()->get_logger(),
                           "Received TwistStamped with zero timestamp, setting it to current "
                           "time, this message will only be shown once");
          msg->header.stamp = get_node()->get_clock()->now();
        }
        receivedVelocityMsg_.set(msg);
      });

  publisher_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>("~/policy_io", rclcpp::SystemDefaultsQoS());
  publisherRealtime_ = std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(publisher_);

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn LeggedTextCommandController::on_activate(const rclcpp_lifecycle::State& previous_state) {
  if (ControllerBase::on_activate(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }
  firstUpdate_ = true;
  receivedVelocityMsg_.set(std::make_shared<Twist>());
  lastActions_.setZero();
  desiredPosition_ = defaultPosition_;

  lastCommandTime_ = get_node()->now();
  commandIndex_ = 0;
  if (!commandList_.empty()) {
    command_ = commandList_[commandIndex_];
    RCLCPP_INFO(get_node()->get_logger(), "Activated with initial command index %zu", commandIndex_);
  }

  // Start the model thread
  modelThreadRunning_ = true;
  modelThread_ = std::thread(&LeggedTextCommandController::modelThreadFunction, this);

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn LeggedTextCommandController::on_deactivate(const rclcpp_lifecycle::State& previous_state) {
  if (ControllerBase::on_deactivate(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }
  // Stop the model thread
  modelThreadRunning_ = false;
  modelCv_.notify_one(); // Wake up the thread if waiting
  if (modelThread_.joinable()) {
    modelThread_.join();
  }

  // Reset action availability
  {
    std::lock_guard<std::mutex> lock(actionMutex_);
    latestAction_.setZero();
    actionAvailable_ = false;
  }

  // Close the JSON array and file
  if (logObs_ == true) {
    std::lock_guard<std::mutex> lock(obsFileMutex_);
    obsJsonFile_ << std::endl << "  ]" << std::endl << "}" << std::endl;
    obsJsonFile_.close();
  }

  // Release the buffer
  observationBuffer_.clear();
  return controller_interface::CallbackReturn::SUCCESS;
}

Observations LeggedTextCommandController::getObservations() {
  const auto leggedModel = leggedModel_->getLeggedModel();
  const auto q = leggedModel->getGeneralizedPosition();
  const auto v = leggedModel->getGeneralizedVelocity();

  const quaternion_t quat(q.segment<4>(3));
  const matrix_t inverseRot = quat.toRotationMatrix().transpose();
  const vector3_t baseLinVel = inverseRot * v.segment<3>(0);
  const auto& angVelArray = imu_->get_angular_velocity();
  const vector3_t baseAngVel(angVelArray[0], angVelArray[1], angVelArray[2]);
  const vector3_t gravityVector(0, 0, -1);
  const vector3_t projectedGravity(inverseRot * gravityVector);

  const vector_t jointPositions = q.tail(lastActions_.size());
  const vector_t jointVelocities = v.tail(lastActions_.size());

  vector_t jointPositionsInPolicy(jointNameInPolicy_.size());
  vector_t jointVelocitiesInPolicy(jointNameInPolicy_.size());
  for (size_t i = 0; i < jointNameInPolicy_.size(); ++i) {
    const size_t hardwareIndex = jointIndexMap_[jointNameInPolicy_[i]];
    jointPositionsInPolicy[i] = jointPositions[hardwareIndex] - defaultPosition_[hardwareIndex];
    jointVelocitiesInPolicy[i] = jointVelocities[hardwareIndex];
  }

  vector_t observations(observationSize_);
  vector_t currentObsWithHistory(obsWithHistorySize_);
  size_t index = 0;
  for (const auto & name : obsWithHistoryNames_) {
    if (name == "base_lin_vel") {
      currentObsWithHistory.segment<3>(index) = baseLinVel;
      index += 3;
    } else if (name == "base_ang_vel") {
      currentObsWithHistory.segment<3>(index) = baseAngVel;
      index += 3;
    } else if (name == "projected_gravity") {
      currentObsWithHistory.segment<3>(index) = projectedGravity;
      index += 3;
    } else if (name == "joint_positions") {
      // currentObsWithHistory.segment(index, jointPositionsInPolicy.size()) = jointPositionsInPolicy;
      currentObsWithHistory.segment(index, jointPositionsInPolicy.size()) = remapJointOrder(jointPositionsInPolicy);
      index += jointPositionsInPolicy.size();
    } else if (name == "joint_velocities") {
      currentObsWithHistory.segment(index, jointVelocitiesInPolicy.size()) = remapJointOrder(jointVelocitiesInPolicy);
      index += jointVelocitiesInPolicy.size();
    } else if (name == "last_action") {
      currentObsWithHistory.segment(index, lastActions_.size()) = lastActions_;
      index += lastActions_.size();
    }
  }
  observations.segment(0, obsWithHistorySize_ * obsHistoryLength_) = updateObsBuffer(currentObsWithHistory);
  index = obsWithHistorySize_ * obsHistoryLength_;
  for (const auto & name : obsWithoutHistoryNames_) {
    if (name == "text_command") {
      observations.segment(index, 512) = command_;
      index += 512;
    }
  }

  Observations obs_struct;
  obs_struct.observations = observations;
  obs_struct.currentProprioception = currentObsWithHistory;

  return obs_struct;
}

vector_t LeggedTextCommandController::updateObsBuffer(const vector_t& currentObsWithHistory){
  // Initialize the buffer if it's empty by filling it with the first observation
  if (observationBuffer_.empty()) {
    for (size_t i = 0; i < obsBufferLength_; ++i) {
      observationBuffer_.emplace_back(currentObsWithHistory);
    }
  } else {
    // Update the buffer by adding the new observation to the front
    observationBuffer_.emplace_front(currentObsWithHistory);

    // Ensure the buffer size does not exceed obsBufferLength_
    if (observationBuffer_.size() > obsBufferLength_) {
      observationBuffer_.pop_back();  // Remove the oldest observation
    }
  }

  // Sample the observation history at fixed intervals
  // Calculate the sampling step to evenly distribute samples over the buffer
  size_t step = 1;
  if (obsHistoryLength_ > 1) {
    step = (obsBufferLength_ - 1) / (obsHistoryLength_ - 1);
    if (step == 0) step = 1;  // Prevent division by zero
  }

  // Ensure that OBS_BUFFER_LENGTH is sufficient
  size_t max_required = step * (obsHistoryLength_ - 1);
  if (max_required >= observationBuffer_.size()) {
    RCLCPP_WARN(get_node()->get_logger(),
                "OBS_BUFFER_LENGTH (%zu) is not sufficient for OBS_HISTORY_LENGTH (%zu) with step (%zu). Adjusting step.",
                obsBufferLength_, obsHistoryLength_, step);
    step = 1;
  }

  // Initialize the history vector
  vector_t obsHistory = vector_t::Zero(obsHistoryLength_ * obsWithHistorySize_);

  // Sample observations from the buffer
  for (size_t i = 0; i < obsHistoryLength_; ++i) {
    // Calculate the index for sampling
    size_t idx = i * step;
    if (idx >= observationBuffer_.size()) {
      idx = observationBuffer_.size() - 1;  // Clamp to the oldest observation
    }

    // Copy the sampled observation into the history vector
    obsHistory.segment(i * obsWithHistorySize_, obsWithHistorySize_) = observationBuffer_[idx];
  }

  return obsHistory;
}

vector_t LeggedTextCommandController::playModel(const Observations& obsStructure) const {
  auto observations = obsStructure.observations;
  auto currentProprioception = obsStructure.currentProprioception;
  // clang-format on
  std::vector<tensor_element_t> observationTensor;
  for (const double i : currentProprioception) {
    observationTensor.push_back(static_cast<tensor_element_t>(i));
  }
  for (const double i : observations) {
    observationTensor.push_back(static_cast<tensor_element_t>(i));
  }
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, observationTensor.data(), observationTensor.size(),
                                                                   inputShapes_[0].data(), inputShapes_[0].size()));
  // run inference
  const Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues = sessionPtr_->Run(runOptions, inputNames_.data(), inputValues.data(), 1, outputNames_.data(), 1);

  vector_t actions(lastActions_.size());
  for (Eigen::Index i = 0; i < actions.size(); ++i) {
    actions[i] = outputValues[0].At<tensor_element_t>({0, static_cast<long int>(i)});
  }
  return actions;
}

vector_t LeggedTextCommandController::remapJointOrder(const vector_t & raw_vector) const
  {
    if (raw_vector.size() != static_cast<long>(jointNameInPolicy_.size())) {
      throw std::runtime_error("Vector size does not match jointNameInPolicy_ size.");
    }

    vector_t obs(obsIndexMap_.size());

    for (size_t i = 0; i < obsIndexMap_.size(); ++i) {
      obs(static_cast<long>(i)) = raw_vector(static_cast<long>(obsIndexMap_[i]));
    }

    return obs;
  }

}  // namespace legged

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(legged::LeggedTextCommandController, controller_interface::ControllerInterface)
