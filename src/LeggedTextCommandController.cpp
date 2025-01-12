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
  command_ = commandList_.front();

  if (firstUpdate_ || (time - lastPlayTime_).seconds() >= 1. / policyFrequency_) {
    auto obs_struct = getObservations();
    lastActions_ = playModel(obs_struct);

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
      for (const double i : obs_struct.currentProprioception) {
        msg.data.push_back(i);
      }
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
  std::ifstream commandFile(get_node()->get_parameter("command.command_file_path").as_string());
  if (!commandFile.is_open()) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to open command file: %s", get_node()->get_parameter("command.command_file_path").as_string().c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  json dataSet;
  commandFile >> dataSet;
  auto task = get_node()->get_parameter("command.task").as_string_array();
  auto task_duration = get_node()->get_parameter("command.task_duration").as_double_array();
  if (!dataSet.contains(task[0])) {
    RCLCPP_ERROR(get_node()->get_logger(), "Task '%s' not found in command file.", task[0].c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  for (size_t i = 0; i < task.size(); ++i) {
    vector_t command_embedding = dataSet[task[i]]["embedding"][0];
    RCLCPP_INFO(get_node()->get_logger(), "Command: %s", dataSet[task[i]]["caption"].get<std::string>().c_str());
    RCLCPP_INFO(get_node()->get_logger(), "Duration: %f", task_duration[i]);
    commandList_.push_back(command_embedding);
    commandDurationList_.push_back(task_duration[i]);
  }

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
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn LeggedTextCommandController::on_deactivate(const rclcpp_lifecycle::State& previous_state) {
  if (ControllerBase::on_deactivate(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
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
