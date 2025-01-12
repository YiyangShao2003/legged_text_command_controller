#include <memory>

#include "legged_text_command_controller/LeggedTextCommandController.h"

namespace legged {
vector_t LeggedTextCommandController::playModel(const vector_t& observations) const {
  std::cerr << "LeggedTextCommandController::playModel" << std::endl;
  return OnnxController::playModel(observations);
}
}  // namespace legged

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(legged::LeggedTextCommandController, controller_interface::ControllerInterface)
