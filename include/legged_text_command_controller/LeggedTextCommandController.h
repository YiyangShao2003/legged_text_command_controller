#pragma once

#include "legged_rl_controllers/OnnxController.h"

namespace legged {
class LeggedTextCommandController : public OnnxController {
 protected:
  vector_t playModel(const vector_t& observations) const override;
};

}  // namespace legged
