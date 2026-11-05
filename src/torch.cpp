#include "mim/plug/torch/torch.h"

#include <mim/plugin.h>

#include "mim/plug/torch/phase/lower.h"

using namespace mim;

namespace mim::plug::torch {

void reg_phases(Flags2Phases& phases) {
    Phase::hook<decompose_torch, phase::Decompose>(phases);
    Phase::hook<lower_torch,     phase::Lower    >(phases);
}

} // namespace mim::plug::torch

MIM_PLUGIN_ENTRY(torch) {
    plugin.register_normalizers = mim::plug::torch::register_normalizers;
    plugin.register_phases      = mim::plug::torch::reg_phases;
}
