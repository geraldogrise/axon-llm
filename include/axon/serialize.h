#pragma once

#include <string>
#include <vector>

#include "axon/tensor.h"

namespace axon {

// Saves a list of tensors (typically model.parameters()) to a simple binary
// file. Format: magic "AXON" + version + n + [ndim, dims..., floats].
void save(const std::vector<Tensor>& tensors, const std::string& path);

// Loads the file's data into the existing tensors (copies in-place),
// preserving the identity of the parameters (the optimizer stays valid).
// Throws if the count or the shapes do not match.
void load(std::vector<Tensor>& tensors, const std::string& path);

}  // namespace axon
