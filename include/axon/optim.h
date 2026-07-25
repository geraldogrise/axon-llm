#pragma once

#include <cstdint>
#include <vector>

#include "axon/tensor.h"

namespace axon::optim {

// Base of an optimizer: holds the parameters and knows how to update them from
// the gradients computed by autograd.
class Optimizer {
  public:
    explicit Optimizer(std::vector<Tensor> params) : params_(std::move(params)) {}
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    void zero_grad();

    // Learning rate (for LR schedulers).
    virtual void set_lr(float lr) = 0;
    [[nodiscard]] virtual float lr() const = 0;

    // Optimizer's internal state (to save/resume training). Each element is a
    // Tensor; the layout is defined by each optimizer.
    [[nodiscard]] virtual std::vector<Tensor> state_dict() const { return {}; }
    virtual void load_state_dict(const std::vector<Tensor>&) {}

  protected:
    std::vector<Tensor> params_;
};

// Stochastic gradient descent (with optional momentum and weight decay).
class SGD : public Optimizer {
  public:
    SGD(std::vector<Tensor> params, float lr, float momentum = 0.0f, float weight_decay = 0.0f);
    void step() override;
    void set_lr(float lr) override { lr_ = lr; }
    [[nodiscard]] float lr() const override { return lr_; }

  private:
    float lr_;
    float momentum_;
    float weight_decay_;
    std::vector<std::vector<float>> velocity_;  // state per parameter
};

// Adam / AdamW: 1st- and 2nd-order moving averages of the gradient.
// weight_decay > 0 applies decoupled decay (AdamW style).
class Adam : public Optimizer {
  public:
    Adam(std::vector<Tensor> params, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
         float eps = 1e-8f, float weight_decay = 0.0f);
    void step() override;
    void set_lr(float lr) override { lr_ = lr; }
    [[nodiscard]] float lr() const override { return lr_; }
    // Layout: [t, m0, v0, m1, v1, ...] (t como Tensor {1}).
    [[nodiscard]] std::vector<Tensor> state_dict() const override;
    void load_state_dict(const std::vector<Tensor>& state) override;

  private:
    float lr_, beta1_, beta2_, eps_, weight_decay_;
    std::int64_t t_ = 0;
    std::vector<std::vector<float>> m_;  // 1st moment
    std::vector<std::vector<float>> v_;  // 2nd moment
};

// RMSProp: divides the step by a moving average of the squared gradient.
class RMSProp : public Optimizer {
  public:
    RMSProp(std::vector<Tensor> params, float lr = 1e-3f, float alpha = 0.99f,
            float eps = 1e-8f, float weight_decay = 0.0f);
    void step() override;
    void set_lr(float lr) override { lr_ = lr; }
    [[nodiscard]] float lr() const override { return lr_; }

  private:
    float lr_, alpha_, eps_, weight_decay_;
    std::vector<std::vector<float>> cache_;  // moving average of grad^2
};

}  // namespace axon::optim
