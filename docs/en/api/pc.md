# PC-ALM

::: mujou.pc
    options:
      members: [Chain, Shown, show, Input, Layer, Output, TimeConstants, Optimizer, BoundOptimizer, SGD, AdamW, LearnClock, Tick, Sample, Activity, Prediction, activity_spec, MSELoss, CrossEntropyLoss]
      # The nodes would repeat every member of Node.
      inherited_members: false

::: mujou.pc.functional
    options:
      members: [vjp, loss_vjp, relax_, mse, cross_entropy]
