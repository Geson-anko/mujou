# PC-ALM

::: mujou.pc
    options:
      members: [Chain, show, Input, Layer, Output, TimeConstants, LearnClock, Tick, Sample, Activity, Prediction, activity_spec, MSELoss, CrossEntropyLoss]
      # The nodes would repeat every member of Node.
      inherited_members: false

::: mujou.pc.functional
    options:
      members: [vjp, loss_vjp, relax_, learn_, mse, cross_entropy]
