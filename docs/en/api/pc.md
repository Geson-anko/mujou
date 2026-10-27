# PC-ALM

::: mujou.pc
    options:
      members: [Chain, Shown, show, Input, Layer, Output, TimeConstants]
      # The nodes would repeat every member of Node.
      inherited_members: false

::: mujou.pc.optim
    options:
      members: [Optimizer, BoundOptimizer, SGD, AdamW]

::: mujou.pc.relax
    options:
      members: [Relaxer, BoundRelaxer, Euler, HeavyBall]

::: mujou.pc.losses
    options:
      members: [MSELoss, CrossEntropyLoss]

::: mujou.pc.messages
    options:
      members: [Sample, Activity, Prediction, activity_spec]

::: mujou.pc.clock
    options:
      members: [LearnClock, Tick]

::: mujou.pc.functional
    options:
      members: [vjp, loss_vjp, relax_, mse, cross_entropy]
