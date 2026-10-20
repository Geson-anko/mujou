# PC-ALM

::: mujou.pc
    options:
      members: [Input, Layer, Output, TimeConstants, carry_spec, Carried, Sample, Activity, Prediction, mse, cross_entropy]
      # The nodes would repeat every member of Node.
      inherited_members: false

::: mujou.pc.functional
    options:
      members: [vjp, loss_vjp, relax_, learn_]
