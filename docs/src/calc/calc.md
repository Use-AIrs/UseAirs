# [Calculator](https://github.com/Use-AIrs/Use-Ai.rs/tree/main/crates/core/lib-calculator)

In `lib-calculator` the most crucial parts are the module `model` and the `operator`.
In `model` we can easily build models within the `operation` trait which are supported by the `opertator` module and trait.
So, the `operator` provides functions through the `PipelineExec` and `PipelinePush` traits.
The `operator` will be executed on the GPUs Kernel.
Every `model` needs to implement the `Operation` trait 