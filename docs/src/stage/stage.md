## [Stage](https://github.com/Use-AIrs/Use-Ai.rs/tree/main/crates/core/lib-stage)
#### Current
Here is where the construction site begins. Currently, we only have a CSV pipeline to show what the plan is with `lib-stager`.
This pipeline shows how we want to handle data on a more abstract layer. For now, we just input some csv file and handle
it from there with different operations on columns or the hole data. This is chosen because we want to reach a first complete
pass through *Use-AI.rs* using a simple XGBoost algorithm for now. Till we expanded that simple XGBoost it's called GBDT here.

### Future
For DQNs and Q-learning we need to expand `lib-stage`. Goal is not to only use files instead this will be used to Configure
a Data Stream which first, will enable the possibility to get Input from program external drivers and also, will add the
possibility for communication between models. But to be able to implement such features we first need to focus on the
`lib-calculator`.
The functions as they are, will be renamed in near Future.