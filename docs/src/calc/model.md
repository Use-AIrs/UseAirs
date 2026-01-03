# Model

When you want to integrate a new `model` you can do that by doing these three things:

1. Build a struct which includes tuples of `MetaData` and `Handle`. This structure is our abstract representation of the shared memory on the GPU. The macro ` #[operator]` above your struct will allocate a `Tensor` on the GPU and leave you a `TensorHandelRef` to work with in the `operation`