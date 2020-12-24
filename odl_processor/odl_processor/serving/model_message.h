#ifndef TENSORFLOW_MODEL_MESSAGE_H
#define TENSORFLOW_MODEL_MESSAGE_H

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
class Tensor;
class TensorInfo;

namespace processor {
struct Request {
  std::vector<std::pair<std::string, Tensor>> inputs;
  std::vector<std::string> output_tensor_names;
};

struct Response {
  std::vector<Tensor> outputs;
};

struct Call {
  Request request;
  Response response;
};

struct BatchCall {
  std::vector<Request> request;
  std::vector<Response> response;

  Request batched_request;
  Response batched_response;
  int call_num; 

  Status BatchRequest();
  Status SplitResponse();
};

} // processor
} // tensorflow

#endif // TENSORFLOW_MODEL_MESSAGE_H

