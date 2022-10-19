/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <unordered_map>
#include "tensorflow/core/graph/stream_subgraph.h"

namespace tensorflow {
namespace stream_subgraph {

namespace {

void GetColocationConstraints(const Node* node,
                              std::vector<string>* constraints) {
  TF_DCHECK_OK(GetNodeAttr(node->attrs(),
      kColocationAttrName, constraints));
}

// I1-I13, C1-C26
// dnn/input_from_feature_columns/input_layer/I11/truediv
// dnn/input_from_feature_columns/input_layer/I
// dnn/input_from_feature_columns/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse
// input_from_feature_columns/input_layer/C

static std::unordered_map<std::string, int> name2id = {
  {"input_from_feature_columns/input_layer/I1/" , 0},
  {"input_from_feature_columns/input_layer/I2/" , 1},
  {"input_from_feature_columns/input_layer/I3/" , 2},
  {"input_from_feature_columns/input_layer/I4/" , 3},
  {"input_from_feature_columns/input_layer/I5/" , 4}, 
  {"input_from_feature_columns/input_layer/I6/" , 5},
  {"input_from_feature_columns/input_layer/I7/" , 6},
  {"input_from_feature_columns/input_layer/I8/" , 7},
  {"input_from_feature_columns/input_layer/I9/" , 8},
  {"input_from_feature_columns/input_layer/I10/" , 9}, 
  {"input_from_feature_columns/input_layer/I11/" , 10},
  {"input_from_feature_columns/input_layer/I12/" , 11},
  {"input_from_feature_columns/input_layer/I13/" , 12},
  {"input_from_feature_columns/input_layer/C1_embedding" , 13},
  {"input_from_feature_columns/input_layer/C2_embedding" , 14}, 
  {"input_from_feature_columns/input_layer/C3_embedding" , 15},
  {"input_from_feature_columns/input_layer/C4_embedding" , 16},
  {"input_from_feature_columns/input_layer/C5_embedding" , 17},
  {"input_from_feature_columns/input_layer/C6_embedding" , 18},
  {"input_from_feature_columns/input_layer/C7_embedding" , 19}, 
  {"input_from_feature_columns/input_layer/C8_embedding" , 20},
  {"input_from_feature_columns/input_layer/C9_embedding" , 21},
  {"input_from_feature_columns/input_layer/C10_embedding" , 22},
  {"input_from_feature_columns/input_layer/C11_embedding" , 23},
  {"input_from_feature_columns/input_layer/C12_embedding" , 24}, 
  {"input_from_feature_columns/input_layer/C13_embedding" , 25},
  {"input_from_feature_columns/input_layer/C14_embedding" , 26},
  {"input_from_feature_columns/input_layer/C15_embedding" , 27},
  {"input_from_feature_columns/input_layer/C16_embedding" , 28},
  {"input_from_feature_columns/input_layer/C17_embedding" , 29}, 
  {"input_from_feature_columns/input_layer/C18_embedding" , 30},
  {"input_from_feature_columns/input_layer/C19_embedding" , 31},
  {"input_from_feature_columns/input_layer/C20_embedding" , 32},
  {"input_from_feature_columns/input_layer/C21_embedding" , 33},
  {"input_from_feature_columns/input_layer/C22_embedding" , 34}, 
  {"input_from_feature_columns/input_layer/C23_embedding" , 35},
  {"input_from_feature_columns/input_layer/C24_embedding" , 36},
  {"input_from_feature_columns/input_layer/C25_embedding" , 37},
  {"input_from_feature_columns/input_layer/C26_embedding" , 38}
};

} // namesapce

// TODO: Fix hack function -----> For testing
//
void MarkStreamSubGraph(Graph* g, const int num_streams) {
  bool train_graph = false;

  for (Node* n : g->nodes()) {
    if (n->type_string() == "ApplyAdamAsync") {
      train_graph = true;
      break;
    }
  }
  if (!train_graph) return;
/*
  for (Node* n : g->nodes()) {
    if (n->type_string() == "IsVariableInitialized" &&
        n->name() != "global_step/IsVariableInitialized") {
        return;
    }
  }*/

  std::unordered_map<std::string, Node*> name_to_node;

  // User marked subgraph
  for (Node* n : g->nodes()) {
	LOG(INFO) << n->name();
    name_to_node[n->name()] = n;
    if (n->assigned_device_name().find("device:GPU:") == std::string::npos) {
      continue;
    }

//======================================================================
    int expert_id = -1;
    for (auto xx : name2id) {
      if (n->name().find(xx.first) != std::string::npos) {
        expert_id = xx.second;
        expert_id %= num_streams;
        break;
      }
    }
    if (expert_id > -1) {
      LOG(INFO) << "=====================> stream_id: " << expert_id << ", node: " << n->name();
      std::string required_device = 
          "/job:localhost/replica:0/task:0/device:GPU:" + std::to_string(expert_id);
      if (n->assigned_device_name() != required_device) { 
        n->set_assigned_device_name(required_device);
      }
    }

		/*
    int expert_id = 0;
    const std::string prefix("my_scope_");
    //const std::string prefix("MMOE/expert_");
    auto idx = n->name().find(prefix);
    if (idx != std::string::npos) {
      idx += prefix.length();
      while (idx < n->name().length() && n->name()[idx] != '/') {
        expert_id *= 10;
        expert_id += (int)(n->name()[idx++]-'0');
      }

      int stream_id = expert_id % num_streams; // stream_id is needed start from 0


LOG(INFO) << "=====================> stream_id: " << stream_id << ", node: " << n->name();
      std::string required_device =
          "/job:localhost/replica:0/task:0/device:GPU:" + std::to_string(stream_id);
      if (n->assigned_device_name() != required_device) {
        n->set_assigned_device_name(required_device);
      }
    }*/
  }

  // Colocate nodes
  for (Node* n : g->nodes()) {
    std::vector<string> constraints;
    GetColocationConstraints(n, &constraints);
    if (constraints.size() > 0) {
      // constraint like "loc:@report_uninitialized_variables_1/IsVariableInitialized_294",
      // skip 'loc:@'
      Node* colocate_node = name_to_node[constraints[0].substr(5)];
      if (!colocate_node) {
        LOG(FATAL) << "Colocate node not existed, " << constraints[0].substr(5)
                   << ", current noe: " << n->DebugString();
      }
      if (n->assigned_device_name() !=
          colocate_node->assigned_device_name()) {
        n->set_assigned_device_name(colocate_node->assigned_device_name());
      }
    }
  }

	// Copy constant op
	for (Node* n : g->nodes()) {
		std::vector<const Edge*> input_edges;
		for (auto e : n->in_edges()) {
			input_edges.push_back(e);
		}
		std::string prefix("/job:localhost/replica:0/task:0/device:GPU:");
		for (auto e : input_edges) {
			Node* input = e->src();
			std::string in_name = input->name();
			if (input->op_def().name() == "Const" && 
					input->assigned_device_name() != n->assigned_device_name()) {
				// get a new name
				if (n->assigned_device_name().find(prefix) == std::string::npos) continue;
      	auto idx = prefix.length();
				int device_id = 0;
				auto& d_name = n->assigned_device_name();
      	while (idx < d_name.length() && d_name[idx] != '/') {
        	device_id *= 10;
        	device_id += (int)(d_name[idx++]-'0');
				}
				std::string copy_name(in_name + "/" +  std::to_string(device_id));

				// check if it's already copied
				if (name_to_node.find(copy_name) != name_to_node.end()) {
					g->AddEdge(name_to_node[copy_name], 0, n, e->dst_input());
				} else {
					// create a new Node
					Node* copied = g->CopyNode(input);
					copied->set_name(copy_name); 
					copied->set_assigned_device_name(n->assigned_device_name());
					g->AddEdge(copied, 0, n, e->dst_input());
					name_to_node[copy_name] = copied;
				}
				g->RemoveEdge(e);
			}
		}	
	}

  // General algorithm
}

void MarkStreamSubGraph2(Graph* g, const int num_streams) {
  bool train_graph = false;

  // judge trained graph
  for (Node* n : g->nodes()) {
    // TODO: FIXME
    if (n->type_string() == "ApplyAdamAsync") {
      train_graph = true;
      break;
    }
  }
  if (!train_graph) return;

  std::unordered_map<std::string, Node*> name_to_node;
  // User marked subgraph
  for (Node* n : g->nodes()) {
    if (n->assigned_device_name().find("device:GPU:") == std::string::npos ||
        n->def().attr().find("stream_idx") == n->def().attr().end()) {
      continue;
    }

    int stream_idx = n->def().attr().at("stream_idx").i();
    std::string required_device =
        "/job:localhost/replica:0/task:0/device:GPU:" + std::to_string(stream_idx);
    if (n->assigned_device_name() != required_device) {
      n->set_assigned_device_name(required_device);
    }
  }

  // Colocate nodes
  for (Node* n : g->nodes()) {
    std::vector<string> constraints;
    GetColocationConstraints(n, &constraints);
    if (constraints.size() > 0) {
      // constraint like "loc:@report_uninitialized_variables_1/IsVariableInitialized_294",
      // skip 'loc:@'
      Node* colocate_node = name_to_node[constraints[0].substr(5)];
      if (!colocate_node) {
        LOG(FATAL) << "Colocate node not existed, " << constraints[0].substr(5)
                   << ", current noe: " << n->DebugString();
      }
      if (n->assigned_device_name() !=
          colocate_node->assigned_device_name()) {
        n->set_assigned_device_name(colocate_node->assigned_device_name());
      }
    }
  }

	// Copy constant op
	for (Node* n : g->nodes()) {
		std::vector<const Edge*> input_edges;
		for (auto e : n->in_edges()) {
			input_edges.push_back(e);
		}
		std::string prefix("/job:localhost/replica:0/task:0/device:GPU:");
		for (auto e : input_edges) {
			Node* input = e->src();
			std::string in_name = input->name();
			if (input->op_def().name() == "Const" && 
					input->assigned_device_name() != n->assigned_device_name()) {
				// get a new name
				if (n->assigned_device_name().find(prefix) == std::string::npos) continue;
      	auto idx = prefix.length();
				int device_id = 0;
				auto& d_name = n->assigned_device_name();
      	while (idx < d_name.length() && d_name[idx] != '/') {
        	device_id *= 10;
        	device_id += (int)(d_name[idx++]-'0');
				}
				std::string copy_name(in_name + "/" +  std::to_string(device_id));

				// check if it's already copied
				if (name_to_node.find(copy_name) != name_to_node.end()) {
					g->AddEdge(name_to_node[copy_name], 0, n, e->dst_input());
				} else {
					// create a new Node
					Node* copied = g->CopyNode(input);
					copied->set_name(copy_name); 
					copied->set_assigned_device_name(n->assigned_device_name());
					g->AddEdge(copied, 0, n, e->dst_input());
					name_to_node[copy_name] = copied;
				}
				g->RemoveEdge(e);
			}
		}	
	}
  // TODO
}

}  // namespace stream_subgraph
}  // namespace tensorflow
