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
#include <queue>
#include "tensorflow/core/grappler/optimizers/multi_stream_optimizer.h"

#include "tensorflow/core/common_runtime/scoped_allocator.h"
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {
namespace grappler {

MultiStreamOptimizer::MultiStreamOptimizer(
    const MultiStreamOptions& opt) : opt_(opt){
}

Status MultiStreamOptimizer::MarkEmbeddingGraphNodes(
    const std::vector<NodeDef*> start_nodes,
    std::unordered_map<std::string, std::vector<NodeDef*>> output_edges,
    GraphDef* optimized_graph) {
  int curr_stream_idx = 0;
  for (auto n : start_nodes) {
    tensorflow::AttrValue stream_idx_attr;
    stream_idx_attr.set_i(curr_stream_idx);
    (*n->mutable_attr())["stream_idx"] = stream_idx_attr;

    std::queue<NodeDef*> q;
    q.push(n);
    while (q.empty()) {
      NodeDef* curr_node = q.front();
      q.pop();
      auto attr = n->attr().at("stream_idx");
      // check all output edges
      for (auto out_node : output_edges[curr_node->name()]) {
        if (out_node->name().find("/embedding_lookup_sparse") !=
            std::string::npos) {
          (*out_node->mutable_attr())["stream_idx"] = attr;
          q.push(out_node);
        }
      }
    }
    ++curr_stream_idx;
    curr_stream_idx %= opt_.multi_stream_num();
  }

  return Status::OK();
}

Status MultiStreamOptimizer::SplitEmbeddingGraph(
    const GrapplerItem& item, GraphDef* optimized_graph) {
  *optimized_graph = item.graph;
  // Find embedding graph start node(variable node)
  std::vector<NodeDef*> start_nodes;
  std::unordered_map<std::string, NodeDef*> name_to_node;
  std::unordered_map<std::string, std::vector<NodeDef*>> output_edges;
  for (const NodeDef& node : optimized_graph->node()) {
    name_to_node[node.name()] = const_cast<NodeDef*>(&node);
    if (node.name().find("/embedding_lookup_sparse") != std::string::npos &&
        IsVariable(node)) {
      start_nodes.emplace_back(const_cast<NodeDef*>(&node));
      for (auto name : node.input()) {
        if (name_to_node.find(name) == name_to_node.end()) {
          LOG(FATAL) << "Can't found input node " << name
                     << ", current node is " << node.DebugString();
        }
        output_edges[name].push_back(const_cast<NodeDef*>(&node));
      }
    }
  }

  return MarkEmbeddingGraphNodes(start_nodes, output_edges, optimized_graph);
}

Status MultiStreamOptimizer::Optimize(
    Cluster* cluster, const GrapplerItem& item,
    GraphDef* optimized_graph) {
  if (opt_.partition_policy() ==
      MultiStreamPartitionPolicy::NO_PARTITION) {
    // nothing
  } else if (opt_.partition_policy() ==
      MultiStreamPartitionPolicy::USER_DEFINED_PARTITION) {
    // TODO
  } else if (opt_.partition_policy() ==
      MultiStreamPartitionPolicy::EMBEDDING_GRAPH_PARTITION) {
    // TODO
    return SplitEmbeddingGraph(item, optimized_graph);
  } else if (opt_.partition_policy() ==
             MultiStreamPartitionPolicy::FULL_GRAPH_PARTITION) {
    // TODO
  }

  return Status::OK();
}

}  // namespace grappler
}  // namespace tensorflow
