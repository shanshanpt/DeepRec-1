/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_HASHMAP_H_
#define TENSORFLOW_CORE_FRAMEWORK_HASHMAP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/typed_allocator.h"
#include "tensorflow/core/framework/kv_interface.h"

namespace tensorflow {

struct RestoreBuffer {
  char* key_buffer;
  char* value_buffer;
  char* version_buffer;

  ~RestoreBuffer() {
    delete key_buffer;
    delete value_buffer;
    delete version_buffer;
  }
};


struct EmbeddingConfig {
  int64 emb_index;
  int64 primary_emb_index;
  int64 block_num;
  int64 slot_num;
  std::string name;
  int64 steps_to_live;

  EmbeddingConfig(int64 emb_index = 0, int64 primary_emb_index = 0,
                 int64 block_num = 1, int slot_num = 1,
                 const std::string& name = "", int64 steps_to_live = 0):
                  emb_index(emb_index),
                  primary_emb_index(primary_emb_index),
                  block_num(block_num), slot_num(slot_num) , name(name),
                  steps_to_live(steps_to_live){}

  bool is_primary() const {
    return emb_index == primary_emb_index;
  }

  int64 total_num() {
    return block_num * (slot_num + 1);
  }

  std::string DebugString() const {
     return strings::StrCat("opname: ", name, " emb_index: ", emb_index,
                         " primary_emb_index: ", primary_emb_index, " block_num: ", block_num,
                         " slot_num: ", slot_num);

  }
};

template <class K, class V>
class HashMap {
 public:
  HashMap(KVInterface<K, V>* kv, Allocator* alloc = cpu_allocator(), bool use_db = false,
             EmbeddingConfig emb_cfg=EmbeddingConfig()):
      kv_(kv),
      default_value_(NULL),
      value_len_(0),
      value_and_version_len_(0),
      alloc_(alloc),
      use_db_(use_db),
      emb_config_(emb_cfg) {}

  Status Init(const Tensor& default_tensor) {
    if (default_tensor.dims() != 1) {
      return errors::InvalidArgument("EV's default_tensor shape must be 1-D");
    } else if (DataTypeToEnum<V>::v() != default_tensor.dtype()) {
       return errors::InvalidArgument("EV's default_tensor DTYPE must be same as HashMap Value Type");
    } else if (kv_ == nullptr) {
       return errors::InvalidArgument("Invalid ht_type to construct HashMap");
    } else {
      if (use_db_) {
        std::string db_name = "./dbhashmap_";
        db_name.append(std::to_string(Env::Default()->NowMicros()));
        db_kv_ = new LevelDBHashMap<K, V>(db_name);
      }
      value_len_ = default_tensor.NumElements();
      default_value_ = TypedAllocator::Allocate<V>(alloc_, value_len_, AllocationAttributes());
      auto default_tensor_flat = default_tensor.flat<V>();
      memcpy(default_value_, &default_tensor_flat(0), default_tensor.TotalBytes());
      value_and_version_len_ = value_len_ + (sizeof(int64) + sizeof(V) - 1) / sizeof(V);
      return Status::OK();
    }
  }

  ~HashMap() {
    if (emb_config_.is_primary()) {
      kv_->Destroy(value_len_, value_and_version_len_);
      delete kv_;
    }
    if (use_db_) {
      delete db_kv_;
    }
    TypedAllocator::Deallocate(alloc_, default_value_, value_len_);
  }

  // for embedding 3.0
   KVInterface<K, V>* kv() { return kv_; }

   V* LookupOrCreateV3(ValuePtr<V>* valptr, const EmbeddingConfig& embcfg, const V* default_v, int64 update_version = -1) {
    bool allocate_version = false;
    if (embcfg.is_primary() && embcfg.steps_to_live != 0) {
      allocate_version = true;
    }
    V* val = valptr->GetOrAllocate(alloc_, value_len_, default_v, embcfg.emb_index, allocate_version);
    if (allocate_version && update_version != -1) {
      int64* version = reinterpret_cast<int64*>(val + value_len_);
      *version = update_version;
    }
    return val;
  }
  V* LookupOrCreateV3(K key, int64 update_version = -1) {
    ValuePtr<V>* valptr = kv_->Lookup(key, emb_config_.total_num());
    return LookupOrCreateV3(valptr,  emb_config_, default_value_, update_version);
  }
  ValuePtr<V>* LookupValuePtr(K key) {
    return kv_->Lookup(key, emb_config_.total_num());
  }
  typename TTypes<V>::Flat flat_emb(ValuePtr<V>* valptr, int64 update_version = -1) {
    V* val = LookupOrCreateV3(valptr, emb_config_, default_value_, update_version);
    Eigen::array<Eigen::DenseIndex, 1> dims({value_len_});
    return typename TTypes<V>::Flat(val, dims);
  }
  typename TTypes<V>::Flat flat_emb(ValuePtr<V>* valptr, const EmbeddingConfig embcfg, int64 update_version = -1) {
    V* val = LookupOrCreateV3(valptr, embcfg, default_value_, update_version);
    Eigen::array<Eigen::DenseIndex, 1> dims({value_len_});
    return typename TTypes<V>::Flat(val, dims);
  }

  typename TTypes<V>::Flat flatV3ForTest(K key, int64 update_version = -1) {
    const V* default_value_ptr =  default_value_;
    ValuePtr<V>* valptr = kv_->Lookup(key, emb_config_.total_num());
    LookupOrCreateV3(valptr, emb_config_, default_value_ptr, update_version);
    return flat_emb(valptr, update_version);
  }
  void LookupOrCreateHybridV3(K key,  V* val, V* default_v)  {
    if (use_db_) {
      std::string db_value;
      Status st = db_kv_->Lookup(key, &db_value);
      if (!st.ok()) {
        memcpy(val, default_value_, sizeof(V) * value_len_);
        //alloc_->Deallocate(new_val, value_and_version_len_);
      } else {
        // found in disk
        memcpy(val, db_value.data(), sizeof(V) * value_len_);
      }
    } else {
      const V* default_value_ptr = (default_v == NULL) ? default_value_ : default_v;
      ValuePtr<V>* valptr = kv_->Lookup(key, emb_config_.total_num());
      V* mem_val = LookupOrCreateV3(valptr, emb_config_, default_value_ptr);
      memcpy(val, mem_val, sizeof(V) * value_len_);
    }
  }
  int64 GetSnapshot(std::vector<K>* key_list, std::vector<V* >* value_list, std::vector<int64>* version_list) {
    return kv_->GetSnapshot(key_list, value_list, version_list,
      emb_config_.emb_index, emb_config_.primary_emb_index, value_len_, emb_config_.steps_to_live);
  }

  Status ImportV3(RestoreBuffer& restore_buff, int64 key_num,
                int bucket_num,
                int64 partition_id,
                int64 partition_num) {
    K* key_buff = (K*)restore_buff.key_buffer;
    V* value_buff = (V*)restore_buff.value_buffer;
    int64* version_buff = (int64*)restore_buff.version_buffer;
    for (auto i = 0; i < key_num; ++i) {
      // this can describe by graph(Mod + DynamicPartition), but memory waste and slow
      if (*(key_buff + i) % bucket_num % partition_num != partition_id) {
        LOG(INFO) << "skip EV key:" << *(key_buff + i);
        continue;
      }
      ValuePtr<V>* valptr = kv_->Lookup(key_buff[i], emb_config_.total_num());
      LookupOrCreateV3(valptr, emb_config_, value_buff + i * value_len_, version_buff[i]);
    }
    return Status::OK();
  }

  bool UseDB() const {
    return use_db_;
  }

  bool GetEmbConfig(const std::string& name, EmbeddingConfig& embconfig) {
    return true;
  }
  // end for embedding v3


  typename TTypes<V>::Flat flat(K key, int64 update_version = -1) {
    V* val = LookupOrCreate(key, default_value_, update_version);
    Eigen::array<Eigen::DenseIndex, 1> dims({value_len_});
    return typename TTypes<V>::Flat(val, dims);
  }

  V* LookupOrCreate(K key, V* default_v, int64 update_version = -1) {
    V* val = NULL;
    Status s = kv_->Lookup(key, &val);
    if (!s.ok()) {
      V* new_val = TypedAllocator::Allocate<V>(alloc_, value_and_version_len_, AllocationAttributes());
      memcpy(new_val, default_v, sizeof(V) * value_len_);
      memset(new_val + value_len_, 0, sizeof(V) * (value_and_version_len_ - value_len_));
      if (Status::OK() != kv_->Insert(key, new_val, &val)) {
        TypedAllocator::Deallocate(alloc_, new_val, value_and_version_len_);
      } else {
        val = new_val;
      }
    }
    //LOG(ERROR) << "val: " << val << " " << value_len_ << " " << value_and_version_len_;
    if (update_version != -1) {
      int64* version = reinterpret_cast<int64*>(val + value_len_);
      *version = update_version;
      //LOG(ERROR) << "version: " << *version;
    }
    return val;
  }

  void HybridInsert(K key, const V* val) {
    if (use_db_) {
      db_kv_->Insert(key, val, sizeof(V) * value_len_);
    } else {
      V* new_val = LookupOrCreate(key, default_value_);
      memcpy(new_val, val, sizeof(V) * value_len_);
    }
  }

  void LookupOrCreateHybrid(K key, V* val, V* default_v) {
    if (use_db_) {
      std::string db_value;
      Status st = db_kv_->Lookup(key, &db_value);
      if (!st.ok()) {
        memcpy(val, default_value_, sizeof(V) * value_len_);
        //TypedAllocator::Deallocate(alloc_, new_val, value_and_version_len_);
      } else {
        // found in disk
        memcpy(val, db_value.data(), sizeof(V) * value_len_);
      }
    } else {
      V* mem_val = LookupOrCreate(key, default_v);
      memcpy(val, mem_val, sizeof(V) * value_len_);
    }
  }

  int64 HybridSize() {
    if (use_db_)
      return db_kv_->Size();
    else
      return kv_->Size();
  }

  int64 GetVersion(K key) {
    return *(reinterpret_cast<int64*>(LookupOrCreateV3(key) + ValueLen()));
  }

  Status Shrink(int64 steps_to_live, int64 gs) {
    kv_->Shrink(steps_to_live, gs, value_and_version_len_, value_len_);
    return Status::OK();
  }

  Status Import(const Tensor& keys,
                const Tensor& values,
                const Tensor& versions) {
    // lock all
    const int64 kv_num = keys.dim_size(0);
    auto key_flat = keys.flat<K>();
    auto value_matrix = values.matrix<V>();
    auto version_flat = versions.flat<int64>();
    for (auto i = 0; i < kv_num; ++i) {
      V* new_val = TypedAllocator::Allocate<V>(alloc_, value_and_version_len_, AllocationAttributes());
      memcpy(new_val, &value_matrix(i, 0), sizeof(V) * value_len_);
      memcpy(new_val + value_len_, &version_flat(i), sizeof(V) * (value_and_version_len_ - value_len_));
      V* val = NULL;
      Status st = kv_->Insert(key_flat(i), new_val, &val) ;
      if (!st.ok()) {
        TypedAllocator::Deallocate(alloc_, new_val, value_and_version_len_);
        //LOG(ERROR) << "EV import not expect insert an exist key" << st.error_message();
      }
    }
    return Status::OK();
  }


  Status ImportV2(RestoreBuffer& restore_buff, int64 key_num,
      int bucket_num,
      int64 partition_id,
      int64 partition_num) {
    K* key_buff = (K*)restore_buff.key_buffer;
    V* value_buff = (V*)restore_buff.value_buffer;
    int64* version_buff = (int64*)restore_buff.version_buffer;
    for (auto i = 0; i < key_num; ++i) {
      // this can describe by graph(Mod + DynamicPartition), but memory waste and slow
      if (*(key_buff + i) % bucket_num % partition_num != partition_id) {
        continue;
      }
      V* new_val = TypedAllocator::Allocate<V>(alloc_, value_and_version_len_, AllocationAttributes());
      memcpy(new_val, value_buff + i * value_len_, sizeof(V) * value_len_);
      memcpy(new_val + value_len_,  version_buff + i, sizeof(V) * (value_and_version_len_ - value_len_));
      V* val = NULL;
      Status st = kv_->Insert(key_buff[i], new_val, &val) ;
      if (!st.ok()) {
        TypedAllocator::Deallocate(alloc_, new_val, value_and_version_len_);
        LOG(ERROR) << "EV import not expect insert an exist key" << st.error_message();
      }
    }
    return Status::OK();

  }

  int64 ValueLen() const {
    return value_len_;
  }

  int64 Size() const {
    return kv_->Size();
  }

  void DebugString() const {
    kv_->DebugString();
  }


  int64 GetSnapshot(std::vector<K>* key_list, std::vector<V* >* value_list)
  { return kv_->GetSnapshot(key_list, value_list); }

 private:
  KVInterface<K, V>* kv_;

  // use db
  bool use_db_;
  KVInterface<K, V>* db_kv_;

  V* default_value_;
  int64 value_len_;
  int64 value_and_version_len_;
  Allocator* alloc_;
  mutable mutex mu_;
  EmbeddingConfig emb_config_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_HASHMAP_H_
