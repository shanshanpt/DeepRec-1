#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

#include <sys/resource.h>
#include "tensorflow/core/framework/hashmap.h"
#include "tensorflow/core/kernels/kv_variable_ops.h"
#ifdef TENSORFLOW_USE_JEMALLOC
#include "jemalloc/jemalloc.h"
#endif

namespace tensorflow {
namespace {
const int THREADNUM = 16;
const int64 max = 2147483647;

struct ProcMemory {
  long size;      // total program size
  long resident;  // resident set size
  long share;     // shared pages
  long trs;       // text (code)
  long lrs;       // library
  long drs;       // data/stack
  long dt;        // dirty pages

  ProcMemory() : size(0), resident(0), share(0),
                 trs(0), lrs(0), drs(0), dt(0) {}
};

ProcMemory getProcMemory() {
  ProcMemory m;
  FILE* fp = fopen("/proc/self/statm", "r");
  if (fp == NULL) {
    LOG(ERROR) << "Fail to open /proc/self/statm.";
    return m;
  }

  if (fscanf(fp, "%ld %ld %ld %ld %ld %ld %ld",
             &m.size, &m.resident, &m.share,
             &m.trs, &m.lrs, &m.drs, &m.dt) != 7) {
    fclose(fp);
    LOG(ERROR) << "Fail to fscanf /proc/self/statm.";
    return m;
  }
  fclose(fp);

  return m;
}

double getSize() {
  ProcMemory m = getProcMemory();
  return m.size;
}

double getResident() {
  ProcMemory m = getProcMemory();
  return m.resident;
}

string Prefix(const string& prefix) {
  return strings::StrCat(testing::TmpDir(), "/", prefix);
}

std::vector<string> AllTensorKeys(BundleReader* reader) {
  std::vector<string> ret;
  reader->Seek(kHeaderEntryKey);
  reader->Next();
  for (; reader->Valid(); reader->Next()) {
    //ret.push_back(reader->key().ToString());
    ret.push_back(std::string(reader->key()));
  }
  return ret;
}


TEST(TensorBundleTest, TestEVShrink) {

  int64 value_size = 64;
  int64 insert_num = 30;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));

  HashMap<int64, float>* emb_var = new HashMap<int64, float>(
      new DenseHashMap<int64, float>(), cpu_allocator());
  emb_var ->Init(value);


  LOG(INFO) << "size:" << emb_var->Size();


  for (int64 i=0; i < insert_num; ++i) {
    emb_var->LookupOrCreate(i, fill_v);
  }

  int size = emb_var->Size();
  emb_var->Shrink(5, insert_num);
  LOG(INFO) << "Before shrink size:" << size;
  LOG(INFO) << "After shrink size:" << emb_var->Size();

}

TEST(TensorBundleTest, TestEVShrinkLockless) {

  int64 value_size = 64;
  int64 insert_num = 30;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));

  HashMap<int64, float>* emb_var = new HashMap<int64, float>(
      new DynamicDenseHashMap<int64, float>(), cpu_allocator(), false, EmbeddingConfig(0, 0, 1, 1, "", 5));
  emb_var ->Init(value);


  LOG(INFO) << "size:" << emb_var->Size();


  for (int64 i=0; i < insert_num; ++i) {
    typename TTypes<float>::Flat vflat = emb_var->flatV3ForTest(i, 1);
  }

  int size = emb_var->Size();
  emb_var->Shrink(5, insert_num);

  LOG(INFO) << "Before shrink size:" << size;
  LOG(INFO) << "After shrink size: " << emb_var->Size();

  ASSERT_EQ(size, insert_num);
  ASSERT_EQ(emb_var->Size(), 0);

}


TEST(EmbeddingVariableTest, TestEmptyEV) {
  int64 value_size = 8;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));


  {
    EmbeddingVar<int64, float>* variable
              = new EmbeddingVar<int64, float>("EmbeddingVar",
                  new HashMap<int64, float>(
                    new DenseHashMap<int64, float>(), cpu_allocator()),
                  1);
    variable->Init(value);

    LOG(INFO) << "size:" << variable->hashmap()->Size();
    Tensor part_offset_tensor(DT_INT32,  TensorShape({kSavedPartitionNum + 1}));

    BundleWriter writer(Env::Default(), Prefix("foo"));
    DumpEmbeddingValues(variable, "var/part_0", &writer, &part_offset_tensor);
    TF_ASSERT_OK(writer.Finish());

    {
      BundleReader reader(Env::Default(), Prefix("foo"));
      TF_ASSERT_OK(reader.status());
      EXPECT_EQ(
          AllTensorKeys(&reader),
          std::vector<string>({"var/part_0-keys", "var/part_0-partition_offset", "var/part_0-values", "var/part_0-versions"}));
      {
        string key = "var/part_0-keys";
        EXPECT_TRUE(reader.Contains(key));
        // Tests for LookupDtypeAndShape().
        DataType dtype;
        TensorShape shape;
        TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
        // Tests for Lookup(), checking tensor contents.
        Tensor val(dtype, TensorShape{0});
        TF_ASSERT_OK(reader.Lookup(key, &val));
        LOG(INFO) << "read keys:" << val.DebugString();
      }
      {
        string key = "var/part_0-values";
        EXPECT_TRUE(reader.Contains(key));
        // Tests for LookupDtypeAndShape().
        DataType dtype;
        TensorShape shape;
        TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
        // Tests for Lookup(), checking tensor contents.
        Tensor val(dtype, TensorShape{0, value_size});
        TF_ASSERT_OK(reader.Lookup(key, &val));
        LOG(INFO) << "read values:" << val.DebugString();
      }
      {
        string key = "var/part_0-versions";
        EXPECT_TRUE(reader.Contains(key));
        // Tests for LookupDtypeAndShape().
        DataType dtype;
        TensorShape shape;
        TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
        // Tests for Lookup(), checking tensor contents.
        Tensor val(dtype, TensorShape{0});
        TF_ASSERT_OK(reader.Lookup(key, &val));
        LOG(INFO) << "read versions:" << val.DebugString();
      }
    }
  }
}

TEST(EmbeddingVariableTest, TestEVExportSmall) {

  int64 value_size = 8;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));

  EmbeddingVar<int64, float>* variable
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        new HashMap<int64, float>(
          new DenseHashMap<int64, float>(), cpu_allocator()),
        1);
  variable->Init(value);
  Tensor part_offset_tensor(DT_INT32,  TensorShape({kSavedPartitionNum + 1}));

  for (int64 i = 0; i < 5; i++) {
    typename TTypes<float>::Flat vflat = variable->hashmap()->flat(i, i);
    vflat(i) = 5.0;
  }

  LOG(INFO) << "size:" << variable->hashmap()->Size();


  BundleWriter writer(Env::Default(), Prefix("foo"));
  DumpEmbeddingValues(variable, "var/part_0", &writer, &part_offset_tensor);
  TF_ASSERT_OK(writer.Finish());

  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"var/part_0-keys", "var/part_0-partition_offset", "var/part_0-values", "var/part_0-versions"}));
    {
      string key = "var/part_0-keys";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{5});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read keys:" << val.DebugString();
    }
    {
      string key = "var/part_0-values";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{5, value_size});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read values:" << val.DebugString();
    }
    {
      string key = "var/part_0-versions";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{5});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read versions:" << val.DebugString();
    }

  }

}

TEST(EmbeddingVariableTest, TestEVExportSmallLockless) {

  int64 value_size = 8;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));

  EmbeddingVar<int64, float>* variable
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        new HashMap<int64, float>(
          new DynamicDenseHashMap<int64, float>(), cpu_allocator()),
        1);
  variable->Init(value);

  Tensor part_offset_tensor(DT_INT32,  TensorShape({kSavedPartitionNum + 1}));

  for (int64 i = 0; i < 5; i++) {
    typename TTypes<float>::Flat vflat = variable->hashmap()->flatV3ForTest(i, i);
    vflat(i) = 5.0;
  }

  LOG(INFO) << "size:" << variable->hashmap()->Size();


  BundleWriter writer(Env::Default(), Prefix("foo"));
  DumpEmbeddingValues(variable, "var/part_0", &writer, &part_offset_tensor);
  TF_ASSERT_OK(writer.Finish());

  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"var/part_0-keys", "var/part_0-partition_offset", "var/part_0-values", "var/part_0-versions"}));
    {
      string key = "var/part_0-keys";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{5});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read keys:" << val.DebugString();
    }
    {
      string key = "var/part_0-values";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{5, value_size});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read values:" << val.DebugString();
    }
    {
      string key = "var/part_0-versions";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{5});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read versions:" << val.DebugString();
    }

  }

}

TEST(EmbeddingVariableTest, TestEVExportLarge) {

  int64 value_size = 128;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));

  EmbeddingVar<int64, float>* variable
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        new HashMap<int64, float>(
          new DenseHashMap<int64, float>(), cpu_allocator()),
        0);
  variable->Init(value);
  Tensor part_offset_tensor(DT_INT32,  TensorShape({kSavedPartitionNum + 1}));

  int64 ev_size = 10048576;
  for (int64 i = 0; i < ev_size; i++) {
    variable->hashmap()->LookupOrCreate(i, fill_v);
  }

  LOG(INFO) << "size:" << variable->hashmap()->Size();

  BundleWriter writer(Env::Default(), Prefix("foo"));
  DumpEmbeddingValues(variable, "var/part_0", &writer, &part_offset_tensor);
  TF_ASSERT_OK(writer.Finish());

  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"var/part_0-keys", "var/part_0-partition_offset", "var/part_0-values", "var/part_0-versions"}));

    {
      string key = "var/part_0-keys";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{ev_size});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read keys:" << val.DebugString();
    }
    {
      string key = "var/part_0-values";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{ev_size, value_size});
      LOG(INFO) << "read values:" << val.DebugString();
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read values:" << val.DebugString();
    }
    {
      string key = "var/part_0-versions";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{ev_size});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read versions:" << val.DebugString();
    }


  }
}

TEST(EmbeddingVariableTest, TestEVExportLargeLockless) {

  int64 value_size = 128;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));

  EmbeddingVar<int64, float>* variable
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        new HashMap<int64, float>(
          new DynamicDenseHashMap<int64, float>(), cpu_allocator()),
        0);
  variable->Init(value);

  Tensor part_offset_tensor(DT_INT32,  TensorShape({kSavedPartitionNum + 1}));

  int64 ev_size = 10048576;
  for (int64 i = 0; i < ev_size; i++) {
    typename TTypes<float>::Flat vflat = variable->hashmap()->flatV3ForTest(i, i);
  }

  LOG(INFO) << "size:" << variable->hashmap()->Size();

  BundleWriter writer(Env::Default(), Prefix("foo"));
  DumpEmbeddingValues(variable, "var/part_0", &writer, &part_offset_tensor);
  TF_ASSERT_OK(writer.Finish());

  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"var/part_0-keys", "var/part_0-partition_offset", "var/part_0-values", "var/part_0-versions"}));


    {
      string key = "var/part_0-keys";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{ev_size});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read keys:" << val.DebugString();
    }
    {
      string key = "var/part_0-values";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{ev_size, value_size});
      LOG(INFO) << "read values:" << val.DebugString();
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read values:" << val.DebugString();
    }
    {
      string key = "var/part_0-versions";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{ev_size});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read versions:" << val.DebugString();
    }
  }
}

TEST(EmbeddingVariableTest, TestColdDataStorage) {

  int64 value_size = 4096;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));

  EmbeddingVar<int64, float>* variable
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        new HashMap<int64, float>(
          new DenseHashMap<int64, float>(),
          cpu_allocator(),
          true));

  variable->Init(value);
  LOG(INFO) << "begin write " << variable->hashmap()->HybridSize();

  float* db_values = (float*)malloc(value_size * sizeof(float));
  for (int i = 0; i < 1024; i++) {
    variable->hashmap()->HybridInsert(i, db_values);
  }


  LOG(INFO) << "after write " << variable->hashmap()->HybridSize();
  srand((unsigned) time(NULL));
  float* gather_values = (float*)malloc(value_size * sizeof(float));
  for (int64 i = 0; i < 1024; i++) {
      variable->hashmap()->LookupOrCreateHybrid(i, gather_values, fill_v);
  }

}

void multi_insertion(EmbeddingVar<int64, float>* variable, int64 value_size){
  for (long j = 0; j < 5; j++) {
    variable->hashmap()->flatV3ForTest(j, j);
  }
}

TEST(EmbeddingVariableTest, TestMultiInsertion) {
  int64 value_size = 128;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));

  EmbeddingVar<int64, float>* variable
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        new HashMap<int64, float>(
          new DynamicDenseHashMap<int64, float>(),
          cpu_allocator(),
          true));

  variable->Init(value);

  std::vector<std::thread> insert_threads(THREADNUM);
  for (size_t i = 0 ; i < THREADNUM; i++) {
    insert_threads[i] = std::thread(multi_insertion,variable, value_size);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  std::vector<int64> tot_key_list;
  std::vector<float* > tot_valueptr_list;
  std::vector<int64> tot_version_list;
  HashMap<int64, float>* hash_map = variable->hashmap();
  int64 total_size = hash_map->GetSnapshot(&tot_key_list, &tot_valueptr_list, &tot_version_list);

  ASSERT_EQ(variable->hashmap()->Size(), 5);
  ASSERT_EQ(variable->hashmap()->Size(), total_size);
}

void InsertAndLookup(EmbeddingVar<int64, int64>* variable, int64 *keys, long ReadLoops, int value_size){
  for (long j = 0; j < ReadLoops; j++) {
    int64 *val = (int64 *)malloc((value_size+1)*sizeof(int64));
    variable->hashmap()->LookupOrCreateHybridV3(keys[j], val, &(keys[j]));
    variable->hashmap()->LookupOrCreateHybridV3(keys[j], val, (&keys[j]+1));
    ASSERT_EQ(keys[j] , val[0]);
    free(val);
  }
}

TEST(EmbeddingVariableTest, TestInsertAndLookup) {
  int64 value_size = 128;
  Tensor value(DT_INT64, TensorShape({value_size}));
  test::FillValues<int64>(&value, std::vector<int64>(value_size, 10));
 // float* fill_v = (int64*)malloc(value_size * sizeof(int64));

  EmbeddingVar<int64, int64>* variable
    = new EmbeddingVar<int64, int64>("EmbeddingVar",
        new HashMap<int64, int64>(
          new DynamicDenseHashMap<int64, int64>(),
          cpu_allocator(),
          false));

  variable->Init(value);

  int64 InsertLoops = 1000;
  bool* flag = (bool *)malloc(sizeof(bool)*max);
  srand((unsigned)time(NULL));
  int64 *keys = (int64 *)malloc(sizeof(int64)*InsertLoops);
  long *counter = (long *)malloc(sizeof(long)*InsertLoops);

  for (long i = 0; i < max; i++) {
    flag[i] = 0;
  }

  for (long i = 0; i < InsertLoops; i++) {
    counter[i] = 1;
  }
  int index = 0;
  while (index < InsertLoops) {
    long j = rand() % max;
    if (flag[j] == 1) // the number is already set as a key
      continue;
    else { // the number is not selected as a key
      keys[index] = j;
      index++;
      flag[j] = 1;
    }
  }
  free(flag);
  std::vector<std::thread> insert_threads(THREADNUM);
  for (size_t i = 0 ; i < THREADNUM; i++) {
    insert_threads[i] = std::thread(InsertAndLookup, variable, &keys[i*InsertLoops/THREADNUM], InsertLoops/THREADNUM, value_size);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

}

EmbeddingVar<int64, float>* InitEV_Lockless(int64 value_size) {
  Tensor value(DT_INT64, TensorShape({value_size}));
  test::FillValues<int64>(&value, std::vector<int64>(value_size, 10));

  EmbeddingVar<int64, float>* variable
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        new HashMap<int64, float>(
          new DynamicDenseHashMap<int64, float>(),
          cpu_allocator(),
          false));

  variable->Init(value);
  return variable;
}

void MultiLookup_lockless(EmbeddingVar<int64, float>* variable, int64 InsertLoop, int thread_num, int i) {
  for (int64 j = i * InsertLoop/thread_num; j < (i+1)*InsertLoop/thread_num; j++) {
    variable->hashmap()->LookupValuePtr(j);
  }
}

void MultiLookup(EmbeddingVar<int64, float>* variable, int64 InsertLoop, int thread_num, int i) {
  for (int64 j = i * InsertLoop/thread_num; j < (i+1)*InsertLoop/thread_num; j++) {
    variable->hashmap()->LookupOrCreate(j, nullptr);
  }
}

void BM_MULTIREAD_LOCKLESS(int iters, int thread_num) {
  testing::StopTiming();
  testing::UseRealTime();

  int64 value_size = 128;
  EmbeddingVar<int64, float>* variable = InitEV_Lockless(value_size);
  int64 InsertLoop =  1000000;

  float* fill_v = (float*)malloc(value_size * sizeof(float));

  for (int64 i = 0; i < InsertLoop; i++){
    variable->hashmap()->flatV3ForTest(i, i);
  }

  testing::StartTiming();
  while(iters--){
    std::vector<std::thread> insert_threads(thread_num);
    for (size_t i = 0 ; i < thread_num; i++) {
      insert_threads[i] = std::thread(MultiLookup_lockless, variable, InsertLoop, thread_num, i);
    }
    for (auto &t : insert_threads) {
      t.join();
    }
  }

}

EmbeddingVar<int64, float>* InitEV(int64 value_size) {
  Tensor value(DT_INT64, TensorShape({value_size}));
  test::FillValues<int64>(&value, std::vector<int64>(value_size, 10));

  EmbeddingVar<int64, float>* variable
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        new HashMap<int64, float>(
          new DenseHashMap<int64, float>(),
          cpu_allocator(),
          true));

  variable->Init(value);
  return variable;
}


void BM_MULTIREAD(int iters, int thread_num) {
  testing::StopTiming();
  testing::UseRealTime();

  int64 value_size = 128;
  EmbeddingVar<int64, float>* variable = InitEV(value_size);
  int64 InsertLoop =  1000000;
  float* fill_v = (float*)malloc(value_size * sizeof(float));

  for (int64 i = 0; i < InsertLoop; i++){
    variable->hashmap()->LookupOrCreate(i, fill_v);
  }

  testing::StartTiming();
  while(iters--){
    std::vector<std::thread> insert_threads(thread_num);
    for (size_t i = 0 ; i < thread_num; i++) {
      insert_threads[i] = std::thread(MultiLookup, variable, InsertLoop, thread_num, i);
    }
    for (auto &t : insert_threads) {
      t.join();
    }
  }
}
void hybrid_process_lockless(EmbeddingVar<int64, float>* variable, int64* keys, int64 InsertLoop, int thread_num, int64 i, int64 value_size) {
  float *val = (float *)malloc(sizeof(float)*(value_size + 1));
  for (int64 j = i * InsertLoop/thread_num; j < (i+1) * InsertLoop/thread_num; j++) {
    variable->hashmap()->LookupOrCreateHybridV3(keys[j], val, nullptr);
  }
}

void hybrid_process(EmbeddingVar<int64, float>* variable, int64* keys, int64 InsertLoop, int thread_num, int64 i) {
  for (int64 j = i * InsertLoop/thread_num; j < (i+1) * InsertLoop/thread_num; j++) {
    variable->hashmap()->LookupOrCreate(keys[j], nullptr);
  }
}

void BM_HYBRID_LOCKLESS(int iters, int thread_num) {
  testing::StopTiming();
  testing::UseRealTime();

  int64 value_size = 128;
  EmbeddingVar<int64, float>* variable = InitEV_Lockless(value_size);
  int64 InsertLoop =  1000000;

  srand((unsigned)time(NULL));
  int64 *keys = (int64 *)malloc(sizeof(int64)*InsertLoop);

  for (int64 i = 0; i < InsertLoop; i++) {
    keys[i] =  rand() % 1000;
  }

  testing::StartTiming();
  while (iters--) {
    std::vector<std::thread> insert_threads(thread_num);
    for (size_t i = 0 ; i < thread_num; i++) {
      insert_threads[i] = std::thread(hybrid_process_lockless, variable, keys, InsertLoop, thread_num, i, value_size);
    }
    for (auto &t : insert_threads) {
      t.join();
    }
  }
}

void BM_HYBRID(int iters, int thread_num) {
  testing::StopTiming();
  testing::UseRealTime();

  int64 value_size = 128;
  EmbeddingVar<int64, float>* variable = InitEV(value_size);
  int64 InsertLoop =  1000000;

  srand((unsigned)time(NULL));
  int64 *keys = (int64 *)malloc(sizeof(int64)*InsertLoop);

  for (int64 i = 0; i < InsertLoop; i++) {
    keys[i] =  rand() % 1000;
  }

  testing::StartTiming();
  while (iters--) {
    std::vector<std::thread> insert_threads(thread_num);
    for (size_t i = 0 ; i < thread_num; i++) {
      insert_threads[i] = std::thread(hybrid_process, variable, keys, InsertLoop, thread_num, i);
    }
    for (auto &t : insert_threads) {
      t.join();
    }
  }
}

BENCHMARK(BM_MULTIREAD_LOCKLESS)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16);

BENCHMARK(BM_MULTIREAD)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16);
BENCHMARK(BM_HYBRID_LOCKLESS)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16);
BENCHMARK(BM_HYBRID)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16);

} // namespace
} // namespace tensorflow