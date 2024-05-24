#include "matxscript/runtime/codegen_all_includes.h"
#include <math.h>

using namespace ::matxscript::runtime;
extern "C" void* __matxscript_module_ctx = NULL;

extern "C" MATX_DLL MATXScriptFuncRegistry __matxscript_func_registry__;

extern "C" MATX_DLL MATXScriptFuncRegistry __matxscript_func_registry__SimpleCutter;
extern "C" MATX_DLL MATXScriptFuncRegistry __matxscript_func_registry__Cutter;
namespace {
// User class forward declarations
struct SimpleCutter;
struct SimpleCutter_SharedView;

// User class forward declarations
struct Cutter;
struct Cutter_SharedView;

SimpleCutter_SharedView SimpleCutter__F___init___wrapper(const unicode_view& cut_type, const unicode_view& location, const unicode_view& cut_level, void* handle_2_71828182846=((void*)(int64_t)0));
int SimpleCutter__F___init___wrapper__c_api(MATXScriptAny*, int, MATXScriptAny*, void*);
Cutter_SharedView Cutter__F___init___wrapper(const unicode_view& cut_type, const unicode_view& resource_path, void* handle_2_71828182846=((void*)(int64_t)0));
int Cutter__F___init___wrapper__c_api(MATXScriptAny*, int, MATXScriptAny*, void*);
MATX_DLL Tuple Cutter__F_segment_str(const Cutter_SharedView& self, const RTView& text, bool keep_and_trans_space=(bool)0, int64_t timeout=(int64_t)-1, const unicode_view& config=unicode_view());
int Cutter__F_segment_str__c_api(MATXScriptAny*, int, MATXScriptAny*, void*);
MATX_DLL RTValue Cutter__F___init__(const Cutter_SharedView& self, const unicode_view& cut_type, const unicode_view& resource_path, void* handle_2_71828182846=((void*)(int64_t)0));
int Cutter__F___init____c_api(MATXScriptAny*, int, MATXScriptAny*, void*);
MATX_DLL Tuple Cutter__F_segment(const Cutter_SharedView& self, const RTView& text, bool keep_and_trans_space=(bool)0, const unicode_view& config=unicode_view());
int Cutter__F_segment__c_api(MATXScriptAny*, int, MATXScriptAny*, void*);
MATX_DLL Unicode SimpleCutter__F___call__(const SimpleCutter_SharedView& self, const unicode_view& text);
int SimpleCutter__F___call____c_api(MATXScriptAny*, int, MATXScriptAny*, void*);
MATX_DLL RTValue SimpleCutter__F___init__(const SimpleCutter_SharedView& self, const unicode_view& cut_type, const unicode_view& location, const unicode_view& cut_level, void* handle_2_71828182846=((void*)(int64_t)0));
int SimpleCutter__F___init____c_api(MATXScriptAny*, int, MATXScriptAny*, void*);
MATX_DLL List Cutter__F_cut(const Cutter_SharedView& self, const RTView& text, const unicode_view& cut_level=unicode_view(U"\u0044\u0045\u0046\u0041\u0055\u004C\u0054", 7), bool keep_and_trans_space=(bool)0, const unicode_view& config=unicode_view());
int Cutter__F_cut__c_api(MATXScriptAny*, int, MATXScriptAny*, void*);
MATX_DLL List Cutter__F___call__(const Cutter_SharedView& self, const RTView& text, const unicode_view& cut_level=unicode_view(U"\u0044\u0045\u0046\u0041\u0055\u004C\u0054", 7), bool keep_and_trans_space=(bool)0, const unicode_view& config=unicode_view());
int Cutter__F___call____c_api(MATXScriptAny*, int, MATXScriptAny*, void*);
struct SimpleCutter : public IUserDataRoot {
  // flags for convert check
  static uint32_t tag_s_2_71828182846_;
  static uint32_t var_num_s_2_71828182846_;
  static string_view class_name_s_2_71828182846_;
  static IUserDataRoot::__FunctionTable__ function_table_s_2_71828182846_;

  // override meta functions
  const char* ClassName_2_71828182846() const override { return "SimpleCutter"; }
  uint32_t tag_2_71828182846() const override { return tag_s_2_71828182846_; }
  uint32_t size_2_71828182846() const override { return var_num_s_2_71828182846_; }

  bool isinstance_2_71828182846(uint64_t tag) override {
    static std::initializer_list<uint64_t> all_tags = {SimpleCutter::tag_s_2_71828182846_};
    return std::find(all_tags.begin(), all_tags.end(), tag) != all_tags.end();
  }

  std::initializer_list<string_view> VarNames_2_71828182846() const override {
    static std::initializer_list<string_view> __var_names_s__ = {"cutter", "cut_level", };
    return __var_names_s__;
  }

  const ska::flat_hash_map<string_view, int64_t>& VarTable_2_71828182846() const override {
    static ska::flat_hash_map<string_view, int64_t> __var_table_s__ = {
      {"cutter", 0}, 
      {"cut_level", 1}, 
    };
    return __var_table_s__;
  }

  // member vars
  UserDataRef cutter;
  Unicode cut_level;

  // Object pointer
  Object* self_node_ptr_2_71828182846 = nullptr;

  // override GetVar_2_71828182846 functions
  RTView GetVar_2_71828182846(int64_t idx) const override {
    switch (idx) {
    case 0: { return cutter; } break;
    case 1: { return cut_level; } break;
    default: { THROW_PY_IndexError("index overflow"); return nullptr; } break;

    }
  }
  // override SetVar_2_71828182846 functions
  void SetVar_2_71828182846(int64_t idx, const Any& val) override {
    switch (idx) {
    case 0: { this->cutter = internal::TypeAsHelper<UserDataRef>::run((val), __FILE__, __LINE__, nullptr, "expect 'val' is 'UserDataRef' type"); } break;
    case 1: { this->cut_level = internal::TypeAsHelper<Unicode>::run((val), __FILE__, __LINE__, nullptr, "expect 'val' is 'py::str' type"); } break;
    default: { THROW_PY_IndexError("index overflow"); } break;

    }
  }

  // virtual methods
  virtual RTValue __init__(const unicode_view& cut_type, const unicode_view& location, const unicode_view& cut_level, void* handle_2_71828182846=((void*)(int64_t)0));
  virtual Unicode __call__(const unicode_view& text);
};

// flags for convert check
uint32_t SimpleCutter::tag_s_2_71828182846_ = -404898741031726452;
uint32_t SimpleCutter::var_num_s_2_71828182846_ = 2;
string_view SimpleCutter::class_name_s_2_71828182846_ = "SimpleCutter";
IUserDataRoot::__FunctionTable__ SimpleCutter::function_table_s_2_71828182846_ = IUserDataRoot::InitFuncTable_2_71828182846(&__matxscript_func_registry__SimpleCutter, "SimpleCutter");

struct SimpleCutter_SharedView: public IUserDataSharedViewRoot {
  // member var
  SimpleCutter *ptr;
  // constructor
  SimpleCutter_SharedView(SimpleCutter *ptr, UserDataRef ref) : ptr(ptr), IUserDataSharedViewRoot(std::move(ref)) {}
  SimpleCutter_SharedView(SimpleCutter *ptr) : ptr(ptr) {}
  SimpleCutter_SharedView() : ptr(nullptr) {}
  SimpleCutter_SharedView(const matxscript::runtime::Any& ref) : SimpleCutter_SharedView(MATXSCRIPT_TYPE_AS(ref, UserDataRef)) {}
  // UserDataRef
  SimpleCutter_SharedView(UserDataRef ref) {
    IUserDataRoot* base_ud_ptr = static_cast<IUserDataRoot*>(ref.check_codegen_ptr("SimpleCutter"));
    if(!base_ud_ptr->isinstance_2_71828182846(SimpleCutter::tag_s_2_71828182846_)) {THROW_PY_TypeError("expect 'SimpleCutter' but get '", base_ud_ptr->ClassName_2_71828182846(), "'");}
    ptr = static_cast<SimpleCutter*>(base_ud_ptr);
    ud_ref = std::move(ref);
  }
  SimpleCutter* operator->() const { return ptr; }
  template <typename T, typename = typename std::enable_if<std::is_convertible<UserDataRef, T>::value>::type>
  operator T() const {return ud_ref;}
};

struct Cutter : public IUserDataRoot {
  // flags for convert check
  static uint32_t tag_s_2_71828182846_;
  static uint32_t var_num_s_2_71828182846_;
  static string_view class_name_s_2_71828182846_;
  static IUserDataRoot::__FunctionTable__ function_table_s_2_71828182846_;

  // override meta functions
  const char* ClassName_2_71828182846() const override { return "Cutter"; }
  uint32_t tag_2_71828182846() const override { return tag_s_2_71828182846_; }
  uint32_t size_2_71828182846() const override { return var_num_s_2_71828182846_; }

  bool isinstance_2_71828182846(uint64_t tag) override {
    static std::initializer_list<uint64_t> all_tags = {Cutter::tag_s_2_71828182846_};
    return std::find(all_tags.begin(), all_tags.end(), tag) != all_tags.end();
  }

  std::initializer_list<string_view> VarNames_2_71828182846() const override {
    static std::initializer_list<string_view> __var_names_s__ = {"cutter", };
    return __var_names_s__;
  }

  const ska::flat_hash_map<string_view, int64_t>& VarTable_2_71828182846() const override {
    static ska::flat_hash_map<string_view, int64_t> __var_table_s__ = {
      {"cutter", 0}, 
    };
    return __var_table_s__;
  }

  // member vars
  UserDataRef cutter;

  // Object pointer
  Object* self_node_ptr_2_71828182846 = nullptr;

  // override GetVar_2_71828182846 functions
  RTView GetVar_2_71828182846(int64_t idx) const override {
    switch (idx) {
    case 0: { return cutter; } break;
    default: { THROW_PY_IndexError("index overflow"); return nullptr; } break;

    }
  }
  // override SetVar_2_71828182846 functions
  void SetVar_2_71828182846(int64_t idx, const Any& val) override {
    switch (idx) {
    case 0: { this->cutter = internal::TypeAsHelper<UserDataRef>::run((val), __FILE__, __LINE__, nullptr, "expect 'val' is 'UserDataRef' type"); } break;
    default: { THROW_PY_IndexError("index overflow"); } break;

    }
  }

  // virtual methods
  virtual RTValue __init__(const unicode_view& cut_type, const unicode_view& resource_path, void* handle_2_71828182846=((void*)(int64_t)0));
  virtual Tuple segment(const RTView& text, bool keep_and_trans_space=(bool)0, const unicode_view& config=unicode_view());
  virtual Tuple segment_str(const RTView& text1, bool keep_and_trans_space1=(bool)0, int64_t timeout=(int64_t)-1, const unicode_view& config1=unicode_view());
  virtual List cut(const RTView& text2, const unicode_view& cut_level=unicode_view(U"\u0044\u0045\u0046\u0041\u0055\u004C\u0054", 7), bool keep_and_trans_space2=(bool)0, const unicode_view& config2=unicode_view());
  virtual List __call__(const RTView& text3, const unicode_view& cut_level1=unicode_view(U"\u0044\u0045\u0046\u0041\u0055\u004C\u0054", 7), bool keep_and_trans_space3=(bool)0, const unicode_view& config3=unicode_view());
};

// flags for convert check
uint32_t Cutter::tag_s_2_71828182846_ = -5990847626626997806;
uint32_t Cutter::var_num_s_2_71828182846_ = 1;
string_view Cutter::class_name_s_2_71828182846_ = "Cutter";
IUserDataRoot::__FunctionTable__ Cutter::function_table_s_2_71828182846_ = IUserDataRoot::InitFuncTable_2_71828182846(&__matxscript_func_registry__Cutter, "Cutter");

struct Cutter_SharedView: public IUserDataSharedViewRoot {
  // member var
  Cutter *ptr;
  // constructor
  Cutter_SharedView(Cutter *ptr, UserDataRef ref) : ptr(ptr), IUserDataSharedViewRoot(std::move(ref)) {}
  Cutter_SharedView(Cutter *ptr) : ptr(ptr) {}
  Cutter_SharedView() : ptr(nullptr) {}
  Cutter_SharedView(const matxscript::runtime::Any& ref) : Cutter_SharedView(MATXSCRIPT_TYPE_AS(ref, UserDataRef)) {}
  // UserDataRef
  Cutter_SharedView(UserDataRef ref) {
    IUserDataRoot* base_ud_ptr = static_cast<IUserDataRoot*>(ref.check_codegen_ptr("Cutter"));
    if(!base_ud_ptr->isinstance_2_71828182846(Cutter::tag_s_2_71828182846_)) {THROW_PY_TypeError("expect 'Cutter' but get '", base_ud_ptr->ClassName_2_71828182846(), "'");}
    ptr = static_cast<Cutter*>(base_ud_ptr);
    ud_ref = std::move(ref);
  }
  Cutter* operator->() const { return ptr; }
  template <typename T, typename = typename std::enable_if<std::is_convertible<UserDataRef, T>::value>::type>
  operator T() const {return ud_ref;}
};

void SimpleCutter_F__deleter__(ILightUserData* ptr) { delete static_cast<SimpleCutter*>(ptr); }
void* SimpleCutter_F__placement_new__(void* buf) { return new (buf) SimpleCutter; }
void SimpleCutter_F__placement_del__(ILightUserData* ptr) { static_cast<SimpleCutter*>(ptr)->SimpleCutter::~SimpleCutter(); }
SimpleCutter_SharedView SimpleCutter__F___init___wrapper(const unicode_view& cut_type, const unicode_view& location, const unicode_view& cut_level, void* handle_2_71828182846) {
  static auto buffer_size = UserDataRef::GetInternalBufferSize();
  if (buffer_size < sizeof(SimpleCutter)) {
    auto self = new SimpleCutter;
    self->function_table_2_71828182846_ = &SimpleCutter::function_table_s_2_71828182846_;
    SimpleCutter__F___init__(self,  cut_type,  location,  cut_level,  handle_2_71828182846);
    UserDataRef self_ref(SimpleCutter::tag_s_2_71828182846_, SimpleCutter::var_num_s_2_71828182846_, self, SimpleCutter_F__deleter__, __matxscript_module_ctx);
    self->self_node_ptr_2_71828182846 = (Object*)(self_ref.get());
    return self_ref;
  } else {
    UserDataRef self(SimpleCutter::tag_s_2_71828182846_, SimpleCutter::var_num_s_2_71828182846_, sizeof(SimpleCutter), SimpleCutter_F__placement_new__, SimpleCutter_F__placement_del__, __matxscript_module_ctx);
    SimpleCutter_SharedView self_view((SimpleCutter*)self.ud_ptr_nocheck());
    self_view->function_table_2_71828182846_ = &SimpleCutter::function_table_s_2_71828182846_;
    SimpleCutter__F___init__(self_view,  cut_type,  location,  cut_level,  handle_2_71828182846);
    self_view->self_node_ptr_2_71828182846 = (Object*)(self.get());
    return self;
  }
}

void Cutter_F__deleter__(ILightUserData* ptr) { delete static_cast<Cutter*>(ptr); }
void* Cutter_F__placement_new__(void* buf) { return new (buf) Cutter; }
void Cutter_F__placement_del__(ILightUserData* ptr) { static_cast<Cutter*>(ptr)->Cutter::~Cutter(); }
Cutter_SharedView Cutter__F___init___wrapper(const unicode_view& cut_type, const unicode_view& resource_path, void* handle_2_71828182846) {
  static auto buffer_size = UserDataRef::GetInternalBufferSize();
  if (buffer_size < sizeof(Cutter)) {
    auto self = new Cutter;
    self->function_table_2_71828182846_ = &Cutter::function_table_s_2_71828182846_;
    Cutter__F___init__(self,  cut_type,  resource_path,  handle_2_71828182846);
    UserDataRef self_ref(Cutter::tag_s_2_71828182846_, Cutter::var_num_s_2_71828182846_, self, Cutter_F__deleter__, __matxscript_module_ctx);
    self->self_node_ptr_2_71828182846 = (Object*)(self_ref.get());
    return self_ref;
  } else {
    UserDataRef self(Cutter::tag_s_2_71828182846_, Cutter::var_num_s_2_71828182846_, sizeof(Cutter), Cutter_F__placement_new__, Cutter_F__placement_del__, __matxscript_module_ctx);
    Cutter_SharedView self_view((Cutter*)self.ud_ptr_nocheck());
    self_view->function_table_2_71828182846_ = &Cutter::function_table_s_2_71828182846_;
    Cutter__F___init__(self_view,  cut_type,  resource_path,  handle_2_71828182846);
    self_view->self_node_ptr_2_71828182846 = (Object*)(self.get());
    return self;
  }
}

Tuple Cutter::segment_str(const RTView& text, bool keep_and_trans_space, int64_t timeout, const unicode_view& config) {  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:54
  (void)unicode_view(U"\u000A\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0063\u0075\u0074\u0020\u0074\u0065\u0078\u0074\u0020\u0074\u006F\u0020\u0074\u0065\u0072\u006D\u0020\u006C\u0069\u0073\u0074\u000A\u000A\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u003A\u0070\u0061\u0072\u0061\u006D\u0020\u0074\u0065\u0078\u0074\u003A\u000A\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u003A\u0072\u0065\u0074\u0075\u0072\u006E\u003A\u0020\u0066\u0069\u006E\u0065\u005F\u0073\u0074\u0072\u002C\u0020\u0063\u006F\u0061\u0072\u0073\u0065\u005F\u0073\u0074\u0072\u000A\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020", 99);  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:60
  return (internal::TypeAsHelper<Tuple>::run(((this->cutter).generic_call_attr("segment_str", {(text), (keep_and_trans_space), (timeout), (config)})), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 61, in segment_str\n", "expect '(this->cutter).generic_call_attr(\"segment_str\", {(text), (keep_and_trans_space), (timeout), (config)})' is 'Tuple' type"));  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:61
}

MATX_DLL Tuple Cutter__F_segment_str(const Cutter_SharedView& self, const RTView& text, bool keep_and_trans_space, int64_t timeout, const unicode_view& config) {  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:54
  return (self->segment_str(text, keep_and_trans_space, timeout, config));  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:54
}

int Cutter__F_segment_str__c_api(MATXScriptAny* args, int num_args, MATXScriptAny* out_ret_value, void* resource_handle = nullptr)
{
  TArgs args_t(args, num_args);

  if (num_args > 0 && args[num_args - 1].code == TypeIndex::kRuntimeKwargs) {
    string_view arg_names[5] {"self", "text", "keep_and_trans_space", "timeout", "config"};
    static RTValue default_args[3]{RTValue((bool)0), RTValue((int64_t)-1), RTValue(unicode_view())};
    KwargsUnpackHelper helper("segment_str", arg_names, 5, default_args, 3);
    RTView pos_args[5];
    helper.unpack(pos_args, args, num_args);  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:54
    auto ret = Cutter__F_segment_str(internal::TypeAsHelper<UserDataRef>::run((pos_args[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 54, in segment_str\n", "expect 'self' is 'UserDataRef' type"), pos_args[1], internal::TypeAsHelper<bool>::run((pos_args[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 54, in segment_str\n", "expect 'keep_and_trans_space' is 'bool' type"), internal::TypeAsHelper<int64_t>::run((pos_args[3]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 54, in segment_str\n", "expect 'timeout' is 'int64_t' type"), internal::TypeAsHelper<unicode_view>::run((pos_args[4]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 54, in segment_str\n", "expect 'config' is 'py::str' type"));  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:54
    RTValue(std::move(ret)).MoveToCHost(out_ret_value);
  } else {
    switch(num_args) {
      case 2: {
        auto ret = Cutter__F_segment_str(internal::TypeAsHelper<UserDataRef>::run((args_t[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 54, in segment_str\n", "expect 'self' is 'UserDataRef' type"), args_t[1], (bool)0, (int64_t)-1, unicode_view());  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:54
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      } break;
      case 3: {
        auto ret = Cutter__F_segment_str(internal::TypeAsHelper<UserDataRef>::run((args_t[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 54, in segment_str\n", "expect 'self' is 'UserDataRef' type"), args_t[1], internal::TypeAsHelper<bool>::run((args_t[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 54, in segment_str\n", "expect 'keep_and_trans_space' is 'bool' type"), (int64_t)-1, unicode_view());  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:54
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      } break;
      case 4: {
        auto ret = Cutter__F_segment_str(internal::TypeAsHelper<UserDataRef>::run((args_t[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 54, in segment_str\n", "expect 'self' is 'UserDataRef' type"), args_t[1], internal::TypeAsHelper<bool>::run((args_t[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 54, in segment_str\n", "expect 'keep_and_trans_space' is 'bool' type"), internal::TypeAsHelper<int64_t>::run((args_t[3]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 54, in segment_str\n", "expect 'timeout' is 'int64_t' type"), unicode_view());  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:54
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      } break;
      case 5: {
        auto ret = Cutter__F_segment_str(internal::TypeAsHelper<UserDataRef>::run((args_t[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 54, in segment_str\n", "expect 'self' is 'UserDataRef' type"), args_t[1], internal::TypeAsHelper<bool>::run((args_t[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 54, in segment_str\n", "expect 'keep_and_trans_space' is 'bool' type"), internal::TypeAsHelper<int64_t>::run((args_t[3]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 54, in segment_str\n", "expect 'timeout' is 'int64_t' type"), internal::TypeAsHelper<unicode_view>::run((args_t[4]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 54, in segment_str\n", "expect 'config' is 'py::str' type"));  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:54
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      } break;
      default: {THROW_PY_TypeError("File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 54, in segment_str\n", "segment_str() takes from 2 to 5 positional arguments but ", num_args, " were given");} break;  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:54
    }
  }

  return 0;
}

RTValue Cutter::__init__(const unicode_view& cut_type, const unicode_view& resource_path, void* handle_2_71828182846) {  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:31
  this->session_handle_2_71828182846_ = handle_2_71828182846;  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:31
  (void)unicode_view(U"\u000A\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u006D\u0061\u006B\u0065\u0020\u0061\u0020\u0063\u0075\u0074\u0074\u0065\u0072\u0020\u006F\u0062\u006A\u0065\u0063\u0074\u0020\u0062\u0079\u0020\u0074\u0079\u0070\u0065\u0020\u0061\u006E\u0064\u0020\u0064\u0061\u0074\u0061\u0020\u0070\u0061\u0074\u0068\u000A\u000A\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u003A\u0070\u0061\u0072\u0061\u006D\u0020\u0063\u0075\u0074\u005F\u0074\u0079\u0070\u0065\u003A\u0020\u0060\u0060\u0073\u0074\u0072\u0060\u0060\u000A\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0022\u0043\u0052\u0046\u005F\u004C\u0041\u0052\u0047\u0045\u0022\u002C\u0020\u0022\u004C\u004D\u005F\u0043\u0052\u0046\u0022\u002C\u0020\u0022\u004C\u004D\u005F\u0052\u0045\u0043\u0022\u000A\u000A\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u003A\u0070\u0061\u0072\u0061\u006D\u0020\u0072\u0065\u0073\u006F\u0075\u0072\u0063\u0065\u005F\u0070\u0061\u0074\u0068\u003A\u0020\u0060\u0060\u0073\u0074\u0072\u0060\u0060\u000A\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u006C\u0069\u0062\u0063\u0075\u0074\u0020\u006D\u006F\u0064\u0065\u006C\u0020\u0070\u0061\u0074\u0068\u000A\u000A\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020", 216);  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:41
  this->cutter = make_native_userdata(string_view("LibcutCutter", 12), cut_type, resource_path);  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:42
  return (None);
}

MATX_DLL RTValue Cutter__F___init__(const Cutter_SharedView& self, const unicode_view& cut_type, const unicode_view& resource_path, void* handle_2_71828182846) {  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:31
  return (self->__init__(cut_type, resource_path, handle_2_71828182846));  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:31
}

int Cutter__F___init____c_api(MATXScriptAny* args, int num_args, MATXScriptAny* out_ret_value, void* resource_handle = nullptr)
{
  TArgs args_t(args, num_args);

  if (num_args > 0 && args[num_args - 1].code == TypeIndex::kRuntimeKwargs) {
    string_view arg_names[3] {"self", "cut_type", "resource_path"};
    KwargsUnpackHelper helper("__init__", arg_names, 3, nullptr, 0);
    RTView pos_args[3];
    helper.unpack(pos_args, args, num_args);  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:31
    auto ret = Cutter__F___init__(internal::TypeAsHelper<UserDataRef>::run((pos_args[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 31, in __init__\n", "expect 'self' is 'UserDataRef' type"), internal::TypeAsHelper<unicode_view>::run((pos_args[1]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 31, in __init__\n", "expect 'cut_type' is 'py::str' type"), internal::TypeAsHelper<unicode_view>::run((pos_args[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 31, in __init__\n", "expect 'resource_path' is 'py::str' type"), resource_handle);  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:31
    RTValue(std::move(ret)).MoveToCHost(out_ret_value);
  } else {
    switch(num_args) {
      case 3: {
        auto ret = Cutter__F___init__(internal::TypeAsHelper<UserDataRef>::run((args_t[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 31, in __init__\n", "expect 'self' is 'UserDataRef' type"), internal::TypeAsHelper<unicode_view>::run((args_t[1]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 31, in __init__\n", "expect 'cut_type' is 'py::str' type"), internal::TypeAsHelper<unicode_view>::run((args_t[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 31, in __init__\n", "expect 'resource_path' is 'py::str' type"), resource_handle);  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:31
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      } break;
      default: {THROW_PY_TypeError("File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 31, in __init__\n", "__init__() takes 3 positional arguments but ", num_args, " were given");} break;  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:31
    }
  }

  return 0;
}

int Cutter__F___init___wrapper__c_api(MATXScriptAny* args, int num_args, MATXScriptAny* out_ret_value, void* resource_handle = nullptr)
{
  TArgs args_t(args, num_args);

  if (num_args > 0 && args[num_args - 1].code == TypeIndex::kRuntimeKwargs) {
    string_view arg_names[2] {"cut_type", "resource_path"};
    KwargsUnpackHelper helper("__init__", arg_names, 2, nullptr, 0);
    RTView pos_args[2];
    helper.unpack(pos_args, args, num_args);  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:31
    auto ret = Cutter__F___init___wrapper(internal::TypeAsHelper<unicode_view>::run((pos_args[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 31, in __init__\n", "expect 'cut_type' is 'py::str' type"), internal::TypeAsHelper<unicode_view>::run((pos_args[1]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 31, in __init__\n", "expect 'resource_path' is 'py::str' type"), resource_handle);  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:31
    (ret.operator RTValue()).MoveToCHost(out_ret_value);
  } else {
    switch(num_args) {
      case 2: {
        auto ret = Cutter__F___init___wrapper(internal::TypeAsHelper<unicode_view>::run((args_t[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 31, in __init__\n", "expect 'cut_type' is 'py::str' type"), internal::TypeAsHelper<unicode_view>::run((args_t[1]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 31, in __init__\n", "expect 'resource_path' is 'py::str' type"), resource_handle);  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:31
        (ret.operator RTValue()).MoveToCHost(out_ret_value);
      } break;
      default: {THROW_PY_TypeError("File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 31, in __init__\n", "__init__() takes 2 positional arguments but ", num_args, " were given");} break;  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:31
    }
  }

  return 0;
}

Tuple Cutter::segment(const RTView& text, bool keep_and_trans_space, const unicode_view& config) {  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:45
  (void)unicode_view(U"\u000A\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0063\u0075\u0074\u0020\u0074\u0065\u0078\u0074\u0020\u0074\u006F\u0020\u0074\u0065\u0072\u006D\u0020\u006C\u0069\u0073\u0074\u000A\u000A\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u003A\u0070\u0061\u0072\u0061\u006D\u0020\u0074\u0065\u0078\u0074\u003A\u0020\u0060\u0060\u0073\u0074\u0072\u0060\u0060\u000A\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u003A\u0072\u0065\u0074\u0075\u0072\u006E\u003A\u0020\u0066\u0069\u006E\u0065\u005F\u0074\u0065\u0072\u006D\u0073\u002C\u0020\u0063\u006F\u0061\u0072\u0073\u0065\u005F\u0074\u0065\u0072\u006D\u0073\u000A\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020", 111);  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:51
  return (internal::TypeAsHelper<Tuple>::run(((this->cutter).generic_call_attr("segment", {(text), (keep_and_trans_space), (config)})), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 52, in segment\n", "expect '(this->cutter).generic_call_attr(\"segment\", {(text), (keep_and_trans_space), (config)})' is 'Tuple' type"));  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:52
}

MATX_DLL Tuple Cutter__F_segment(const Cutter_SharedView& self, const RTView& text, bool keep_and_trans_space, const unicode_view& config) {  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:45
  return (self->segment(text, keep_and_trans_space, config));  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:45
}

int Cutter__F_segment__c_api(MATXScriptAny* args, int num_args, MATXScriptAny* out_ret_value, void* resource_handle = nullptr)
{
  TArgs args_t(args, num_args);

  if (num_args > 0 && args[num_args - 1].code == TypeIndex::kRuntimeKwargs) {
    string_view arg_names[4] {"self", "text", "keep_and_trans_space", "config"};
    static RTValue default_args[2]{RTValue((bool)0), RTValue(unicode_view())};
    KwargsUnpackHelper helper("segment", arg_names, 4, default_args, 2);
    RTView pos_args[4];
    helper.unpack(pos_args, args, num_args);  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:45
    auto ret = Cutter__F_segment(internal::TypeAsHelper<UserDataRef>::run((pos_args[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 45, in segment\n", "expect 'self' is 'UserDataRef' type"), pos_args[1], internal::TypeAsHelper<bool>::run((pos_args[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 45, in segment\n", "expect 'keep_and_trans_space' is 'bool' type"), internal::TypeAsHelper<unicode_view>::run((pos_args[3]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 45, in segment\n", "expect 'config' is 'py::str' type"));  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:45
    RTValue(std::move(ret)).MoveToCHost(out_ret_value);
  } else {
    switch(num_args) {
      case 2: {
        auto ret = Cutter__F_segment(internal::TypeAsHelper<UserDataRef>::run((args_t[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 45, in segment\n", "expect 'self' is 'UserDataRef' type"), args_t[1], (bool)0, unicode_view());  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:45
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      } break;
      case 3: {
        auto ret = Cutter__F_segment(internal::TypeAsHelper<UserDataRef>::run((args_t[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 45, in segment\n", "expect 'self' is 'UserDataRef' type"), args_t[1], internal::TypeAsHelper<bool>::run((args_t[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 45, in segment\n", "expect 'keep_and_trans_space' is 'bool' type"), unicode_view());  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:45
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      } break;
      case 4: {
        auto ret = Cutter__F_segment(internal::TypeAsHelper<UserDataRef>::run((args_t[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 45, in segment\n", "expect 'self' is 'UserDataRef' type"), args_t[1], internal::TypeAsHelper<bool>::run((args_t[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 45, in segment\n", "expect 'keep_and_trans_space' is 'bool' type"), internal::TypeAsHelper<unicode_view>::run((args_t[3]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 45, in segment\n", "expect 'config' is 'py::str' type"));  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:45
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      } break;
      default: {THROW_PY_TypeError("File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 45, in segment\n", "segment() takes from 2 to 4 positional arguments but ", num_args, " were given");} break;  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:45
    }
  }

  return 0;
}

Unicode SimpleCutter::__call__(const unicode_view& text) {  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:328
  List words = (internal::TypeAsHelper<List>::run(((this->cutter).generic_call({(text), (this->cut_level)})), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 329, in __call__\n", "expect '(this->cutter).generic_call({(text), (this->cut_level)})' is 'List' type"));  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:329
  return (kernel_unicode_join(unicode_view(U"\u0020", 1), words));  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:330
}

MATX_DLL Unicode SimpleCutter__F___call__(const SimpleCutter_SharedView& self, const unicode_view& text) {  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:328
  return (self->__call__(text));  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:328
}

int SimpleCutter__F___call____c_api(MATXScriptAny* args, int num_args, MATXScriptAny* out_ret_value, void* resource_handle = nullptr)
{
  TArgs args_t(args, num_args);

  if (num_args > 0 && args[num_args - 1].code == TypeIndex::kRuntimeKwargs) {
    string_view arg_names[2] {"self", "text"};
    KwargsUnpackHelper helper("__call__", arg_names, 2, nullptr, 0);
    RTView pos_args[2];
    helper.unpack(pos_args, args, num_args);  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:328
    auto ret = SimpleCutter__F___call__(internal::TypeAsHelper<UserDataRef>::run((pos_args[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 328, in __call__\n", "expect 'self' is 'UserDataRef' type"), internal::TypeAsHelper<unicode_view>::run((pos_args[1]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 328, in __call__\n", "expect 'text' is 'py::str' type"));  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:328
    RTValue(std::move(ret)).MoveToCHost(out_ret_value);
  } else {
    switch(num_args) {
      case 2: {
        auto ret = SimpleCutter__F___call__(internal::TypeAsHelper<UserDataRef>::run((args_t[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 328, in __call__\n", "expect 'self' is 'UserDataRef' type"), internal::TypeAsHelper<unicode_view>::run((args_t[1]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 328, in __call__\n", "expect 'text' is 'py::str' type"));  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:328
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      } break;
      default: {THROW_PY_TypeError("File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 328, in __call__\n", "__call__() takes 2 positional arguments but ", num_args, " were given");} break;  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:328
    }
  }

  return 0;
}

RTValue SimpleCutter::__init__(const unicode_view& cut_type, const unicode_view& location, const unicode_view& cut_level, void* handle_2_71828182846) {  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:324
  this->session_handle_2_71828182846_ = handle_2_71828182846;  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:324
  this->cutter = Cutter__F___init___wrapper(cut_type, location, this->session_handle_2_71828182846_);  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:325
  this->cut_level = GenericValueConverter<Unicode>{}(cut_level);  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:326
  return (None);
}

MATX_DLL RTValue SimpleCutter__F___init__(const SimpleCutter_SharedView& self, const unicode_view& cut_type, const unicode_view& location, const unicode_view& cut_level, void* handle_2_71828182846) {  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:324
  return (self->__init__(cut_type, location, cut_level, handle_2_71828182846));  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:324
}

int SimpleCutter__F___init____c_api(MATXScriptAny* args, int num_args, MATXScriptAny* out_ret_value, void* resource_handle = nullptr)
{
  TArgs args_t(args, num_args);

  if (num_args > 0 && args[num_args - 1].code == TypeIndex::kRuntimeKwargs) {
    string_view arg_names[4] {"self", "cut_type", "location", "cut_level"};
    KwargsUnpackHelper helper("__init__", arg_names, 4, nullptr, 0);
    RTView pos_args[4];
    helper.unpack(pos_args, args, num_args);  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:324
    auto ret = SimpleCutter__F___init__(internal::TypeAsHelper<UserDataRef>::run((pos_args[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 324, in __init__\n", "expect 'self' is 'UserDataRef' type"), internal::TypeAsHelper<unicode_view>::run((pos_args[1]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 324, in __init__\n", "expect 'cut_type' is 'py::str' type"), internal::TypeAsHelper<unicode_view>::run((pos_args[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 324, in __init__\n", "expect 'location' is 'py::str' type"), internal::TypeAsHelper<unicode_view>::run((pos_args[3]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 324, in __init__\n", "expect 'cut_level' is 'py::str' type"), resource_handle);  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:324
    RTValue(std::move(ret)).MoveToCHost(out_ret_value);
  } else {
    switch(num_args) {
      case 4: {
        auto ret = SimpleCutter__F___init__(internal::TypeAsHelper<UserDataRef>::run((args_t[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 324, in __init__\n", "expect 'self' is 'UserDataRef' type"), internal::TypeAsHelper<unicode_view>::run((args_t[1]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 324, in __init__\n", "expect 'cut_type' is 'py::str' type"), internal::TypeAsHelper<unicode_view>::run((args_t[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 324, in __init__\n", "expect 'location' is 'py::str' type"), internal::TypeAsHelper<unicode_view>::run((args_t[3]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 324, in __init__\n", "expect 'cut_level' is 'py::str' type"), resource_handle);  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:324
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      } break;
      default: {THROW_PY_TypeError("File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 324, in __init__\n", "__init__() takes 4 positional arguments but ", num_args, " were given");} break;  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:324
    }
  }

  return 0;
}

int SimpleCutter__F___init___wrapper__c_api(MATXScriptAny* args, int num_args, MATXScriptAny* out_ret_value, void* resource_handle = nullptr)
{
  TArgs args_t(args, num_args);

  if (num_args > 0 && args[num_args - 1].code == TypeIndex::kRuntimeKwargs) {
    string_view arg_names[3] {"cut_type", "location", "cut_level"};
    KwargsUnpackHelper helper("__init__", arg_names, 3, nullptr, 0);
    RTView pos_args[3];
    helper.unpack(pos_args, args, num_args);  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:324
    auto ret = SimpleCutter__F___init___wrapper(internal::TypeAsHelper<unicode_view>::run((pos_args[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 324, in __init__\n", "expect 'cut_type' is 'py::str' type"), internal::TypeAsHelper<unicode_view>::run((pos_args[1]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 324, in __init__\n", "expect 'location' is 'py::str' type"), internal::TypeAsHelper<unicode_view>::run((pos_args[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 324, in __init__\n", "expect 'cut_level' is 'py::str' type"), resource_handle);  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:324
    (ret.operator RTValue()).MoveToCHost(out_ret_value);
  } else {
    switch(num_args) {
      case 3: {
        auto ret = SimpleCutter__F___init___wrapper(internal::TypeAsHelper<unicode_view>::run((args_t[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 324, in __init__\n", "expect 'cut_type' is 'py::str' type"), internal::TypeAsHelper<unicode_view>::run((args_t[1]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 324, in __init__\n", "expect 'location' is 'py::str' type"), internal::TypeAsHelper<unicode_view>::run((args_t[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 324, in __init__\n", "expect 'cut_level' is 'py::str' type"), resource_handle);  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:324
        (ret.operator RTValue()).MoveToCHost(out_ret_value);
      } break;
      default: {THROW_PY_TypeError("File \"/home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py\", line 324, in __init__\n", "__init__() takes 3 positional arguments but ", num_args, " were given");} break;  // /home/tiger/.local/lib/python3.7/site-packages/ptx/matx/pipeline.py:324
    }
  }

  return 0;
}

List Cutter::cut(const RTView& text, const unicode_view& cut_level, bool keep_and_trans_space, const unicode_view& config) {  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:63
  (void)unicode_view(U"\u000A\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0063\u0075\u0074\u0020\u0074\u0065\u0078\u0074\u0020\u0074\u006F\u0020\u0074\u0065\u0072\u006D\u0073\u0020\u0075\u0073\u0065\u0020\u0073\u0070\u0065\u0063\u0069\u0066\u0069\u0065\u0064\u0020\u006C\u0065\u0076\u0065\u006C\u000A\u000A\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u003A\u0070\u0061\u0072\u0061\u006D\u0020\u0074\u0065\u0078\u0074\u003A\u0020\u0060\u0060\u0073\u0074\u0072\u0060\u0060\u000A\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u003A\u0070\u0061\u0072\u0061\u006D\u0020\u0063\u0075\u0074\u005F\u006C\u0065\u0076\u0065\u006C\u003A\u0020\u0060\u0060\u0073\u0074\u0072\u0060\u0060\u000A\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0022\u0046\u0049\u004E\u0045\u0022\u002C\u0020\u0022\u0043\u004F\u0041\u0052\u0053\u0045\u0022\u002C\u0020\u0022\u0044\u0045\u0046\u0041\u0055\u004C\u0054\u0022\u000A\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u003A\u0072\u0065\u0074\u0075\u0072\u006E\u003A\u0020\u0060\u0060\u006C\u0069\u0073\u0074\u005B\u0073\u0074\u0072\u005D\u0060\u0060\u000A\u0020\u0020\u0020\u0020\u0020\u0020\u0020\u0020", 194);  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:71
  return (internal::TypeAsHelper<List>::run(((this->cutter).generic_call_attr("cut", {(text), (cut_level), (keep_and_trans_space), (config)})), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 72, in cut\n", "expect '(this->cutter).generic_call_attr(\"cut\", {(text), (cut_level), (keep_and_trans_space), (config)})' is 'List' type"));  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:72
}

MATX_DLL List Cutter__F_cut(const Cutter_SharedView& self, const RTView& text, const unicode_view& cut_level, bool keep_and_trans_space, const unicode_view& config) {  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:63
  return (self->cut(text, cut_level, keep_and_trans_space, config));  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:63
}

int Cutter__F_cut__c_api(MATXScriptAny* args, int num_args, MATXScriptAny* out_ret_value, void* resource_handle = nullptr)
{
  TArgs args_t(args, num_args);

  if (num_args > 0 && args[num_args - 1].code == TypeIndex::kRuntimeKwargs) {
    string_view arg_names[5] {"self", "text", "cut_level", "keep_and_trans_space", "config"};
    static RTValue default_args[3]{RTValue(unicode_view(U"\u0044\u0045\u0046\u0041\u0055\u004C\u0054", 7)), RTValue((bool)0), RTValue(unicode_view())};
    KwargsUnpackHelper helper("cut", arg_names, 5, default_args, 3);
    RTView pos_args[5];
    helper.unpack(pos_args, args, num_args);  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:63
    auto ret = Cutter__F_cut(internal::TypeAsHelper<UserDataRef>::run((pos_args[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 63, in cut\n", "expect 'self' is 'UserDataRef' type"), pos_args[1], internal::TypeAsHelper<unicode_view>::run((pos_args[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 63, in cut\n", "expect 'cut_level' is 'py::str' type"), internal::TypeAsHelper<bool>::run((pos_args[3]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 63, in cut\n", "expect 'keep_and_trans_space' is 'bool' type"), internal::TypeAsHelper<unicode_view>::run((pos_args[4]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 63, in cut\n", "expect 'config' is 'py::str' type"));  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:63
    RTValue(std::move(ret)).MoveToCHost(out_ret_value);
  } else {
    switch(num_args) {
      case 2: {
        auto ret = Cutter__F_cut(internal::TypeAsHelper<UserDataRef>::run((args_t[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 63, in cut\n", "expect 'self' is 'UserDataRef' type"), args_t[1], unicode_view(U"\u0044\u0045\u0046\u0041\u0055\u004C\u0054", 7), (bool)0, unicode_view());  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:63
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      } break;
      case 3: {
        auto ret = Cutter__F_cut(internal::TypeAsHelper<UserDataRef>::run((args_t[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 63, in cut\n", "expect 'self' is 'UserDataRef' type"), args_t[1], internal::TypeAsHelper<unicode_view>::run((args_t[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 63, in cut\n", "expect 'cut_level' is 'py::str' type"), (bool)0, unicode_view());  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:63
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      } break;
      case 4: {
        auto ret = Cutter__F_cut(internal::TypeAsHelper<UserDataRef>::run((args_t[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 63, in cut\n", "expect 'self' is 'UserDataRef' type"), args_t[1], internal::TypeAsHelper<unicode_view>::run((args_t[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 63, in cut\n", "expect 'cut_level' is 'py::str' type"), internal::TypeAsHelper<bool>::run((args_t[3]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 63, in cut\n", "expect 'keep_and_trans_space' is 'bool' type"), unicode_view());  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:63
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      } break;
      case 5: {
        auto ret = Cutter__F_cut(internal::TypeAsHelper<UserDataRef>::run((args_t[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 63, in cut\n", "expect 'self' is 'UserDataRef' type"), args_t[1], internal::TypeAsHelper<unicode_view>::run((args_t[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 63, in cut\n", "expect 'cut_level' is 'py::str' type"), internal::TypeAsHelper<bool>::run((args_t[3]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 63, in cut\n", "expect 'keep_and_trans_space' is 'bool' type"), internal::TypeAsHelper<unicode_view>::run((args_t[4]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 63, in cut\n", "expect 'config' is 'py::str' type"));  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:63
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      } break;
      default: {THROW_PY_TypeError("File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 63, in cut\n", "cut() takes from 2 to 5 positional arguments but ", num_args, " were given");} break;  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:63
    }
  }

  return 0;
}

List Cutter::__call__(const RTView& text, const unicode_view& cut_level, bool keep_and_trans_space, const unicode_view& config) {  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:74
  return (internal::TypeAsHelper<List>::run(((this->cutter).generic_call_attr("cut", {(text), (cut_level), (keep_and_trans_space), (config)})), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 75, in __call__\n", "expect '(this->cutter).generic_call_attr(\"cut\", {(text), (cut_level), (keep_and_trans_space), (config)})' is 'List' type"));  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:75
}

MATX_DLL List Cutter__F___call__(const Cutter_SharedView& self, const RTView& text, const unicode_view& cut_level, bool keep_and_trans_space, const unicode_view& config) {  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:74
  return (self->__call__(text, cut_level, keep_and_trans_space, config));  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:74
}

int Cutter__F___call____c_api(MATXScriptAny* args, int num_args, MATXScriptAny* out_ret_value, void* resource_handle = nullptr)
{
  TArgs args_t(args, num_args);

  if (num_args > 0 && args[num_args - 1].code == TypeIndex::kRuntimeKwargs) {
    string_view arg_names[5] {"self", "text", "cut_level", "keep_and_trans_space", "config"};
    static RTValue default_args[3]{RTValue(unicode_view(U"\u0044\u0045\u0046\u0041\u0055\u004C\u0054", 7)), RTValue((bool)0), RTValue(unicode_view())};
    KwargsUnpackHelper helper("__call__", arg_names, 5, default_args, 3);
    RTView pos_args[5];
    helper.unpack(pos_args, args, num_args);  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:74
    auto ret = Cutter__F___call__(internal::TypeAsHelper<UserDataRef>::run((pos_args[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 74, in __call__\n", "expect 'self' is 'UserDataRef' type"), pos_args[1], internal::TypeAsHelper<unicode_view>::run((pos_args[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 74, in __call__\n", "expect 'cut_level' is 'py::str' type"), internal::TypeAsHelper<bool>::run((pos_args[3]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 74, in __call__\n", "expect 'keep_and_trans_space' is 'bool' type"), internal::TypeAsHelper<unicode_view>::run((pos_args[4]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 74, in __call__\n", "expect 'config' is 'py::str' type"));  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:74
    RTValue(std::move(ret)).MoveToCHost(out_ret_value);
  } else {
    switch(num_args) {
      case 2: {
        auto ret = Cutter__F___call__(internal::TypeAsHelper<UserDataRef>::run((args_t[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 74, in __call__\n", "expect 'self' is 'UserDataRef' type"), args_t[1], unicode_view(U"\u0044\u0045\u0046\u0041\u0055\u004C\u0054", 7), (bool)0, unicode_view());  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:74
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      } break;
      case 3: {
        auto ret = Cutter__F___call__(internal::TypeAsHelper<UserDataRef>::run((args_t[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 74, in __call__\n", "expect 'self' is 'UserDataRef' type"), args_t[1], internal::TypeAsHelper<unicode_view>::run((args_t[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 74, in __call__\n", "expect 'cut_level' is 'py::str' type"), (bool)0, unicode_view());  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:74
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      } break;
      case 4: {
        auto ret = Cutter__F___call__(internal::TypeAsHelper<UserDataRef>::run((args_t[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 74, in __call__\n", "expect 'self' is 'UserDataRef' type"), args_t[1], internal::TypeAsHelper<unicode_view>::run((args_t[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 74, in __call__\n", "expect 'cut_level' is 'py::str' type"), internal::TypeAsHelper<bool>::run((args_t[3]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 74, in __call__\n", "expect 'keep_and_trans_space' is 'bool' type"), unicode_view());  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:74
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      } break;
      case 5: {
        auto ret = Cutter__F___call__(internal::TypeAsHelper<UserDataRef>::run((args_t[0]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 74, in __call__\n", "expect 'self' is 'UserDataRef' type"), args_t[1], internal::TypeAsHelper<unicode_view>::run((args_t[2]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 74, in __call__\n", "expect 'cut_level' is 'py::str' type"), internal::TypeAsHelper<bool>::run((args_t[3]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 74, in __call__\n", "expect 'keep_and_trans_space' is 'bool' type"), internal::TypeAsHelper<unicode_view>::run((args_t[4]), __FILE__, __LINE__, "File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 74, in __call__\n", "expect 'config' is 'py::str' type"));  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:74
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      } break;
      default: {THROW_PY_TypeError("File \"/home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py\", line 74, in __call__\n", "__call__() takes from 2 to 5 positional arguments but ", num_args, " were given");} break;  // /home/tiger/.local/lib/python3.7/site-packages/text_cutter/cut/cutter.py:74
    }
  }

  return 0;
}


} // namespace

extern "C" {

MATX_DLL MATXScriptBackendPackedCFunc __matxscript_func_array__Cutter[] = {
    (MATXScriptBackendPackedCFunc)Cutter__F___init___wrapper__c_api,
    (MATXScriptBackendPackedCFunc)Cutter__F___init____c_api,
    (MATXScriptBackendPackedCFunc)Cutter__F_segment__c_api,
    (MATXScriptBackendPackedCFunc)Cutter__F_segment_str__c_api,
    (MATXScriptBackendPackedCFunc)Cutter__F_cut__c_api,
    (MATXScriptBackendPackedCFunc)Cutter__F___call____c_api,
};
MATX_DLL MATXScriptFuncRegistry __matxscript_func_registry__Cutter = {
    "6\000Cutter__F___init___wrapper\000Cutter__F___init__\000Cutter__F_segment\000Cutter__F_segment_str\000Cutter__F_cut\000Cutter__F___call__\000",    __matxscript_func_array__Cutter,
};

} // extern C

extern "C" {

MATX_DLL MATXScriptBackendPackedCFunc __matxscript_func_array__SimpleCutter[] = {
    (MATXScriptBackendPackedCFunc)SimpleCutter__F___init___wrapper__c_api,
    (MATXScriptBackendPackedCFunc)SimpleCutter__F___init____c_api,
    (MATXScriptBackendPackedCFunc)SimpleCutter__F___call____c_api,
};
MATX_DLL MATXScriptFuncRegistry __matxscript_func_registry__SimpleCutter = {
    "3\000SimpleCutter__F___init___wrapper\000SimpleCutter__F___init__\000SimpleCutter__F___call__\000",    __matxscript_func_array__SimpleCutter,
};

} // extern C

extern "C" {

MATX_DLL MATXScriptBackendPackedCFunc __matxscript_func_array__[] = {
    (MATXScriptBackendPackedCFunc)SimpleCutter__F___init___wrapper__c_api,
    (MATXScriptBackendPackedCFunc)Cutter__F___init___wrapper__c_api,
    (MATXScriptBackendPackedCFunc)SimpleCutter__F___call____c_api,
};
MATX_DLL MATXScriptFuncRegistry __matxscript_func_registry__ = {
    "3\000SimpleCutter__F___init___wrapper\000Cutter__F___init___wrapper\000SimpleCutter__F___call__\000",    __matxscript_func_array__,
};

} // extern C

extern "C" {

MATX_DLL const char* __matxscript_closures_names__ = "4\000SimpleCutter__F___init___wrapper\000Cutter__F___init___wrapper\000Cutter__F___init__\000SimpleCutter__F___init__\000";

} // extern C

