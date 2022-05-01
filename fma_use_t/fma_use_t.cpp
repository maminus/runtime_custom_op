#include <cassert>
#include <vector>
#include <string_view>
#include <memory>	// unique_ptr
#include <mutex>	// lock_guard, mutex

#define ORT_API_MANUAL_INIT
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

// internal linkage
namespace {

// FMA calculation
template <typename T>
void fma_core(Ort::CustomOpApi& ort, const OrtValue* input_a, const OrtValue* input_b, const OrtValue* input_c, OrtValue* output_0, size_t element_count) {
	auto ptr_a = ort.GetTensorData<T>(input_a);
	auto ptr_b = ort.GetTensorData<float>(input_b);
	auto ptr_c = (input_c)? ort.GetTensorData<float>(input_c) : static_cast<float*>(nullptr);
	auto ptr_0 = ort.GetTensorMutableData<T>(output_0);

	if (ptr_c) {
		for (decltype(element_count) i=0; i<element_count; i++) {
			*ptr_0++ = (*ptr_a++) * (*ptr_b++) + (*ptr_c++);
		}
	} else {
		for (decltype(element_count) i=0; i<element_count; i++) {
			*ptr_0++ = (*ptr_a++) * (*ptr_b++);
		}
	}
}

// kernel of FMA operation
struct FmaKernel {
	FmaKernel(OrtApi api):api_(api), ort_(api_) {}

	void Compute(OrtKernelContext* context) {
		assert(ort_.KernelContext_GetInputCount(context) >= 2);

		const auto input_a = ort_.KernelContext_GetInput(context, 0);
		const auto input_b = ort_.KernelContext_GetInput(context, 1);

		auto info_a = ort_.GetTensorTypeAndShape(input_a);
		auto etype_a = ort_.GetTensorElementType(info_a);
		auto shape_a = ort_.GetTensorShape(info_a);
		auto element_count = ort_.GetTensorShapeElementCount(info_a);
		ort_.ReleaseTensorTypeAndShapeInfo(info_a);
		auto info_b = ort_.GetTensorTypeAndShape(input_b);
		auto shape_b = ort_.GetTensorShape(info_b);
		ort_.ReleaseTensorTypeAndShapeInfo(info_b);
		assert(shape_a == shape_b);

		const OrtValue* input_c = nullptr;
		if (ort_.KernelContext_GetInputCount(context) == 3) {
			input_c = ort_.KernelContext_GetInput(context, 2);
			auto info_c = ort_.GetTensorTypeAndShape(input_c);
			auto shape_c = ort_.GetTensorShape(info_c);
			ort_.ReleaseTensorTypeAndShapeInfo(info_c);
			assert(shape_a == shape_c);
		}

		auto output_0 = ort_.KernelContext_GetOutput(context, 0, shape_a.data(), shape_a.size());

		if (etype_a == Ort::TypeToTensorType<float>::type) {
			fma_core<float>(ort_, input_a, input_b, input_c, output_0, element_count);
		}
		else if (etype_a == Ort::TypeToTensorType<double>::type) {
			fma_core<double>(ort_, input_a, input_b, input_c, output_0, element_count);
		}
		else {
			assert(false && "not supported type");
		}
	}

private:
	OrtApi api_;
	Ort::CustomOpApi ort_;
};

// FMA operator
struct CustomOpFma : Ort::CustomOpBase<CustomOpFma, FmaKernel> {
	void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
		return new FmaKernel(api);
	}
	const char* GetName() const {
		return "Fma";
	}
	ONNXTensorElementDataType GetInputType(size_t index) const {
		if (index > 0) {
			// input[1:] can't has type "T". so, fixed type
			return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
		}

		// input[0] has type "T" (any type)
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	}
	size_t GetInputTypeCount() const {
		return 3;
	}
	OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
		if (index > 1) {
			// input[2] is optional argument
			return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
		}
		return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
	}
	ONNXTensorElementDataType GetOutputType(size_t index) const {
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	}
	size_t GetOutputTypeCount() const {
		return 1;
	}
};


struct DomainDeleter {
	DomainDeleter(const OrtApi* api):api_(api) {}
	void operator() (OrtCustomOpDomain* domain) const {
		api_->ReleaseCustomOpDomain(domain);
	}

private:
	const OrtApi* api_;
};

using domain_ptr_type = std::unique_ptr<OrtCustomOpDomain, DomainDeleter>;
std::vector<domain_ptr_type> domain_registry;
void register_domain(OrtCustomOpDomain* domain, const OrtApi* api) {
	static std::mutex m;

	std::lock_guard<std::mutex> lock(m);

	domain_registry.emplace_back(domain_ptr_type(domain, DomainDeleter(api)));
}


constexpr std::string_view domain_name = "my_ops";
CustomOpFma op_fma;

}	// namespace


// external linkage

extern "C" {

// call from SessionOptions.register_custom_ops_library()
OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
	const OrtApi* ort_api = api_base->GetApi(ORT_API_VERSION);
	OrtStatus* status;

	OrtCustomOpDomain* domain = nullptr;
	if (status = ort_api->CreateCustomOpDomain(domain_name.data(), &domain)) {
		return status;
	}

	register_domain(domain, ort_api);

	if (status = ort_api->CustomOpDomain_Add(domain, &op_fma)) {
		return status;
	}

	status = ort_api->AddCustomOpDomain(options, domain);
	return status;
}

}	// extern "C"
