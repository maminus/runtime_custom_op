#include <cassert>
#include <vector>
#include <string>
#include <string_view>
#include <tuple>	// forward_as_tuple
#include <memory>	// unique_ptr
#include <mutex>	// lock_guard, mutex

#define ORT_API_MANUAL_INIT
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include "cuda_kernel.cuh"

// internal linkage
namespace {

// kernel of FMA operation
template <typename T>
struct FmaKernel {
	FmaKernel(const OrtApi &api):ort_(api) {}

	void Compute(OrtKernelContext* context) {
		assert(ort_.KernelContext_GetInputCount(context) >= 2);

		const auto input_a = ort_.KernelContext_GetInput(context, 0);
		const auto input_b = ort_.KernelContext_GetInput(context, 1);

		auto info_a = ort_.GetTensorTypeAndShape(input_a);
		auto shape_a = ort_.GetTensorShape(info_a);
		auto element_count = ort_.GetTensorShapeElementCount(info_a);
		ort_.ReleaseTensorTypeAndShapeInfo(info_a);
		auto info_b = ort_.GetTensorTypeAndShape(input_b);
		auto shape_b = ort_.GetTensorShape(info_b);
		ort_.ReleaseTensorTypeAndShapeInfo(info_b);
		assert(shape_a == shape_b);

		auto output_0 = ort_.KernelContext_GetOutput(context, 0, shape_a.data(), shape_a.size());

		// get device memories
		auto ptr_a = ort_.GetTensorData<T>(input_a);
		auto ptr_b = ort_.GetTensorData<T>(input_b);
		auto ptr_0 = ort_.GetTensorMutableData<T>(output_0);

		// get CUDA stream ID
		cudaStream_t stream = reinterpret_cast<cudaStream_t>(ort_.KernelContext_GetGPUComputeStream(context));

		if (ort_.KernelContext_GetInputCount(context) > 2) {
			const auto input_c = ort_.KernelContext_GetInput(context, 2);
			auto info_c = ort_.GetTensorTypeAndShape(input_c);
			auto shape_c = ort_.GetTensorShape(info_c);
			ort_.ReleaseTensorTypeAndShapeInfo(info_c);
			assert(shape_a == shape_b && shape_a == shape_c);
			auto ptr_c = ort_.GetTensorData<T>(input_c);

			// call CUDA host function
			custom_kernel::fma_core<T>(ptr_0, ptr_a, ptr_b, ptr_c, element_count, stream);
		} else {
			custom_kernel::fma_core<T>(ptr_0, ptr_a, ptr_b, element_count, stream);
		}
	}

private:
	Ort::CustomOpApi ort_;
};

// FMA operator
template <typename T>
struct CustomOpFma : Ort::CustomOpBase<CustomOpFma<T>, FmaKernel<T>> {
	void* CreateKernel(const OrtApi &api, const OrtKernelInfo* info) const {
		return new FmaKernel<T>(api);
	}
	const char* GetName() const {
		// same as op_type
		return "Fma";
	}
	const char* GetExecutionProviderType() const {
		return "CUDAExecutionProvider";
	}
	ONNXTensorElementDataType GetInputType(size_t index) const {
		return Ort::TypeToTensorType<T>::type;
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
		return Ort::TypeToTensorType<T>::type;
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


constexpr std::string_view domain_base = "my_ops";
CustomOpFma<float> op_fma_float;	// float type FMA operator
CustomOpFma<double> op_fma_double;	// double type FMA operator

// domain name and corresponding operator
std::vector<std::tuple<std::string, OrtCustomOp*>> op_list = {
	std::forward_as_tuple(std::string(domain_base) + "." + "float", &op_fma_float),
	std::forward_as_tuple(std::string(domain_base) + "." + "double", &op_fma_double),
};

}	// namespace


// external linkage

extern "C" {

// call from SessionOptions.register_custom_ops_library()
OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
	const OrtApi* ort_api = api_base->GetApi(ORT_API_VERSION);
	OrtStatus* status = nullptr;

	for (const auto& [domain_name, op] : op_list) {
		OrtCustomOpDomain* domain = nullptr;
		if (status = ort_api->CreateCustomOpDomain(domain_name.c_str(), &domain)) {
			return status;
		}

		register_domain(domain, ort_api);

		if (status = ort_api->CustomOpDomain_Add(domain, op)) {
			return status;
		}

		if (status = ort_api->AddCustomOpDomain(options, domain)) {
			return status;
		}
	}

	return status;
}

}	// extern "C"
