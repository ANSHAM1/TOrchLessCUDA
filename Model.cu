#include "Core/Cnn.cuh"

int main() {
	//IF_CUDA_AVAILABLE({
		ARCH::Sequential<__half> Model;
		Model.Input({ 60000, 1, 28, 28 });
	//})
	return 0;
}