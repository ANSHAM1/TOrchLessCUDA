#include "Core/Cnn.cuh"

int main() {
	ARCH::Sequential<__half> Model;
	Model.Input({ 60000, 1, 28, 28 });
	Model.Conv2D(32, 1, 3, 3, 1, 1);
	Model.Activate("relu");
	Model.Conv2D(64, 32, 3, 3, 1, 1);
	Model.Activate("relu");
	Model.Conv2D(128, 64, 3, 3, 1, 1);
	Model.Activate("relu");
	Model.Conv2D(256, 128, 3, 3, 1, 1);
	Model.Activate("relu");
	Model.MaxPooling(2, 2, 1, 0);
	Model.Dense(1024);
	Model.Activate("relu");
	Model.Dense(10);
	Model.Output("softmax");

	Model.Compile();

		
	return 0;
}