#include <iostream>
#include <cmath>
#include <vector>
#include "LinearRegression.cpp"

using namespace std;

int main() {
	vector<float> X{ 0,1,2,3,4,5,6,7,8,9 };
	vector <float> y{ 1,5,6,8,2,9,11,14,14,12 };
	float B_1;
	float B_0;
	vector<vector<float>> data(2);
	LinearRegression<float, float> lr;

	data[0] = X;
	data[1] = y;
	lr.fit(data);
	cout << "Estimated equation: y = " << lr.B_1 << "x + (" << lr.B_0 << ")\n" << endl;

	for (int i = 0; i < X.size(); i++) {
		float y_pred = lr.predict(X[i]);
		cout << X[i] << " " << y_pred << endl;
	}

	return 0;
}
