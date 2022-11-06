#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>

using namespace std;

template <class A, class B>
class LinearRegression {
public:
	A B_0;
	A B_1;
	void fit(vector<vector<B>> dataset) {
		vector<B> X = dataset[0]; //x
		vector<B> y = dataset[1]; //y

		A X_sum;
		A y_sum;
		int n = X.size();

		for (int i = 0; i < X.size(); i++) {
			X_sum += X[i];
			y_sum += y[i];
		}

		A X_mean = X_sum / n;
		A y_mean = y_sum / n;

		A SSxx = 0;
		A SSxy = 0;
		//declaring SSxy
		{
			A t_sum = 0;
			for (int i = 0; i < n; i++) {
				t_sum += X[i] * y[i];
			}
			SSxy = t_sum - n * X_mean * y_mean;
		}

		//declaring SSxx
		{
			A t_sum = 0;
			for (int i = 0; i < n; i++) {
				t_sum += X[i] * X[i];
			}
			SSxx = t_sum - n * X_mean * X_mean;
		}

		B_1 = SSxy / SSxx;
		B_0 = y_mean - B_1 * X_mean;
	};

	B predict(const B& dataset) {
		return B_0 + (B_1 * dataset);
	};

};