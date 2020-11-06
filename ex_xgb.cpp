#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <string>
#include <xgboost/c_api.h>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "inc/cnarray.hpp"

namespace py = pybind11;


#define safe_xgboost(call) {                                            \
int err = (call);                                                       \
if (err != 0) {                                                         \
  fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
  exit(1);                                                              \
}                                                                       \
}


class XgbShannonPredictor {
public:
	XgbShannonPredictor(const std::string & fname)
		:XgbShannonPredictor(fname.c_str()) {}

	XgbShannonPredictor(const char * fname) {

		DMatrixHandle dmats[2];
		safe_xgboost(XGBoosterCreate(dmats, 0, &booster_));
		safe_xgboost(XGBoosterSetParam(booster_, "nthread", "4"));

		safe_xgboost(XGBoosterLoadModel(booster_, fname));
	}
	~XgbShannonPredictor() {
		safe_xgboost(XGBoosterFree(booster_));
	}

	float predict(const float * values, const bst_ulong rows, const bst_ulong cols) {
		DMatrixHandle dmat;
		safe_xgboost(XGDMatrixCreateFromMat(values, rows, cols, 0.0, &dmat));

		bst_ulong out_len = 0;
		const float* out_result = NULL;

		safe_xgboost(XGBoosterPredict(booster_, dmat, 0, 0, 0, &out_len, &out_result));
		if(out_len != 1){
			throw std::runtime_error("runtime_error error in XGBoosterPredict.");
		}
		safe_xgboost(XGDMatrixFree(dmat));

		return out_result[0];
	}

	py::array predict_py(py::array & x) {
		const auto & dtype = pybind11::dtype::of<float>();
		if (!x.dtype().equal(dtype)) {
			throw std::runtime_error("x must be with dtype np.float32");
		}
		if (2 != x.ndim()) {
			throw std::runtime_error("x must be a matrix with (samples, features)");
		}
		const int rows = x.shape(0), cols = x.shape(1);
		cndarray<float, 2> xx(x);

		cndarray<float, 1> y;
		y.init(rows);

/*
		DMatrixHandle dmat;
		safe_xgboost(XGDMatrixCreateFromMat(xx.data(), rows, cols, 0.0, &dmat));
		bst_ulong out_len = 0;
		const float* out_result = NULL;

		safe_xgboost(XGBoosterPredict(booster_, dmat, 0, 0, 0, &out_len, &out_result));
		if(out_len != rows){
			throw std::runtime_error("runtime_error error in XGBoosterPredict.");
		}
		for (int i = 0; i < rows; ++i) y.ix(i) = out_result[i];
		safe_xgboost(XGDMatrixFree(dmat));
	*/

		for (int i = 0; i < rows; ++i) {
			y.ix(i) = predict(xx.view(i, 0).data(), 1, cols);
		}
		return y.to_ndarray();
	}
protected:
	BoosterHandle booster_;
};



PYBIND11_MODULE(XgbShannon, m)
{
	py::class_<XgbShannonPredictor>(m, "XgbShannonPredictor")
		.def(py::init<const char *>())
		.def("predict", &XgbShannonPredictor::predict_py)
		;

}
