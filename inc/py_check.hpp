#ifndef KLEIN_CPP_INC_CHECK_HPP
#define KLEIN_CPP_INC_CHECK_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename T>
inline bool internal_py_check_dtype(const py::array & v1, const T & dtype){
	return v1.dtype().equal(dtype);
}

template<typename T>
inline bool py_check_dtype(const py::array & v1){
	const auto & dtype = pybind11::dtype::of<T>();
	return internal_py_check_dtype(v1, dtype);
}

template<typename T>
inline bool py_check_dtype(const py::array & v1, const py::array & v2){
	const auto & dtype = pybind11::dtype::of<T>();
	return internal_py_check_dtype(v1, dtype) && internal_py_check_dtype(v2, dtype);
}

template<typename T>
inline bool py_check_dtype(const py::array & v1, const py::array & v2, const py::array & v3){
	const auto & dtype = pybind11::dtype::of<T>();
	return internal_py_check_dtype(v1, dtype) && internal_py_check_dtype(v2, dtype) && internal_py_check_dtype(v3, dtype);	
}

template<typename T>
inline bool py_check_dtype(const py::array & v1, const py::array & v2, const py::array & v3, const py::array & v4){
	const auto & dtype = pybind11::dtype::of<T>();
	return internal_py_check_dtype(v1, dtype) && internal_py_check_dtype(v2, dtype) && internal_py_check_dtype(v3, dtype) && internal_py_check_dtype(v4, dtype);		
}


template<typename T>
inline bool py_check_dtype(const py::array & v1, const py::array & v2, const py::array & v3, const py::array & v4, const py::array & v5){
	const auto & dtype = pybind11::dtype::of<T>();
	return internal_py_check_dtype(v1, dtype) && internal_py_check_dtype(v2, dtype) && internal_py_check_dtype(v3, dtype) && internal_py_check_dtype(v4, dtype)
			&& internal_py_check_dtype(v5, dtype)
			;		
}


template<typename T>
inline bool py_check_dtype(const py::array & v1, const py::array & v2, const py::array & v3, const py::array & v4, const py::array & v5, const py::array & v6){
	const auto & dtype = pybind11::dtype::of<T>();
	return internal_py_check_dtype(v1, dtype) && internal_py_check_dtype(v2, dtype) && internal_py_check_dtype(v3, dtype) && internal_py_check_dtype(v4, dtype)
			&& internal_py_check_dtype(v5, dtype) && internal_py_check_dtype(v6, dtype)
			;		
}


template<typename T>
inline bool py_check_dtype(const py::array & v1, const py::array & v2, const py::array & v3, const py::array & v4, const py::array & v5, const py::array & v6, const py::array & v7){
	const auto & dtype = pybind11::dtype::of<T>();
	return internal_py_check_dtype(v1, dtype) && internal_py_check_dtype(v2, dtype) && internal_py_check_dtype(v3, dtype) && internal_py_check_dtype(v4, dtype)
			&& internal_py_check_dtype(v5, dtype) && internal_py_check_dtype(v6, dtype) && internal_py_check_dtype(v7, dtype)
			;		
}

template<typename T>
inline bool py_check_dtype(const py::array & v1, const py::array & v2, const py::array & v3, const py::array & v4, const py::array & v5, const py::array & v6, const py::array & v7, const py::array & v8){
	const auto & dtype = pybind11::dtype::of<T>();
	return internal_py_check_dtype(v1, dtype) && internal_py_check_dtype(v2, dtype) && internal_py_check_dtype(v3, dtype) && internal_py_check_dtype(v4, dtype)
			&& internal_py_check_dtype(v5, dtype) && internal_py_check_dtype(v6, dtype) && internal_py_check_dtype(v7, dtype) && internal_py_check_dtype(v8, dtype)
			;		
}


template<int N>
inline bool py_check_ndim(const py::array & v1){
	return N == v1.ndim();
}

template<int N>
inline bool py_check_ndim(const py::array & v1, const py::array & v2){
	return N == v1.ndim() && N == v2.ndim();
}

template<int N>
inline bool py_check_ndim(const py::array & v1, const py::array & v2, const py::array & v3){
	return N == v1.ndim() && N == v2.ndim() && N == v3.ndim();	
}

template<int N>
inline bool py_check_ndim(const py::array & v1, const py::array & v2, const py::array & v3, const py::array & v4){
	return N == v1.ndim() && N == v2.ndim() && N == v3.ndim() && N == v4.ndim();		
}

template<int N>
inline bool py_check_ndim(const py::array & v1, const py::array & v2, const py::array & v3, const py::array & v4, const py::array & v5){
	return N == v1.ndim() && N == v2.ndim() && N == v3.ndim() && N == v4.ndim()
			&& N == v5.ndim()
		;		
}

template<int N>
inline bool py_check_ndim(const py::array & v1, const py::array & v2, const py::array & v3, const py::array & v4, const py::array & v5, const py::array & v6){
	return N == v1.ndim() && N == v2.ndim() && N == v3.ndim() && N == v4.ndim()
			&& N == v5.ndim() && N == v6.ndim()
		;		
}

template<int N>
inline bool py_check_ndim(const py::array & v1, const py::array & v2, const py::array & v3, const py::array & v4, const py::array & v5, const py::array & v6, const py::array & v7){
	return N == v1.ndim() && N == v2.ndim() && N == v3.ndim() && N == v4.ndim()
			&& N == v5.ndim() && N == v6.ndim() && N == v7.ndim()
		;		
}

template<int N>
inline bool py_check_ndim(const py::array & v1, const py::array & v2, const py::array & v3, const py::array & v4, const py::array & v5, const py::array & v6, const py::array & v7, const py::array & v8){
	return N == v1.ndim() && N == v2.ndim() && N == v3.ndim() && N == v4.ndim()
			&& N == v5.ndim() && N == v6.ndim() && N == v7.ndim() && N == v8.ndim()
		;		
}


#endif
