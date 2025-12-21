#ifndef EIGEN_SERIALIZATION_HPP
#define EIGEN_SERIALIZATION_HPP

#include <Eigen/Dense>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/binary_object.hpp> // For binary_object
#include <boost/archive/text_oarchive.hpp>


BOOST_SERIALIZATION_SPLIT_FREE(Eigen::MatrixXd)
BOOST_SERIALIZATION_SPLIT_FREE(Eigen::VectorXd)
BOOST_SERIALIZATION_SPLIT_FREE(Eigen::Matrix3d)


namespace boost {
namespace serialization {

template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void save(Archive & ar, const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& M, const unsigned int /*version*/) {
    typename Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>::Index rows = M.rows();
    typename Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>::Index cols = M.cols();
    ar << rows;
    ar << cols;
    ar << make_binary_object(M.data(), M.size() * sizeof(_Scalar));
}

template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void load(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& M, const unsigned int /*version*/) {
    typename Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>::Index rows, cols;
    ar >> rows;
    ar >> cols;
    M.resize(rows, cols);
    ar >> make_binary_object(M.data(), M.size() * sizeof(_Scalar));
}

template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline void serialize(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& M, const unsigned int file_version) {
    split_free(ar, M, file_version);
}

} // namespace serialization
} // namespace boost


#endif // EIGEN_SERIALIZATION_HPP







