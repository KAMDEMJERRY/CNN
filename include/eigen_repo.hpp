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
    ar << make_array(M.data(), M.size()); // Or binary_object(M.data(), M.size() * sizeof(_Scalar))
}

template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void load(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& M, const unsigned int /*version*/) {
    typename Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>::Index rows, cols;
    ar >> rows;
    ar >> cols;
    M.resize(rows, cols);
    ar >> make_array(M.data(), M.size()); // Or binary_object(M.data(), M.size() * sizeof(_Scalar))
}

template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline void serialize(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& M, const unsigned int file_version) {
    split_free(ar, M, file_version);
}

} // namespace serialization
} // namespace boost

// namespace boost
// {
//     namespace serialization
//     {

//         // MatrixXd
//         template <class Archive>
//         void save(Archive &ar, const Eigen::MatrixXd &m, const unsigned int version)
//         {
//             size_t rows = m.rows();
//             size_t cols = m.cols();
//             ar & rows & cols;

//             for (size_t i = 0; i < rows; ++i)
//             {
//                 for (size_t j = 0; j < cols; ++j)
//                 {
//                     ar &m(i, j);
//                 }
//             }
//         }

//         template <class Archive>
//         void load(Archive &ar, Eigen::MatrixXd &m, const unsigned int version)
//         {
//             size_t rows, cols;
//             ar & rows & cols;

//             m.resize(rows, cols);

//             for (size_t i = 0; i < rows; ++i)
//             {
//                 for (size_t j = 0; j < cols; ++j)
//                 {
//                     ar &m(i, j);
//                 }
//             }
//         }

//         // VectorXd
//         template <class Archive>
//         void save(Archive &ar, const Eigen::VectorXd &v, const unsigned int version)
//         {
//             size_t size = v.size();
//             ar & size;

//             for (size_t i = 0; i < size; ++i)
//             {
//                 ar &v(i);
//             }
//         }

//         template <class Archive>
//         void load(Archive &ar, Eigen::VectorXd &v, const unsigned int version)
//         {
//             size_t size;
//             ar & size;

//             v.resize(size);

//             for (size_t i = 0; i < size; ++i)
//             {
//                 ar &v(i);
//             }
//         }

//         // RowVectorXd
//         template <class Archive>
//         void save(Archive &ar, const Eigen::RowVectorXd &v, const unsigned int version)
//         {
//             size_t size = v.size();
//             ar & size;

//             for (size_t i = 0; i < size; ++i)
//             {
//                 ar &v(i);
//             }
//         }

//         template <class Archive>
//         void load(Archive &ar, Eigen::RowVectorXd &v, const unsigned int version)
//         {
//             size_t size;
//             ar & size;

//             v.resize(size);

//             for (size_t i = 0; i < size; ++i)
//             {
//                 ar &v(i);
//             }
//         }

//     } // namespace serialization
// } // namespace boost

#endif // EIGEN_SERIALIZATION_HPP







