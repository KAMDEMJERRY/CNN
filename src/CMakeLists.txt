add_library(cnn_lib
    convolution.cpp
    model.cpp
    dense.cpp
    imgdataset.cpp
    utils.cpp

    # tf_core.cpp
    # tf_layer.cpp
    # tf_model.cpp
)

target_link_libraries(cnn_lib
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${OPENMP_LIBRARIES}
    ${Boost_LIBRARIES}
    ${Python3_LIBRARIES}
)



add_executable(CNN
    main.cpp
)

target_link_libraries(CNN
    cnn_lib
)


add_executable(EVAL
    eval.cpp
)

target_link_libraries(EVAL
    cnn_lib
)

# add_executable(TENSOR
#     tf_main.cpp
# )


# target_link_libraries(TENSOR
#     cnn_lib
# )