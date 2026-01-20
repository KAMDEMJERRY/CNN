add_executable(CNNtest
    convolution_test.cpp
)

add_executable(DataSet
    dataset_test.cpp
)

add_executable(BoostTest
    dump_test.cpp
    dump_test1.cpp

)

target_link_libraries(CNNtest
    cnn_lib
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
    Eigen3::Eigen
    GTest::gtest
    GTest::gtest_main
    pthread

)

target_link_libraries(DataSet
    cnn_lib
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
    GTest::gtest
    GTest::gtest_main
    pthread
)

target_link_libraries(BoostTest
    cnn_lib
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
    GTest::gtest
    GTest::gtest_main
    pthread
)
enable_testing()
add_test(NAME MyTests COMMAND CNNtest)
add_test(NAME DataSetTests COMMAND DataSet)
add_test(NAME BoostTest COMMAND BoostTest)