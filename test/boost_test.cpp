#define BOOST_TEST_MODULE MonProjetTests
#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_CASE(Test1) {
    BOOST_TEST(1 + 1 == 2);
}

BOOST_AUTO_TEST_CASE(TestBoost) {
    BOOST_TEST(true);
}