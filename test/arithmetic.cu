// arithmetic.cu
// XXX here I use integer values for operations, but it is not sefety, because
// cuda has problem with signed integer overflow

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

//


#include "cuda/arithmetic.hpp"
#include <random>
#include <vector>


template <typename IOType, typename Op>
std::vector<IOType> binary_op(const std::vector<IOType> &lhs,
                              const std::vector<IOType> &rhs,
                              Op                         op) {
  assert(lhs.size() == rhs.size());

  std::vector<IOType> output(lhs.size());
  for (int i = 0; i < lhs.size(); ++i) {
    output[i] = op(lhs[i], rhs[i]);
  }

  return output;
}


SCENARIO("arithmetic", "[arithmetic]") {
  GIVEN("empty matricies") {
    cuda::gpu_mat<int> first{0, 0};
    cuda::gpu_mat<int> second{0, 0};

    WHEN("process") {
      cuda::subtract(first, second);
      cuda::add(first, second);
      cuda::divide(first, second);
      cuda::multiply(first, second);

      THEN("don't have zero devide error") {
      }
    }
  }

  GIVEN("random matricies with same size") {
    size_t height = GENERATE(3, 100, 512, 1024, 1153, 2048, 2049);
    size_t width  = GENERATE(3, 100, 512, 1024, 1352, 2048, 2051);
    size_t total  = width * height;


    std::random_device                 rd;
    std::mt19937                       gen{rd()};
    std::uniform_int_distribution<int> dist(-1024, +1024);

    std::vector<int> first_data(total);
    std::vector<int> second_data(total);

    for (int i = 0; i < total; ++i) {
      first_data[i]  = dist(gen);
      second_data[i] = dist(gen);
    }


    cuda::gpu_mat<int> first{height, width, first_data.data()};
    cuda::gpu_mat<int> second{height, width, second_data.data()};


    WHEN("subtract") {
      cuda::gpu_mat<int> out = cuda::subtract<int>(first, second);

      THEN("check result") {
        std::vector<int> out_data(total);
        out.download(out_data.data());

        CHECK(out_data ==
              binary_op(first_data, second_data, std::minus<int>{}));
      }
    }

    WHEN("add") {
      cuda::gpu_mat<int> out = cuda::add<int>(first, second);

      THEN("check result") {
        std::vector<int> out_data(total);
        out.download(out_data.data());

        CHECK(out_data == binary_op(first_data, second_data, std::plus<int>{}));
      }
    }

    WHEN("divide") {
      cuda::gpu_mat<int> out = cuda::divide<int>(first, second);

      THEN("check result") {
        std::vector<int> out_data(total);
        out.download(out_data.data());

        // XXX we can't divide on zero, so check it safety
        for (int i = 0; i < total; ++i) {
          int divider = second_data[i];
          if (divider != 0) {
            REQUIRE(out_data[i] == (first_data[i] / divider));
          } else {
            REQUIRE(out_data[i] == 0);
          }
        }
      }
    }

    WHEN("multiply") {
      cuda::gpu_mat<int> out = cuda::multiply<int>(first, second);

      THEN("check result") {
        std::vector<int> out_data(total);
        out.download(out_data.data());

        CHECK(out_data ==
              binary_op(first_data, second_data, std::multiplies<int>{}));
      }
    }
  }
}
