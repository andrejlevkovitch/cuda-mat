// warp.cu

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

//


#include "cuda/warp.hpp"
#include <iostream>
#include <random>


SCENARIO("warp", "[warp]") {
  GIVEN("empty matrix") {
    cuda::gpu_mat<int> input{0, 0, 0};
    cuda::gpu_mat<int> output{0, 0};

    cuda::affine_matrix      mat_a{{1, 0, 0, 0, 1, 0}};
    cuda::perspective_matrix mat_p{{1, 0, 0, 0, 1, 0, 0, 0, 1}};

    WHEN("warp") {
      cuda::warpAffine(input, output, mat_a);
      cuda::warpPerspective(input, output, mat_p);

      THEN("haven't error about zero division") {
      }
    }
  }

  GIVEN("different matrix input/output sizes and some random data") {
    int height        = GENERATE(3, 100, 512, 1024, 1153, 2048, 2049);
    int width         = GENERATE(3, 100, 512, 1024, 1352, 2048, 2051);
    int height_output = GENERATE(3, 100, 512, 1024, 1153, 2048, 2049);
    int width_output  = GENERATE(3, 100, 512, 1024, 1352, 2048, 2051);
    int total         = width * height;

    char border_value = 5;


    std::random_device                  rd;
    std::mt19937                        gen{rd()};
    std::uniform_int_distribution<char> dist{-128, 127};

    std::vector<char> input_data(total);
    for (char &val : input_data) {
      val = dist(gen);
    }


    AND_GIVEN("sync pipeline") {
      cuda::gpu_mat<char> input(height, width, input_data.data());

      cuda::gpu_mat<char> output_a(height_output, width_output);
      cuda::gpu_mat<char> output_p(height_output, width_output);


      AND_GIVEN("identity matrix") {
        cuda::affine_matrix      mat_a{{1, 0, 0, 0, 1, 0}};
        cuda::perspective_matrix mat_p{{1, 0, 0, 0, 1, 0, 0, 0, 1}};


        WHEN("warp") {
          cuda::warpAffine(input, output_a, mat_a, 0, 0, border_value);
          cuda::warpPerspective(input, output_p, mat_p, 0, 0, border_value);

          THEN("output must be same as input") {
            std::vector<char> output_data_a(output_a.total());
            output_a.download(output_data_a.data());

            std::vector<char> output_data_p(output_p.total());
            output_p.download(output_data_p.data());


            REQUIRE(output_data_p == output_data_a);

            for (size_t row = 0; row < height_output; ++row) {
              for (size_t col = 0; col < width_output; ++col) {
                size_t offset_out = row * width_output + col;
                size_t offset_in  = row * width + col;

                if (row < height && col < width) {
                  REQUIRE(output_data_a[offset_out] == input_data[offset_in]);
                } else {
                  REQUIRE(output_data_a[offset_out] == border_value);
                }
              }
            }
          }
        }
      }

      AND_GIVEN("move matrix") {
        int min_side = height < width ? height : width;
        std::uniform_int_distribution<int> move_dist{-min_side, min_side};
        float                              move_x = move_dist(gen);
        float                              move_y = move_dist(gen);


        cuda::affine_matrix      mat_a{{1, 0, move_x, 0, 1, move_y}};
        cuda::perspective_matrix mat_p{{1, 0, move_x, 0, 1, move_y, 0, 0, 1}};


        WHEN("warp") {
          cuda::warpAffine(input, output_a, mat_a, 0, 0, border_value);
          cuda::warpPerspective(input, output_p, mat_p, 0, 0, border_value);


          THEN("check that output matrix was moved") {
            std::vector<char> output_data_a(output_a.total());
            output_a.download(output_data_a.data());

            std::vector<char> output_data_p(output_p.total());
            output_p.download(output_data_p.data());


            REQUIRE(output_data_p == output_data_a);


            for (size_t row = 0; row < height_output; ++row) {
              for (size_t col = 0; col < width_output; ++col) {
                size_t offset_out = row * width_output + col;

                size_t input_row = row - move_y;
                size_t input_col = col - move_x;
                if (input_row < height && input_col < width) {
                  size_t offset_in = input_row * width + input_col;
                  REQUIRE(output_data_a[offset_out] == input_data[offset_in]);
                } else {
                  REQUIRE(output_data_a[offset_out] == border_value);
                }
              }
            }
          }
        }
      }
    }

    GIVEN("async pipeline") {
      cuda::stream        stream{1};
      cuda::gpu_mat<char> input(height, width, input_data.data(), stream);
      cuda::gpu_mat<char> output_async(height, width, stream);
      cuda::gpu_mat<char> output_sync(height, width);


      AND_GIVEN("affine transformation matrix") {
        std::uniform_int_distribution<int> t_mat_val_dist{-10, 10};
        cuda::affine_matrix                mat_a{{(float)t_mat_val_dist(gen),
                                   (float)t_mat_val_dist(gen),
                                   (float)t_mat_val_dist(gen),
                                   (float)t_mat_val_dist(gen),
                                   (float)t_mat_val_dist(gen),
                                   (float)t_mat_val_dist(gen)}};


        WHEN("affine warp") {
          cuda::warpAffine(input,
                           output_async,
                           mat_a,
                           0,
                           0,
                           border_value,
                           stream);
          cuda::warpAffine(input, output_sync, mat_a, 0, 0, border_value);


          THEN("check that async and sync outputs are equal") {
            output_async.download(stream);
            stream.synchronize();


            std::vector<char> output_data_async(output_async.total());
            std::memcpy(output_data_async.data(),
                        output_async.host_ptr(),
                        output_data_async.size());

            std::vector<char> output_data_sync(output_sync.total());
            output_sync.download(output_data_sync.data());


            REQUIRE(output_data_sync == output_data_async);
          }
        }
      }
    }
  }
}

// TODO add test for warping with rotaion matrix
// TODO add test for complete perspective warping
