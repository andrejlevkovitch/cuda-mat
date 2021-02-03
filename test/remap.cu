// remap.cu

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

//


#include "cuda/remap.hpp"
#include <random>

SCENARIO("remap", "[remap]") {
  GIVEN("empty matrix") {
    cuda::gpu_mat<int>  input{0, 0, 0};
    cuda::gpu_mat<int>  output{0, 0};
    cuda::gpu_mat<int2> map{0, 0};

    WHEN("remap") {
      cuda::remap(input, output, map);

      THEN("output must be empty") {
        CHECK(output.total() == 0);
      }
    }
  }

  GIVEN("not empty matrix size and random data for matrix") {
    int height        = GENERATE(3, 100, 1024, 1153, 2049);
    int width         = GENERATE(3, 100, 1024, 1352, 2051);
    int output_height = GENERATE(3, 100, 1024, 1153, 2049);
    int output_width  = GENERATE(3, 100, 1024, 1352, 2051);

    char border_value = 0;


    std::random_device                  rd;
    std::mt19937                        gen{rd()};
    std::uniform_int_distribution<char> dist{-128, 127};


    std::vector<char> input_data(height * width);
    for (char &val : input_data) {
      val = dist(gen);
    }


    AND_GIVEN("sync pipeline matricies") {
      cuda::gpu_mat<char> input(height, width, input_data.data());
      cuda::gpu_mat<char> output(0, 0);

      AND_GIVEN("map with values outside of input matrix") {
        cuda::gpu_mat<int2> map(output_height, output_width, {-1, -1});


        WHEN("remap") {
          // XXX in this case all values in output will be equal to border_value
          remap(input, output, map, 0, 0, border_value);

          THEN("entire output must be equal to border value") {
            std::vector<decltype(output)::value_type> output_v(output_height *
                                                               output_width);
            output.download(output_v.data());

            auto found = std::find_if(output_v.begin(),
                                      output_v.end(),
                                      [border_value](char val) {
                                        if (val != border_value) {
                                          return true;
                                        }
                                        return false;
                                      });

            CHECK(found == output_v.end());
          }
        }
      }

      AND_GIVEN("map with same order") {
        std::vector<int2> map_data(output_height * output_width);
        for (int row = 0; row < output_height; ++row) {
          for (int col = 0; col < output_width; ++col) {
            int offset       = row * output_width + col;
            map_data[offset] = make_int2(col, row);
          }
        }

        cuda::gpu_mat<int2> map(output_height, output_width, map_data.data());


        WHEN("remap") {
          remap(input, output, map, 0, 0, border_value);


          THEN("input must be equal to output at intersection, and all outside "
               "of intersection must be equal to border value") {
            std::vector<char> output_data(output_height * output_width);
            output.download(output_data.data());

            for (int row = 0; row < output_height; ++row) {
              for (int col = 0; col < output_width; ++col) {
                size_t output_offset = row * output_width + col;
                size_t input_offset  = row * width + col;

                if (row < height && col < width) {
                  REQUIRE(output_data[output_offset] ==
                          input_data[input_offset]);
                } else {
                  REQUIRE(output_data[output_offset] == border_value);
                }
              }
            }
          }
        }
      }

      AND_GIVEN("map with revert order") {
        std::vector<int2> map_data(output_height * output_width);
        for (int row = 0; row < output_height; ++row) {
          for (int col = 0; col < output_width; ++col) {
            int offset       = row * output_width + col;
            map_data[offset] = make_int2(width - col - 1, height - row - 1);
          }
        }

        cuda::gpu_mat<int2> map(output_height, output_width, map_data.data());


        WHEN("remap") {
          remap(input, output, map, 0, 0, border_value);


          THEN("check revert order of output") {
            std::vector<char> output_data(output_height * output_width);
            output.download(output_data.data());


            for (int row = 0; row < output_height; ++row) {
              for (int col = 0; col < output_width; ++col) {
                size_t output_offset = row * output_width + col;

                size_t input_row    = (height - row - 1);
                size_t input_col    = (width - col - 1);
                size_t input_offset = input_row * width + input_col;

                if (input_row < height && input_col < width) {
                  REQUIRE(input_data[input_offset] ==
                          output_data[output_offset]);
                } else {
                  REQUIRE(border_value == output_data[output_offset]);
                }
              }
            }
          }
        }
      }

      AND_GIVEN("random order map") {
        std::uniform_int_distribution<int> map_dist{
            -10,
            height > width ? height + 10 : width + 10};

        std::vector<int2> map_data(output_width * output_height);
        for (int2 &val : map_data) {
          val = make_int2(map_dist(gen), map_dist(gen));
        }

        cuda::gpu_mat<int2> map(output_height, output_width, map_data.data());


        WHEN("remap") {
          remap(input, output, map, 0, 0, border_value);


          THEN("check output") {
            std::vector<char> output_data(output_width * output_height);
            output.download(output_data.data());


            // check
            for (int row = 0; row < output_height; ++row) {
              for (int col = 0; col < output_width; ++col) {
                int output_offset = row * output_width + col;
                int x             = map_data[output_offset].x;
                int y             = map_data[output_offset].y;
                int mapped_val    = output_data[output_offset];

                if (x >= 0 && x < width && y >= 0 && y < height) {
                  int orig_val = input_data[y * width + x];
                  REQUIRE(mapped_val == orig_val);
                } else {
                  REQUIRE(mapped_val == border_value);
                }
              }
            }
          }
        }
      }
    }

    AND_GIVEN("stream pipeline") {
      cuda::stream        stream{1};
      cuda::gpu_mat<char> input(height, width, input_data.data(), stream);


      AND_GIVEN("map with random values") {
        std::uniform_int_distribution<int> map_dist{
            -10,
            height > width ? height + 10 : width + 10};

        std::vector<int2> map_data(output_height * output_width);
        for (int2 &val : map_data) {
          val = make_int2(map_dist(gen), map_dist(gen));
        }

        cuda::gpu_mat<int2> map(output_height,
                                output_width,
                                map_data.data(),
                                stream);


        WHEN("remap") {
          cuda::gpu_mat<char> output(0, 0, stream);
          remap(input, output, map, 0, 0, border_value, stream);

          THEN("check result") {
            // XXX async downloading works in other way, so be carefull!
            output.download(stream);
            stream.synchronize();
            const char *output_data = output.host_ptr();


            // check
            for (int row = 0; row < output_height; ++row) {
              for (int col = 0; col < output_width; ++col) {
                int output_offset = row * output_width + col;
                int x             = map_data[output_offset].x;
                int y             = map_data[output_offset].y;
                int mapped_val    = output_data[output_offset];

                if (x >= 0 && x < width && y >= 0 && y < height) {
                  int orig_val = input_data[y * width + x];
                  REQUIRE(mapped_val == orig_val);
                } else {
                  REQUIRE(mapped_val == border_value);
                }
              }
            }
          }
        }
      }
    }
  }
}
