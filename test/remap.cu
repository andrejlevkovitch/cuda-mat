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
    int height = GENERATE(3, 100, 512, 1024, 1153, 2048, 2049);
    int width  = GENERATE(3, 100, 512, 1024, 1352, 2048, 2051);
    int total  = width * height;

    char border_value = GENERATE(0, 1, 16, 117, -52, -34);


    std::random_device                  rd;
    std::mt19937                        gen{rd()};
    std::uniform_int_distribution<char> dist{-128, 127};


    std::vector<char> input_data(total);
    for (char &val : input_data) {
      val = dist(gen);
    }


    AND_GIVEN("sunc pipeline matricies") {
      cuda::gpu_mat<char> input(height, width, input_data.data());
      cuda::gpu_mat<char> output(0, 0);

      AND_GIVEN("map with values outside of input matrix") {
        cuda::gpu_mat<int2> map(height, width, {-1, -1});


        WHEN("remap") {
          // XXX in this case all values in output will be equal to border_value
          remap(input, output, map, 0, 0, border_value);

          THEN("entire output must be equal to border value") {
            std::vector<decltype(output)::value_type> output_v(total);
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
        std::vector<int2> map_data(total);
        for (int row = 0; row < height; ++row) {
          for (int col = 0; col < width; ++col) {
            int offset       = row * width + col;
            map_data[offset] = make_int2(col, row);
          }
        }

        cuda::gpu_mat<int2> map(height, width, map_data.data());


        WHEN("remap") {
          remap(input, output, map, 0, 0, border_value);


          THEN("input must be equal to output") {
            std::vector<char> output_data(total);
            output.download(output_data.data());


            CHECK(input_data == output_data);
          }
        }
      }

      AND_GIVEN("map with revert order") {
        std::vector<int2> map_data(total);
        for (int row = 0; row < height; ++row) {
          for (int col = 0; col < width; ++col) {
            int offset       = row * width + col;
            map_data[offset] = make_int2(width - col - 1, height - row - 1);
          }
        }

        cuda::gpu_mat<int2> map(height, width, map_data.data());


        WHEN("remap") {
          remap(input, output, map, 0, 0, border_value);


          THEN("check revert order of output") {
            std::vector<char> output_data(total);
            output.download(output_data.data());


            for (int row = 0; row < height; ++row) {
              for (int col = 0; col < width; ++col) {
                int offset = row * width + col;
                int mapped_offset =
                    (height - row - 1) * width + (width - col - 1);

                REQUIRE(input_data[offset] == output_data[mapped_offset]);
              }
            }
          }
        }
      }

      AND_GIVEN("random order map") {
        std::uniform_int_distribution<int> map_dist{
            -10,
            height > width ? height + 10 : width + 10};

        std::vector<int2> map_data(total);
        for (int row = 0; row < height; ++row) {
          for (int col = 0; col < width; ++col) {
            int offset       = row * width + col;
            map_data[offset] = make_int2(map_dist(gen), map_dist(gen));
          }
        }

        cuda::gpu_mat<int2> map(height, width, map_data.data());


        WHEN("remap") {
          remap(input, output, map, 0, 0, border_value);


          THEN("check output") {
            std::vector<char> output_data(total);
            output.download(output_data.data());


            // check
            for (int row = 0; row < height; ++row) {
              for (int col = 0; col < width; ++col) {
                int  offset     = row * width + col;
                int  x          = map_data[offset].x;
                int  y          = map_data[offset].y;
                char mapped_val = output_data[offset];

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

        std::vector<int2> map_data(total);
        for (int row = 0; row < height; ++row) {
          for (int col = 0; col < width; ++col) {
            int offset       = row * width + col;
            map_data[offset] = make_int2(map_dist(gen), map_dist(gen));
          }
        }

        cuda::gpu_mat<int2> map(height, width, map_data.data(), stream);


        WHEN("remap") {
          cuda::gpu_mat<char> output(0, 0, stream);
          remap(input, output, map, 0, 0, border_value, stream);

          THEN("check result") {
            // XXX async downloading works in other way, so be carefull!
            output.download(stream);
            stream.synchronize();
            const char *output_data = output.host_ptr();


            // check
            for (int row = 0; row < height; ++row) {
              for (int col = 0; col < width; ++col) {
                int  offset     = row * width + col;
                int  x          = map_data[offset].x;
                int  y          = map_data[offset].y;
                char mapped_val = output_data[offset];

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
