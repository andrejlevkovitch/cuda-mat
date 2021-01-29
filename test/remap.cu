// remap.cu

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

//


#include "gpu_mat.hpp"
#include "remap.hpp"
#include <random>

TEST_CASE("remap", "[remap]") {
  SECTION("try remap empty matrix") {
    cuda::gpu_mat<int> input{0, 0, 0};
    cuda::gpu_mat<int> output{0, 0};
    cuda::gpu_mat<int> x_map{0, 0};
    cuda::gpu_mat<int> y_map{0, 0};

    cuda::remap(input, output, x_map, y_map);

    CHECK(output.total() == 0);
  }

  SECTION("map with values outside of input matrix") {
    int height = GENERATE(3, 100, 512, 1024, 1153, 2048, 2049);
    int width  = GENERATE(3, 100, 512, 1024, 1352, 2048, 2051);
    int total  = width * height;

    char border_value = GENERATE(0, 1, 16, 117, -52, -34);


    cuda::gpu_mat<char> input(height, width);
    cuda::gpu_mat<char> output(0, 0);
    cuda::gpu_mat<int>  x_map(height, width, -1);
    cuda::gpu_mat<int>  y_map(height, width, -1);


    // XXX in this case all values in output will be equal to border_value
    remap(input, output, x_map, y_map, 0, 0, border_value);


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

  SECTION("process matrix with same order, so result matix must be same") {
    int height = GENERATE(3, 100, 512, 1024, 1153, 2048, 2049);
    int width  = GENERATE(3, 100, 512, 1024, 1352, 2048, 2051);
    int total  = width * height;

    char border_value = 0;


    std::random_device                  rd;
    std::mt19937                        gen{rd()};
    std::uniform_int_distribution<char> dist{-128, 127};


    std::vector<char> input_data(total);
    for (char &val : input_data) {
      val = dist(gen);
    }

    std::vector<int> x_map_data(total);
    std::vector<int> y_map_data(total);
    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        int offset         = row * width + col;
        y_map_data[offset] = row;
        x_map_data[offset] = col;
      }
    }


    cuda::gpu_mat<char> input(height, width, input_data.data());
    cuda::gpu_mat<char> output(0, 0);
    cuda::gpu_mat<int>  x_map(height, width, x_map_data.data());
    cuda::gpu_mat<int>  y_map(height, width, y_map_data.data());


    remap(input, output, x_map, y_map, 0, 0, border_value);


    std::vector<char> output_data(total);
    output.download(output_data.data());


    CHECK(input_data == output_data);
  }

  SECTION("process matrix in revert order") {
    int height = GENERATE(3, 100, 512, 1024, 1153, 2048, 2049);
    int width  = GENERATE(3, 100, 512, 1024, 1352, 2048, 2051);
    int total  = width * height;

    char border_value = 0;


    std::random_device                  rd;
    std::mt19937                        gen{rd()};
    std::uniform_int_distribution<char> dist{-128, 127};


    std::vector<char> input_data(total);
    for (char &val : input_data) {
      val = dist(gen);
    }

    std::vector<int> x_map_data(total);
    std::vector<int> y_map_data(total);
    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        int offset         = row * width + col;
        y_map_data[offset] = height - row - 1;
        x_map_data[offset] = width - col - 1;
      }
    }


    cuda::gpu_mat<char> input(height, width, input_data.data());
    cuda::gpu_mat<char> output(0, 0);
    cuda::gpu_mat<int>  x_map(height, width, x_map_data.data());
    cuda::gpu_mat<int>  y_map(height, width, y_map_data.data());


    remap(input, output, x_map, y_map, 0, 0, border_value);


    std::vector<char> output_data(total);
    output.download(output_data.data());


    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        int offset        = row * width + col;
        int mapped_offset = (height - row - 1) * width + (width - col - 1);

        REQUIRE(input_data[offset] == output_data[mapped_offset]);
      }
    }
  }

  SECTION("process matrix in random order") {
    int height = GENERATE(3, 100, 512, 1024, 1153, 2048, 2049);
    int width  = GENERATE(3, 100, 512, 1024, 1352, 2048, 2051);
    int total  = width * height;

    char border_value = GENERATE(2, 0, -3, 7, 64, -32);


    std::random_device                  rd;
    std::mt19937                        gen{rd()};
    std::uniform_int_distribution<char> val_dist{-128, 127};
    std::uniform_int_distribution<int>  map_dist{-10,
                                                height > width ? height + 10
                                                               : width + 10};


    std::vector<char> input_data(total);
    for (char &val : input_data) {
      val = val_dist(gen);
    }

    std::vector<int> x_map_data(total);
    std::vector<int> y_map_data(total);
    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        int offset         = row * width + col;
        y_map_data[offset] = map_dist(gen);
        x_map_data[offset] = map_dist(gen);
      }
    }


    cuda::gpu_mat<char> input(height, width, input_data.data());
    cuda::gpu_mat<char> output(0, 0);
    cuda::gpu_mat<int>  x_map(height, width, x_map_data.data());
    cuda::gpu_mat<int>  y_map(height, width, y_map_data.data());


    remap(input, output, x_map, y_map, 0, 0, border_value);


    std::vector<char> output_data(total);
    output.download(output_data.data());


    // check
    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        int  offset     = row * width + col;
        int  x          = x_map_data[offset];
        int  y          = y_map_data[offset];
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

  SECTION("async process matrix in random order") {
    int height = GENERATE(3, 100, 512, 1024, 1153, 2048, 2049);
    int width  = GENERATE(3, 100, 512, 1024, 1352, 2048, 2051);
    int total  = width * height;

    char border_value = GENERATE(2, 0, -3, 7, 64, -32);


    std::random_device                  rd;
    std::mt19937                        gen{rd()};
    std::uniform_int_distribution<char> val_dist{-128, 127};
    std::uniform_int_distribution<int>  map_dist{-10,
                                                height > width ? height + 10
                                                               : width + 10};


    std::vector<char> input_data(total);
    for (char &val : input_data) {
      val = val_dist(gen);
    }

    std::vector<int> x_map_data(total);
    std::vector<int> y_map_data(total);
    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        int offset         = row * width + col;
        y_map_data[offset] = map_dist(gen);
        x_map_data[offset] = map_dist(gen);
      }
    }


    cuda::stream stream{1};

    cuda::gpu_mat<char> input(height, width, input_data.data(), stream);
    cuda::gpu_mat<char> output(0, 0, stream);
    cuda::gpu_mat<int>  x_map(height, width, x_map_data.data(), stream);
    cuda::gpu_mat<int>  y_map(height, width, y_map_data.data(), stream);


    remap(input, output, x_map, y_map, 0, 0, border_value, stream);


    // XXX async downloading works in other way, so be carefull!
    output.download(stream);
    stream.synchronize();
    const char *output_data = output.host_ptr();


    // check
    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        int  offset     = row * width + col;
        int  x          = x_map_data[offset];
        int  y          = y_map_data[offset];
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
