/* Copyright 2020 The TensorFlow Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow_graphics/rendering/kernels/rasterize_triangles_impl.h"

#include "gtest/gtest.h"
#include "eigen3/Eigen/Core"

namespace {

class RasterizeTrianglesImplTest : public testing::Test {
 protected:
  void CallRasterizeTrianglesImplWithParameters(
      const float* vertices, const int32* triangles, int32 triangle_count,
      int num_layers, bool return_barycentrics, FaceCullingMode culling_mode) {
    const int num_pixels = num_layers * image_height_ * image_width_;
    triangle_ids_buffer_.resize(num_pixels);

    constexpr float kClearDepth = 1.0;
    z_buffer_.resize(num_pixels, kClearDepth);

    float* barycentrics_ptr = nullptr;
    if (return_barycentrics) {
      barycentrics_buffer_.resize(num_pixels * 3);
      barycentrics_ptr = barycentrics_buffer_.data();
    }

    RasterizeTrianglesImpl(vertices, triangles, triangle_count, image_width_,
                           image_height_, num_layers, culling_mode,
                           triangle_ids_buffer_.data(), z_buffer_.data(),
                           barycentrics_ptr);
  }
  void CallRasterizeTrianglesImpl(const float* vertices, const int32* triangles,
                                  int32 triangle_count) {
    CallRasterizeTrianglesImplWithParameters(
        vertices, triangles, triangle_count, 1, true, FaceCullingMode::kNone);
  }

  // Expects that the sum of barycentric weights at a pixel is close to a
  // given value.
  void ExpectBarycentricSumIsNear(int x, int y, float expected) const {
    constexpr float kEpsilon = 1e-6f;
    auto it = barycentrics_buffer_.begin() + y * image_width_ * 3 + x * 3;
    EXPECT_NEAR(*it + *(it + 1) + *(it + 2), expected, kEpsilon);
  }
  // Expects that a pixel is covered by verifying that its barycentric
  // coordinates sum to one.
  void ExpectIsCovered(int x, int y) const {
    ExpectBarycentricSumIsNear(x, y, 1.0);
  }
  // Expects that a pixel is not covered by verifying that its barycentric
  // coordinates sum to zero.
  void ExpectIsNotCovered(int x, int y) const {
    ExpectBarycentricSumIsNear(x, y, 0.0);
  }

  int image_height_ = 480;
  int image_width_ = 640;
  int channels_ = 3;
  std::vector<float> barycentrics_buffer_;
  std::vector<int32> triangle_ids_buffer_;
  std::vector<float> z_buffer_;
};

TEST_F(RasterizeTrianglesImplTest, WorksWhenPixelIsOnTriangleEdge) {
  // Verifies that a pixel that lies exactly on a triangle edge is considered
  // inside the triangle.
  image_width_ = 641;
  const int x_pixel = image_width_ / 2;
  const float x_ndc = 0.0;
  constexpr int yPixel = 5;

  const std::vector<float> vertices = {x_ndc, -1.0, 0.5, 1.0,  x_ndc, 1.0,
                                       0.5,   1.0,  0.5, -1.0, 0.5,   1.0};
  {
    const std::vector<int32> triangles = {0, 1, 2};

    CallRasterizeTrianglesImpl(vertices.data(), triangles.data(), 1);

    ExpectIsCovered(x_pixel, yPixel);
  }
  {
    // Test the triangle with the same vertices in reverse order.
    const std::vector<int32> triangles = {2, 1, 0};

    CallRasterizeTrianglesImpl(vertices.data(), triangles.data(), 1);

    ExpectIsCovered(x_pixel, yPixel);
  }
}

TEST_F(RasterizeTrianglesImplTest, CoversEdgePixelsOfImage) {
  // Verifies that the pixels along image edges are correctly covered.

  const std::vector<float> vertices = {-1.0, -1.0, 0.0, 1.0, 1.0, -1.0,
                                       0.0,  1.0,  1.0, 1.0, 0.0, 1.0,
                                       -1.0, 1.0,  0.0, 1.0};
  const std::vector<int32> triangles = {0, 1, 2, 0, 2, 3};

  CallRasterizeTrianglesImpl(vertices.data(), triangles.data(), 2);

  ExpectIsCovered(0, 0);
  ExpectIsCovered(image_width_ - 1, 0);
  ExpectIsCovered(image_width_ - 1, image_height_ - 1);
  ExpectIsCovered(0, image_height_ - 1);
}

TEST_F(RasterizeTrianglesImplTest, PixelOnDegenerateTriangleIsNotInside) {
  // Verifies that a pixel lying exactly on a triangle with zero area is
  // counted as lying outside the triangle.
  image_width_ = 1;
  image_height_ = 1;
  const std::vector<float> vertices = {-1.0, -1.0, 0.0, 1.0, 1.0, 1.0,
                                       0.0,  1.0,  0.0, 0.0, 0.0, 1.0};
  const std::vector<int32> triangles = {0, 1, 2};

  CallRasterizeTrianglesImpl(vertices.data(), triangles.data(), 1);

  ExpectIsNotCovered(0, 0);
}

}  // namespace
