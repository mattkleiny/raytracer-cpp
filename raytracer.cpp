/*
 * Copyright 2018, the project authors. All rights reserved.
 * Use of this source code is governed by a MIT-style
 * license that can be found in the LICENSE.md file.
 */

#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <cassert>
#include <png++/png.hpp>

/** Defines a color in RGBA color space. */
struct Color {
  Color() : Color(0, 0, 0) {}
  Color(uint8_t red, uint8_t green, uint8_t blue) : Color(red, green, blue, 255) {}
  Color(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha) : r(red), g(green), b(blue), a(alpha) {}

  union {
    struct {
      uint8_t r;
      uint8_t g;
      uint8_t b;
      uint8_t a;
    };
    uint32_t packed;
  };

  Color operator*(float other) const {
    return Color(r * other, g * other, b * other, a * other);
  }

  Color operator+(const Color &other) const {
    return Color(r + other.r, g + other.g, b + other.b, a + other.a);
  }

  Color operator-(const Color &other) const {
    return Color(r - other.r, g - other.g, b - other.b, a - other.a);
  }

  Color operator*(const Color &other) const {
    return Color(r * other.r, g * other.g, b * other.b, a * other.a);
  }

  static const Color Black;
  static const Color Red;
  static const Color Green;
  static const Color Blue;
  static const Color White;
};

// commonly used colors
const Color Color::Black(0, 0, 0);
const Color Color::Red(255, 0, 0);
const Color Color::Green(0, 255, 0);
const Color Color::Blue(0, 0, 255);
const Color Color::White(255, 255, 255);

/** A bitmapped image of pixels. */
class Image {
 public:
  Image(uint32_t width, uint32_t height) : width_(width), height_(height) {
    pixels_ = new Color[width * height];
  }

  ~Image() {
    delete[] pixels_;
  }

  const uint32_t width() const { return width_; }
  const uint32_t height() const { return height_; }

  const Color &get(int x, int y) const {
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);

    return pixels_[x + y * width_];
  }

  void set(int x, int y, const Color &color) {
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);

    pixels_[x + y * width_] = color;
  }

  /** Exports the image to a file at the given path. */
  void save(const char *path) const {
    auto image = png::image<png::basic_rgba_pixel<uint8_t>>(width_, height_);

    for (uint32_t y = 0; y < height_; ++y) {
      for (uint32_t x = 0; x < width_; ++x) {
        const auto color = pixels_[x + y * width_];
        const auto pixel = png::basic_rgba_pixel<uint8_t>(color.r, color.g, color.b, color.a);

        image.set_pixel(x, y, pixel);
      }
    }

    image.write(path);
  }

 private:
  uint32_t width_;
  uint32_t height_;

  Color *pixels_;
};

/** Defines a vector in 3-space. */
struct Vector {
  Vector() : Vector(0, 0, 0) {}
  Vector(float x, float y, float z) : x(x), y(y), z(z) {}

  float x, y, z;

  float dot(const Vector &other) const {
    return x * other.x + y * other.y + z * other.z;
  }

  float magnitude() const {
    return static_cast<float>(sqrt(x * x + y * y + z * z));
  }

  Vector normalize() const {
    auto magnitude = this->magnitude();

    return Vector(x / magnitude, y / magnitude, z / magnitude);
  }

  Vector operator*(float value) const {
    return Vector(x * value, y * value, z * value);
  }

  Vector operator+(const Vector &other) const {
    return Vector(x + other.x, y + other.y, z + other.z);
  }

  Vector operator-(const Vector &other) const {
    return Vector(x - other.x, y - other.y, z - other.z);
  }

  static const Vector Zero;
  static const Vector UnitX;
  static const Vector UnitY;
  static const Vector UnitZ;
};

// commonly used vectors
const Vector Vector::Zero(0, 0, 0);
const Vector Vector::UnitX(1, 0, 0);
const Vector Vector::UnitY(0, 1, 0);
const Vector Vector::UnitZ(0, 0, 1);

/** Defines a ray in 3-space. */
struct Ray {
  Ray(const Vector &origin, const Vector &direction) : origin(origin), direction(direction) {}

  /** Reflects the ray about the given position via the given normal. */
  Ray reflect(const Vector &position, const Vector &normal) const {
    // TODO: implement me
  }

  /** Refracts the ray about the given position via the given normal. */
  Ray refract(const Vector &position, const Vector &normal, bool inside) const {
    // TODO: implement me
  }

  Vector origin;
  Vector direction;
};

/** Defines the material for some scene node. */
struct Material {
  Material(const Color &diffuse) : Material(diffuse, 0, 0) {}
  Material(const Color &diffuse, float reflectivity, float transpareny)
      : diffuse(diffuse), reflectivity(reflectivity), transparency(transpareny) {}

  Color diffuse;
  float reflectivity;
  float transparency;
};

/** Defines a light in the scene. */
struct Light {
  Light(const Vector &position, const Color &emissive) : position(position), emissive(emissive) {}

  Vector position;
  Color  emissive;
};

/** Defines a camera in the scene. */
struct Camera {
  Camera() : Camera(90.0f) {}
  Camera(float fieldOfView) : fieldOfView(fieldOfView) {}

  float fieldOfView;
};

/** Defines a node for use in scene rendering. */
class SceneNode {
 public:
  virtual bool intersects(const Ray &ray) const =0;

  virtual const Material &getMaterial() const =0;
};

/** Defines a sphere in the scene. */
class Sphere : public SceneNode {
 public:
  Sphere(const Vector &center, float radius, const Material &material)
      : center_(center), radius_(radius), material_(material) {}

  bool intersects(const Ray &ray) const override {
    auto line     = center_ - ray.origin;
    auto adjacent = line.dot(ray.direction);
    auto distance = line.dot(line) - (adjacent * adjacent);

    return distance < (radius_ * radius_);
  }

  const Material &getMaterial() const override {
    return material_;
  }

 private:
  Vector   center_;
  float    radius_;
  Material material_;
};

/** Defines a scene for use in our ray-tracing algorithm. */
class Scene {
  static const int MaxTraceDepth = 3;

 public:
  Scene(const Color &backgroundColor,
        const Camera &camera,
        const std::vector<Light> &lights,
        const std::vector<SceneNode *> &nodes)
      :
      backgroundColor_(backgroundColor),
      camera_(camera),
      lights_(lights),
      nodes_(nodes) {
  }

  ~Scene() {
    for (auto node : nodes_) {
      delete node;
    }
  }

  /** Renders the scene to an image of RGBA pixels. */
  std::unique_ptr<Image> render(uint32_t width, uint32_t height) const {
    auto image = std::make_unique<Image>(width, height);

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        // project a ray into the scene for each pixel in our resultant image
        const auto ray = project(x, y, width, height);

        for (const auto &node : nodes_) {
          if (node->intersects(ray)) {
            // if the ray intersects with an object, apply it's material to the image
            image->set(x, y, node->getMaterial().diffuse);
          } else {
            // otherwise, sample the background color
            image->set(x, y, backgroundColor_);
          }
        }
      }
    }

    return image;
  }

 private:
  /** Projects a ray into the scene at the given coordinates. */
  Ray project(float x, float y, float width, float height) const {
    assert(width > height);

    auto fov_adjustment = tan(to_radians(camera_.fieldOfView) / 2.0);
    auto aspect_ratio   = width / height;
    auto sensor_x       = ((((x + 0.5) / width) * 2.0 - 1.0) * aspect_ratio) * fov_adjustment;
    auto sensor_y       = (1.0f - ((y + 0.5) / height) * 2.0) * fov_adjustment;

    auto direction = Vector(sensor_x, sensor_y, -1.0).normalize();

    return Ray(Vector::Zero, direction);
  }

  /** Converts the given value to radians from degrees. */
  inline static float to_radians(float degrees) {

  }

 private:
  Camera                   camera_;
  Color                    backgroundColor_;
  std::vector<Light>       lights_;
  std::vector<SceneNode *> nodes_;
};

/** A builder syntax for constructing new scenes. */
class SceneBuilder {
 public:
  SceneBuilder &setBackgroundColor(const Color &color) {
    backgroundColor_ = color;
    return *this;
  }

  SceneBuilder &setCamera(const Camera &camera) {
    camera_ = camera;
    return *this;
  }

  SceneBuilder &addLight(const Light &light) {
    lights_.push_back(light);
    return *this;
  }

  SceneBuilder &addNode(SceneNode *node) {
    nodes_.push_back(node);
    return *this;
  }

  /** Builds the resultant scene. */
  std::unique_ptr<Scene> build() const {
    return std::make_unique<Scene>(
        backgroundColor_,
        camera_,
        lights_,
        nodes_
    );
  }

 private:
  Camera                   camera_;
  Color                    backgroundColor_;
  std::vector<Light>       lights_;
  std::vector<SceneNode *> nodes_;
};

/** Entry point for the ray-tracer. */
int main() {
  try {
    // the scene to be rendered by the ray-tracer
    std::cout << "Building scene configuration" << std::endl;
    const auto scene = SceneBuilder()
        .setBackgroundColor(Color::Black)
        .setCamera(Camera(90.0f))
        .addNode(new Sphere(Vector(0, 0, -5), 1.0, Color::Green))
        .build();

    // render the scene into an in-memory bitmap
    std::cout << "Rendering scene to image" << std::endl;
    const auto image = scene->render(800, 600);

    // render the bitmap to a .png file
    std::cout << "Rendering image to .PNG file" << std::endl;
    image->save("output.png");
  } catch (const std::exception &e) {
    std::cerr << "An unexpected error occurred:" << e.what() << std::endl;
    return -1;
  }
  return 0;
}

