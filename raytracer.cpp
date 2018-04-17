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

/** Minimum resolution for our floating point comparisons. */
const float Epsilon = 1e-6;

/** Defines a simple optional value that encodes the presence/absence of an result. */
template<typename T>
class Optional {
 public:
  Optional() : valid_(false) {}
  Optional(const T &value) : value_(value), valid_(true) {}

  T value() const {
    assert(valid_);
    return value_;
  }

  bool valid() const {
    return valid_;
  }

 private:
  T    value_;
  bool valid_;
};

/** Returns an optional value with the given value. */
template<typename T>
static Optional<T> some(const T &value) { return Optional<T>(value); }

/** Returns an optional value without any value. */
template<typename T>
static Optional<T> none() { return Optional<T>(); }

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
  Image(uint32_t width, uint32_t height)
      : width_(width), height_(height) {
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
    const auto magnitude = this->magnitude();

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

  Vector operator-() const {
    return Vector(-x, -y, -z);
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
  Ray(const Vector &origin, const Vector &direction)
      : origin(origin), direction(direction) {}

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
  Material(const Color &albedo) : Material(albedo, 0, 0) {}
  Material(const Color &albedo, float reflectivity, float transpareny)
      : albedo(albedo), reflectivity(reflectivity), transparency(transpareny) {}

  Color albedo;
  float reflectivity;
  float transparency;
};

/** Defines a light in the scene. */
struct Light {
  Light(const Vector &position, const Color &emissive)
      : position(position), emissive(emissive) {}

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
  /** Determines if the object intersects with the given ray,
   * and returns the distance along the ray at which the intersection occurs. */
  virtual Optional<float> intersects(const Ray &ray) const =0;

  /** Returns the material to use when rendering this node. */
  virtual const Material &material() const =0;
};

/** Defines a sphere in the scene. */
class Sphere : public SceneNode {
 public:
  Sphere(const Vector &center, float radius, const Material &material)
      : center_(center), radius_(radius), material_(material) {}

  Optional<float> intersects(const Ray &ray) const override {
    const auto line      = center_ - ray.origin;
    const auto adjacent  = line.dot(ray.direction);
    const auto distance2 = line.dot(line) - (adjacent * adjacent);
    const auto radius2   = radius_ * radius_;

    if (distance2 > radius2) {
      return none<float>();
    }

    const auto thc = sqrt(radius2 - distance2);
    const auto t0  = adjacent - thc;
    const auto t1  = adjacent + thc;

    if (t0 < 0.0 && t1 < 0.0) {
      return none<float>();
    }

    const auto distance = t0 < t1 ? t0 : t1;

    return some(static_cast<float>(distance));
  }

  const Material &material() const override {
    return material_;
  }

 private:
  Vector   center_;
  float    radius_;
  Material material_;
};

/** Defines a plane in the scene. */
class Plane : public SceneNode {
 public:
  Plane(const Vector &point, const Vector &normal, const Material &material)
      : point_(point), normal_(normal), material_(material) {}

  Optional<float> intersects(const Ray &ray) const override {
    const auto d = normal_.dot(ray.direction);

    if (d >= Epsilon) {
      const auto direction = point_ - ray.origin;
      const auto distance  = direction.dot(normal_) / d;

      if (distance >= 0.0f) {
        return some(distance);
      }
    }

    return none<float>();
  }

  const Material &material() const override {
    return material_;
  }

 private:
  Vector   point_;
  Vector   normal_;
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
      : backgroundColor_(backgroundColor), camera_(camera), lights_(lights), nodes_(nodes) {
  }

  ~Scene() {
    // we've taken direct ownership of the scene nodes; so go ahead and manually deallocate them
    for (const auto node : nodes_) {
      delete node;
    }
  }

  /** Renders the scene to an image of RGBA pixels. */
  auto render(uint32_t width, uint32_t height) const {
    auto image = std::make_unique<Image>(width, height);

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        // project a ray into the scene for each pixel in our resultant image
        const auto ray  = project(x, y, width, height);
        const auto node = findClosestNode(ray);

        // if we're able to locate a valid object for this ray
        if (node.valid()) {
          // sample the node's material
          const auto material = node.value()->material();
          image->set(x, y, material.albedo);
        } else {
          // sample the background color, otherwise
          image->set(x, y, backgroundColor_);
        }
      }
    }

    return image;
  }

 private:
  /** Projects a ray into the scene at the given coordinates. */
  Ray project(float x, float y, float width, float height) const {
    assert(width > height);

    const auto fov_adjustment = tan(to_radians(camera_.fieldOfView) / 2.0);
    const auto aspect_ratio   = width / height;
    const auto sensor_x       = ((((x + 0.5) / width) * 2.0 - 1.0) * aspect_ratio) * fov_adjustment;
    const auto sensor_y       = (1.0f - ((y + 0.5) / height) * 2.0) * fov_adjustment;

    const auto direction = Vector(static_cast<float>(sensor_x), static_cast<float>(sensor_y), -1.0f).normalize();

    return Ray(Vector::Zero, direction);
  }

  /** Projects a ray into the scene, attempting to locate the closest object. */
  Optional<SceneNode *> findClosestNode(const Ray &ray) const {
    SceneNode *result  = nullptr;
    float     distance = 9999999999.0f;

    // walk through all nodes in the scene
    for (const auto &node : nodes_) {
      const auto intersection = node->intersects(ray);

      // if our ray intersects with the node
      if (intersection.valid()) {
        const auto d = intersection.value();

        // and the intersection point is the closest we've located so far
        if (d < distance) {
          distance = d;
          result   = node; // then record the result
        }
      }
    }

    if (result != nullptr) {
      return some(result);
    }

    return none<SceneNode *>();
  }

  /** Converts the given value to radians from degrees. */
  inline static float to_radians(float degrees) { return static_cast<float>(degrees * (M_PI / 180.0)); }

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
    const auto scene = SceneBuilder()
        .setBackgroundColor(Color::Black)
        .setCamera(Camera(90.0f))
        .addNode(new Sphere(Vector(3, 0, -5), 1.0, Color::Green))
        .addNode(new Sphere(Vector(-3, 0, -5), 1.0, Color::Red))
        .addNode(new Plane(Vector(0, -3, 0), -Vector::UnitY, Color::White))
        .addLight(Light(Vector::Zero, Color::White))
        .build();

    // render the scene into an in-memory bitmap
    const auto image = scene->render(1920, 1080);

    // render the bitmap to a .png file
    image->save("output.png");
  } catch (const std::exception &e) {
    std::cerr << "An unexpected error occurred:" << e.what() << std::endl;
    return -1;
  }
  return 0;
}

