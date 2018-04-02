/*
 * Copyright 2018, the project authors. All rights reserved.
 * Use of this source code is governed by a MIT-style
 * license that can be found in the LICENSE.md file.
 */

#include <memory>
#include <vector>
#include <cmath>
#include <cassert>

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

/** A bitmapped image of pixels with variable type. */
template<typename TPixel>
class Image {
 public:
  Image(int32_t width, uint32_t height) : width_(width), height_(height) {
    pixels_ = new TPixel[width * height];
  }

  ~Image() {
    delete[] pixels_;
  }

  const int32_t width() const { return width_; }
  const int32_t height() const { return height_; }

  const TPixel &get(int x, int y) const {
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);

    return pixels_[x + y * width_];
  }

  void set(int x, int y, const TPixel &color) {
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);

    pixels_[x + y * width_] = color;
  }

 private:
  int32_t width_;
  int32_t height_;

  TPixel *pixels_;
};

/** Defines a vector in 3-space. */
struct Vector {
  Vector() : Vector(0, 0, 0) {}
  Vector(float x, float y, float z) : x(x), y(y), z(z) {}

  float x, y, z;

  float dot(const Vector &other) const {
    return x * other.x + y * other.y + z * other.z;
  }

  Vector negate() const {
    return Vector(-x, -y, -z);
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
    auto origin    = position + normal;
    auto direction = this->direction - normal * 2.0f * this->direction.dot(normal);

    return Ray(origin, direction);
  }

  /** Refracts the ray about the given position via the given normal. */
  Ray refract(const Vector &position, const Vector &normal, bool inside) const {
    // TODO: implement me
  }

  const Vector origin;
  const Vector direction;
};

/** Defines the material for some scene node. */
struct Material {
  Material(const Color &diffuse) : Material(diffuse, 0, 0) {}
  Material(const Color &diffuse, float reflectivity, float transpareny)
      : diffuse(diffuse), reflectivity(reflectivity), transparency(transpareny) {}

  const Color diffuse;
  const float reflectivity;
  const float transparency;
};

/** Defines a light in the scene. */
struct Light {
  Light(const Vector &position, const Color &emissive) : position(position), emissive(emissive) {}

  Vector position;
  Color  emissive;
};

/** Defines a camera in the scene. */
struct Camera {
  Camera() : Camera(Vector::Zero, 75.0) {}
  Camera(const Vector &position) : Camera(position, 75.0) {}
  Camera(const Vector &position, float fieldOfView) : position(position), fieldOfView(fieldOfView) {}

  Vector position;
  float  fieldOfView;
};

/** Defines a node for use in scene rendering. */
class SceneNode {
 public:
  virtual bool intersects(const Ray &ray, float &distance) const =0;

  virtual const Material &material() const =0;
  virtual const Vector &position() const =0;
};

/** Defines a sphere in the scene. */
class Sphere : public SceneNode {
 public:
  Sphere(const Vector &center, float radius, const Material &material)
      : center_(center), radius_(radius), material_(material) {}

  bool intersects(const Ray &ray, float &distance) const override {
    return false; // TODO: implement me
  }

  const Material &material() const override {
    return material_;
  }

  const Vector &position() const override {
    return center_;
  }

 private:
  Vector   center_;
  float    radius_;
  Material material_;
};

/** Defines a cube in the scene. */
class Cube : public SceneNode {
 public:
  Cube(const Vector &center, float radius, const Material &material)
      : center_(center), radius_(radius), material_(material) {}

  bool intersects(const Ray &ray, float &distance) const override {
    return false; // TODO: implement me
  }

  const Material &material() const override {
    return material_;
  }

  const Vector &position() const override {
    return center_;
  }

 private:
  Vector   center_;
  float    radius_;
  Material material_;
};

/** Defines a scene for use in our ray-tracing algorithm. */
class Scene {
  const int MaxTraceDepth = 3;

 public:
  Scene(const Color &backgroundColor,
        const Camera &camera,
        const std::vector<Light> &lights,
        const std::vector<SceneNode *> &nodes)
      : backgroundColor_(backgroundColor), camera_(camera), lights_(lights), nodes_(nodes) {
  }

  ~Scene() {
    for (auto node : nodes_) {
      delete node;
    }
  }

  /** Renders the scene to an image of RGBA pixels. */
  std::unique_ptr<Image<Color>> render(int width, int height) const {
    auto image = std::make_unique<Image<Color>>(width, height);

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        auto ray   = project(x, y, width, height, 75, 16 / 9);
        auto color = sample(ray, 0, MaxTraceDepth);

        image->set(x, y, color);
      }
    }

    return image;
  }

 private:
  /** Projects a ray into the scene. */
  Ray project(float x, float y, int width, int height, float angle, float aspectRatio) const {
    auto pX = (2 * ((x + 0.5) / width) - 1) * angle * aspectRatio;
    auto pY = 1 - 2 * ((y + 0.5) / height) * angle;

    auto direction = Vector(pX, pY, -1);

    return Ray(Vector::Zero, direction);
  }

  /** Samples the color by projecting the given ray into the scene. */
  Color sample(const Ray &ray, int depth, const int maxDepth) const {
    float     distance = 0.0f;
    SceneNode *node    = nullptr;
    Vector    hit(0, 0, 0);
    Vector    normal(0, 0, 0);

    // no object? just use the image background color
    if (!findIntersection(ray, node, hit, normal)) {
      return backgroundColor_;
    }

    // sample image color, apply transparency and reflectivity
    const auto material     = node->material();
    auto       sampledColor = Color::Black;

    if ((material.transparency > 0 || material.reflectivity > 0) && depth < maxDepth) {
      // compute the fresnel lens effect for reflective/transparent surfaces
      auto inside = false;
      if (ray.direction.dot(normal) > 0) {
        normal = normal.negate();
        inside = true;
      }

      const auto fresnel = computeFresnel(ray.direction, normal);

      // compute reflective and refractive color by recursively tracing light along reflective and
      // refractive angles; then combine the resultant colours
      auto reflectiveColor = sample(ray.reflect(hit, normal), depth + 1, maxDepth);
      auto refractiveColor = Color::Black;

      if (material.transparency > 0) {
        reflectiveColor = sample(ray.refract(hit, normal, inside), depth + 1, maxDepth);
      }

      sampledColor = reflectiveColor * fresnel + refractiveColor * (1 - fresnel) * material.transparency * material.diffuse;
    } else {
      // compute diffuse illumination, accounting for light sources and shadows
      for (const auto &light : lights_) {
        auto transmission = Color::White;
        auto lightRay     = Ray(hit + normal, light.position - hit);

        // determine if the object is in shadow, eliminate color transmission
        for (const auto &other : nodes_) {
          if (other->intersects(lightRay, distance)) {
            transmission = Color::Black;
            break;
          }
        }

        sampledColor = sampledColor + material.diffuse * transmission * fmax(0, normal.dot(lightRay.direction)) * light.emissive;
      }
    }

    return sampledColor;
  }

  /** Determines if any scene node intersects the given ray. */
  bool findIntersection(const Ray &ray, SceneNode *&intersection, Vector &hit, Vector &normal) const {
    auto distance = 0.0f;
    auto nearest  = 999999999.0f;

    // find the nearest object along the ray direction
    for (const auto &node : nodes_) {
      if (node->intersects(ray, distance)) {
        if (distance < nearest) {
          nearest      = distance;
          intersection = node;
        }
      }
    }

    // calculate hit and normals
    if (intersection != nullptr) {
      hit    = ray.origin + ray.direction * nearest;
      normal = hit - intersection->position();

      return true;
    }

    return false;
  }

  /** Computes the fresnel lens effect for reflective/transparent surfaces
   *  see https://en.wikipedia.org/wiki/Fresnel_lens for more information. */
  inline float computeFresnel(const Vector &normal, const Vector &direction) const {
    return mix(pow(1 + direction.dot(normal), 3), 1, 0.1);
  }

  inline float mix(float a, float b, float mix) const {
    return b * mix + a * (1 - mix);
  }

 private:
  Camera                   camera_;
  Color                    backgroundColor_;
  std::vector<Light>       lights_;
  std::vector<SceneNode *> nodes_;
};

/** A builder for constructing new scenes. */
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

// the scene to be rendered by the ray-tracer
static const auto scene = SceneBuilder()
    .setBackgroundColor(Color::White)
    .setCamera(Vector::Zero)
    .addLight(Light(Vector(-20, 30, 20), Color::White))
    .addNode(new Sphere(Vector(5, -1, -15), 2.0, Color::Red))
    .addNode(new Sphere(Vector(3, 0, -35), 2.0, Color::Green))
    .addNode(new Sphere(Vector(-5, 0, -15), 3.0, Color::Blue))
    .build();

/** Entry point for the ray-tracer. */
int main() {
  auto image = scene->render(1920, 1080);

  return 0;
}

