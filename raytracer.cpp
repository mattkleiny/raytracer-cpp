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

/** Minimum resolution for our floating-point comparisons. */
const double Epsilon = 1e-7;

/** Clamps the given value between it's given lower and upper bounds. */
template<typename T>
inline static T clamp(const T &value, const T &lower, const T &upper) {
  if (value < lower) return lower;
  if (value > upper) return upper;

  return value;
}

/** Converts the given value to radians from degrees. */
inline static double toRadians(double degrees) { return degrees * (M_PI / 180.0); }

/** Defines a simple optional value that encodes the presence/absence of an result. */
template<typename T>
class Optional {
 public:
  Optional() : valid_(false) {}
  Optional(const T &value) : value_(value), valid_(true) {}

  /** Retrieves the underlying value from the optional.
   * Asserts the value is valid before accessing. */
  T value() const {
    assert(valid_);
    return value_;
  }

  /** Determines if the value is present. */
  bool valid() const {
    return valid_;
  }

 private:
  T    value_;
  bool valid_;
};

/** Returns an optional value with the given value. */
template<typename T>
static Optional<T> some(const T &value) {
  return Optional<T>(value);
}

/** Returns an optional value without any value. */
template<typename T>
static Optional<T> none() {
  return Optional<T>();
}

/** Defines a color in floating-point RGBA color space. */
struct Color {
  Color() : Color(0, 0, 0) {}
  Color(double red, double green, double blue) : Color(red, green, blue, 1.0f) {}
  Color(double red, double green, double blue, double alpha) : r(red), g(green), b(blue), a(alpha) {}

  double r;
  double g;
  double b;
  double a;

  /** Clamps all of the color ranges between (0, 1). */
  Color clamp() const {
    return Color(
        ::clamp(r, 0.0, 1.0),
        ::clamp(g, 0.0, 1.0),
        ::clamp(b, 0.0, 1.0),
        ::clamp(a, 0.0, 1.0)
    );
  }

  Color operator*(double other) const {
    return Color(r * other, g * other, b * other, a * other);
  }

  Color operator/(double other) const {
    return Color(r / other, g / other, b / other, a / other);
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
const Color Color::Red(1, 0, 0);
const Color Color::Green(0, 1, 0);
const Color Color::Blue(0, 0, 1);
const Color Color::White(1, 1, 1);

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
        // sample the pixel, re-encode to byte representation with gamma correction
        const auto pixel     = pixels_[x + y * width_];
        const auto corrected = png::basic_rgba_pixel<uint8_t>(
            static_cast<uint8_t>(gamma(pixel.r) * 255.0),
            static_cast<uint8_t>(gamma(pixel.g) * 255.0),
            static_cast<uint8_t>(gamma(pixel.b) * 255.0),
            255
        );

        image.set_pixel(x, y, corrected);
      }
    }

    image.write(path);
  }

 private:
  /** Corrects gamma over the given linear input. */
  static inline double gamma(double linear) {
    const double Factor = 2.2f;
    return pow(linear, 1.0f / Factor);
  }

  uint32_t width_;
  uint32_t height_;

  Color *pixels_;
};

/** Defines a vector in 3-space. */
struct Vector {
  Vector() : Vector(0, 0, 0) {}
  Vector(double x, double y, double z) : x(x), y(y), z(z) {}

  double x, y, z;

  double dot(const Vector &other) const {
    return x * other.x + y * other.y + z * other.z;
  }

  double magnitude() const {
    return sqrt(x * x + y * y + z * z);
  }

  Vector normalize() const {
    const auto magnitude = this->magnitude();

    return Vector(x / magnitude, y / magnitude, z / magnitude);
  }

  Vector operator*(double value) const {
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

  Vector origin;
  Vector direction;
};

/** Defines the material for some scene node. */
struct Material {
  Material(const Color &albedo) : Material(albedo, 0, 0) {}
  Material(const Color &albedo, double reflectivity, double transpareny)
      : albedo(albedo), reflectivity(reflectivity), transparency(transpareny) {}

  Color  albedo;
  double reflectivity;
  double transparency;
};

/** Encapsulates UV texture mapping coordinates. */
struct UV {
  UV() : UV(0, 0) {}
  UV(double u, double v) : u(u), v(v) {}

  double u;
  double v;
};

/** Defines a light in the scene. */
class Light {
 public:
  /** The possible types of lights that we support. */
  enum Type {
    DIRECTIONAL,
    SPHERICAL
  };

 public:
  /** Retrieves the type of light implemented. */
  virtual Type type() const =0;
};

/** Implements a directional light in the scene. */
class DirectionalLight : public Light {
 public:
  DirectionalLight(const Vector &direction, const Color &emissive, double intensity)
      : direction(direction), emissive(emissive), intensity(intensity) {}

  Vector direction;
  Color  emissive;
  double intensity;

  Type type() const override {
    return DIRECTIONAL;
  }
};

/** Implements a spherical light in the scene. */
class SphericalLight : public Light {
 public:
  SphericalLight(const Vector &position, const Color &emissive, double intensity)
      : position(position), emissive(emissive), intensity(intensity) {}

  Vector position;
  Color  emissive;
  double intensity;

  Type type() const override {
    return SPHERICAL;
  }
};

/** Defines a camera in the scene. */
struct Camera {
  Camera() : Camera(90.0f) {}
  Camera(double fieldOfView) : fieldOfView(fieldOfView) {}

  double fieldOfView;
};

/** Defines a node for use in scene rendering. */
class SceneNode {
 public:
  /** Determines if the object intersects with the given ray,
   * and returns the distance along the ray at which the intersection occurs. */
  virtual Optional<double> intersects(const Ray &ray) const =0;

  /** Calculates the normal of the surface of the object at the given point. */
  virtual Vector calculateNormal(const Vector &point) const =0;

  /** Calculates the UV of the surface of the object at the given point. */
  virtual UV calculateUV(const Vector &point) const =0;

  /** Returns the material to use when rendering this node. */
  virtual const Material &material() const =0;
};

/** Defines a sphere in the scene. */
class Sphere : public SceneNode {
 public:
  Sphere(const Vector &center, double radius, const Material &material)
      : center_(center), radius_(radius), material_(material) {}

  Optional<double> intersects(const Ray &ray) const override {
    const auto line      = center_ - ray.origin;
    const auto adjacent  = line.dot(ray.direction);
    const auto distance2 = line.dot(line) - (adjacent * adjacent);
    const auto radius2   = radius_ * radius_;

    if (distance2 > radius2) {
      return none<double>();
    }

    const auto thc = sqrt(radius2 - distance2);
    const auto t0  = adjacent - thc;
    const auto t1  = adjacent + thc;

    if (t0 < 0.0 && t1 < 0.0) {
      return none<double>();
    }

    const auto distance = t0 < t1 ? t0 : t1;

    return some(distance);
  }

  Vector calculateNormal(const Vector &point) const override {
    return (point - center_).normalize();
  }

  UV calculateUV(const Vector &point) const override {
    const auto spherical = point - center_;

    const auto u = (1.0 + (atan2(spherical.z, spherical.x) / M_PI)) * 0.5;
    const auto v = acos(spherical.y / radius_) / M_PI;

    return UV(u, v);
  }

  const Material &material() const override {
    return material_;
  }

 private:
  Vector   center_;
  double   radius_;
  Material material_;
};

/** Defines a plane in the scene. */
class Plane : public SceneNode {
 public:
  Plane(const Vector &point, const Vector &normal, const Material &material)
      : point_(point), normal_(normal), material_(material) {}

  Optional<double> intersects(const Ray &ray) const override {
    const auto d = normal_.dot(ray.direction);

    if (d >= Epsilon) {
      const auto direction = point_ - ray.origin;
      const auto distance  = direction.dot(normal_) / d;

      if (distance >= 0.0f) {
        return some(distance);
      }
    }

    return none<double>();
  }

  Vector calculateNormal(const Vector &point) const override {
    return -normal_;
  }

  UV calculateUV(const Vector &point) const override {
    return UV(0, 0); // TODO: implement me
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
 public:
  Scene(const Color &backgroundColor,
        const Camera &camera,
        const std::vector<Light *> &lights,
        const std::vector<SceneNode *> &nodes)
      : backgroundColor_(backgroundColor), camera_(camera), lights_(lights), nodes_(nodes) {
  }

  ~Scene() {
    // we've taken direct ownership of the scene nodes and lights;
    // so go ahead and manually deallocate them
    for (const auto light: lights_) {
      delete light;
    }
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
        const auto cameraRay    = project(x, y, width, height);
        const auto intersection = trace(cameraRay);

        // if we're able to locate a valid intersection for this ray
        if (intersection.valid()) {
          const auto distance = intersection.value().distance;
          const auto node     = intersection.value().node;
          const auto material = node->material();

          // calculate the hit point on the surface of the object
          const auto hitPoint      = cameraRay.origin + cameraRay.direction * distance;
          const auto surfaceNormal = node->calculateNormal(hitPoint);

          // evaluate color for this pixel based on the object and it's surrounding lights
          auto color = Color::Black;

          for (const auto &light__ : lights_) {
            // TODO: tidy this up
            switch (light__->type()) {
              case Light::Type::DIRECTIONAL: {
                const auto light = dynamic_cast<DirectionalLight *>(light__);

                const auto directionToLight = -light->direction;

                // cast a ray from the intersection point back to the light to see if we're in shadow
                const auto shadowRay = Ray(hitPoint + surfaceNormal * Epsilon, directionToLight);
                const auto inShadow  = trace(shadowRay).valid();

                // mix light color based on distance and intensity
                const auto lightPower     = surfaceNormal.dot(directionToLight) * (inShadow ? 0.0f : light->intensity);
                const auto lightReflected = material.albedo / M_PI;
                const auto lightColor     = light->emissive * lightPower * lightReflected;

                color = color + material.albedo * lightColor;
              }
                break;

              case Light::Type::SPHERICAL: {
                // TODO: fix spherical lighting
                const auto light = dynamic_cast<SphericalLight *>(light__);

                const auto distanceToLight  = (light->position - hitPoint).normalize();
                const auto directionToLight = distanceToLight;

                // cast a ray from the intersection point back to the light to see if we're in shadow
                const auto shadowRay          = Ray(hitPoint + surfaceNormal * Epsilon, directionToLight);
                const auto shadowIntersection = trace(shadowRay);
                const auto inLight            = !shadowIntersection.valid() || shadowIntersection.value().distance > (light->position - hitPoint).magnitude();

                // mix light color based on distance and intensity
                const auto lightPower     = surfaceNormal.dot(directionToLight) * (inLight ? (light->intensity / (4.0 * M_PI * distance)) : 0.0f);
                const auto lightReflected = material.albedo / M_PI;
                const auto lightColor     = light->emissive * lightPower * lightReflected;

                color = color + material.albedo * lightColor;
              }
                break;
            }
          }

          // sample the resultant color
          image->set(x, y, color.clamp());
        } else {
          // sample the background color, otherwise
          image->set(x, y, backgroundColor_);
        }
      }
    }

    return image;
  }

 private:
  /** Contains information about an intersection in the scene when tracing rays */
  struct Intersection {
    Intersection() : Intersection(nullptr, 0.0f) {}
    Intersection(SceneNode *node, double distance) : node(node), distance(distance) {}

    SceneNode *node;
    double    distance;
  };

  /** Projects a ray into the scene at the given coordinates. */
  Ray project(double x, double y, double width, double height) const {
    assert(width > height);

    const auto fovAdjustment = tan(toRadians(camera_.fieldOfView) / 2.0);
    const auto aspectRatio   = width / height;
    const auto sensorX       = ((((x + 0.5) / width) * 2.0 - 1.0) * aspectRatio) * fovAdjustment;
    const auto sensorY       = (1.0f - ((y + 0.5) / height) * 2.0) * fovAdjustment;

    const auto direction = Vector(sensorX, sensorY, -1.0f).normalize();

    return Ray(Vector::Zero, direction);
  }

  /** Projects a ray into the scene, attempting to locate the closest object. */
  Optional<Intersection> trace(const Ray &ray) const {
    SceneNode *result  = nullptr;
    double    distance = 9999999999.0f;

    // walk through all nodes in the scene
    for (const auto &node : nodes_) {
      const auto intersection = node->intersects(ray);

      // if our ray intersects with the node
      if (intersection.valid()) {
        const auto hitDistance = intersection.value();

        // and the intersection point is the closest we've located so far
        if (hitDistance < distance) {
          distance = hitDistance;
          result   = node; // then record the result
        }
      }
    }

    // if we've managed to locate an object, yield it's distance and node
    if (result != nullptr) {
      return some(Intersection(result, distance));
    }

    return none<Intersection>();
  }

 private:
  Camera                   camera_;
  Color                    backgroundColor_;
  std::vector<Light *>     lights_;
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

  SceneBuilder &addLight(Light *light) {
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
  std::vector<Light *>     lights_;
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
        .addLight(new DirectionalLight(-Vector::UnitY, Color::White, 0.8f))
        .addLight(new DirectionalLight(-Vector::UnitX, Color::White, 0.3f))
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
