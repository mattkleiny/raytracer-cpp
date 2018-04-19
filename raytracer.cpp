/*
 * Copyright 2018, the project authors. All rights reserved.
 * Use of this source code is governed by a MIT-style
 * license that can be found in the LICENSE.md file.
 */

#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
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
  Optional(T value) : value_(std::move(value)), valid_(true) {}

  /** Retrieves the underlying value from the optional.
   * Asserts the value is valid before accessing. */
  T get() const {
    assert(valid_);
    return value_;
  }

  /** Determines if the value is present. */
  bool isValid() const {
    return valid_;
  }

  /** Determines if the value is not present. */
  bool isEmpty() const {
    return !valid_;
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

  double r, g, b, a;

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

  Color operator/(const Color &other) const {
    return Color(r / other.r, g / other.g, b / other.b, a / other.a);
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

/** A bit-mapped image of pixels. */
class Image {
 public:
  Image(const Image &image) = delete;

  Image(uint32_t width, uint32_t height)
      : width_(width), height_(height) {
    pixels_ = new Color[width * height];
  }

  ~Image() {
    delete[] pixels_;
  }

  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }

  const Color &get(uint32_t x, uint32_t y) const {
    assert(x < width_);
    assert(y < height_);

    return pixels_[x + y * width_];
  }

  void set(uint32_t x, uint32_t y, const Color &color) {
    assert(x < width_);
    assert(y < height_);

    pixels_[x + y * width_] = color;
  }

  /** Loads an image from the given path. */
  static Image *load(const char *path) {
    const auto image = png::image<png::basic_rgba_pixel<uint8_t>>(path);

    auto result = new Image(
        static_cast<uint32_t>(image.get_width()),
        static_cast<uint32_t>(image.get_height())
    );

    // transfer pixels, correcting for gamma
    for (uint32_t y = 0; y < result->height(); ++y) {
      for (uint32_t x = 0; x < result->width(); ++x) {
        const auto source    = image.get_pixel(x, y);
        const auto corrected = Color(
            static_cast<float>(decodeGamma(source.red) * 255),
            static_cast<float>(decodeGamma(source.green) * 255),
            static_cast<float>(decodeGamma(source.blue) * 255),
            255
        );

        result->set(x, y, corrected);
      }
    }

    return result;
  }

  /** Exports the image to a file at the given path. */
  void save(const char *path) const {
    auto image = png::image<png::basic_rgba_pixel<uint8_t>>(width_, height_);

    for (uint32_t y = 0; y < height_; ++y) {
      for (uint32_t x = 0; x < width_; ++x) {
        // sample the pixel, re-encode to byte representation with gamma correction
        const auto source    = pixels_[x + y * width_];
        const auto corrected = png::basic_rgba_pixel<uint8_t>(
            static_cast<uint8_t>(encodeGamma(source.r) * 255.0),
            static_cast<uint8_t>(encodeGamma(source.g) * 255.0),
            static_cast<uint8_t>(encodeGamma(source.b) * 255.0),
            255
        );

        image.set_pixel(x, y, corrected);
      }
    }

    image.write(path);
  }

 private:
  static constexpr double Gamma = 2.2f;

  static inline double encodeGamma(double linear) {
    return pow(linear, 1.0f / Gamma);
  }

  static inline double decodeGamma(double linear) {
    return pow(linear, Gamma);
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

  Vector cross(const Vector &other) const {
    const auto i = y * other.z - z * other.y;
    const auto j = z * other.x - x * other.z;
    const auto k = x * other.y - y * other.x;

    return Vector(i, j, k);
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

  /** Creates a ray reflected around the given intersection point with the given normal and incidence. */
  static Ray createReflection(const Vector &normal,
                              const Vector &incidence,
                              const Vector &intersection,
                              double bias) {
    const auto origin    = intersection + (normal * bias);
    const auto direction = incidence - (normal * 2.0 * incidence.dot(normal));

    return Ray(origin, direction);
  }

  Vector origin;
  Vector direction;
};

/** Encapsulates UV texture mapping coordinates. */
struct UV {
  UV() : UV(0, 0) {}
  UV(double u, double v) : u(u), v(v) {}

  double u;
  double v;
};

/** Defines the material for some scene node. */
class Material {
 public:
  explicit Material(double reflectivity) : reflectivity(reflectivity) {}

  /** Samples the material at the given UV coordinates and returns the color. */
  virtual Color sample(const UV &coords) const =0;

  double reflectivity;
};

/** A solid material defined by a single color. */
class SolidMaterial : public Material {
 public:
  explicit SolidMaterial(const Color &albedo) : SolidMaterial(albedo, 0.0) {}
  SolidMaterial(const Color &albedo, double reflectivity)
      : albedo_(albedo), Material(reflectivity) {}

  Color sample(const UV &coords) const override {
    return albedo_;
  }

 private:
  Color albedo_;
};

/** A material defined by some texture source. */
class TexturedMaterial : public Material {
 public:
  explicit TexturedMaterial(Image *image) : TexturedMaterial(image, 0.0) {}
  TexturedMaterial(Image *image, double reflectivity)
      : image_(image), Material(reflectivity) {}

  Color sample(const UV &coords) const override {
    const auto x = wrap(coords.u, image_->width());
    const auto y = wrap(coords.v, image_->height());

    const auto color = image_->get(x, y);

    return color;
  }

 private:
  /** Wraps the given floating point range between 0 and the given upper bound. */
  static uint32_t wrap(double value, uint32_t bound) {
    const auto signedBound = static_cast<int32_t>(bound);
    const auto floatCoord  = value * static_cast<double>(bound);

    const auto wrappedCoord = (static_cast<int32_t>(floatCoord)) % signedBound;

    if (wrappedCoord < 0) {
      return static_cast<uint32_t>(wrappedCoord + signedBound);
    } else {
      return static_cast<uint32_t>(wrappedCoord);
    }
  }

  std::shared_ptr<Image> image_;
};

/** Defines a light in the scene. */
class Light {
 public:
  /** The possible types of lights that we support. */
  enum Type {
    DIRECTIONAL,
    SPHERICAL
  };

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
  explicit Camera(double fieldOfView) : fieldOfView(fieldOfView) {}

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
  virtual const Material *getMaterial() const =0;
};

/** Defines a sphere in the scene. */
class Sphere : public SceneNode {
 public:
  Sphere(const Vector &center, double radius, Material *material)
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

  const Material *getMaterial() const override {
    return material_.get();
  }

 private:
  Vector center_;
  double radius_;

  std::shared_ptr<Material> material_;
};

/** Defines a plane in the scene. */
class Plane : public SceneNode {
 public:
  Plane(const Vector &origin, const Vector &normal, Material *material)
      : origin_(origin), normal_(normal), material_(material) {}

  Optional<double> intersects(const Ray &ray) const override {
    const auto d = normal_.dot(ray.direction);

    if (d >= Epsilon) {
      const auto direction = origin_ - ray.origin;
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
    auto axisX = normal_.cross(Vector::UnitZ);

    if (axisX.magnitude() == 0.0) {
      axisX = normal_.cross(Vector::UnitY);
    }

    auto axisY = normal_.cross(axisX);

    auto line = point - origin_;

    const auto u = line.dot(axisX);
    const auto v = line.dot(axisY);

    return UV(u, v);
  }

  const Material *getMaterial() const override {
    return material_.get();
  }

 private:
  Vector origin_;
  Vector normal_;

  std::shared_ptr<Material> material_;
};

/** Defines a scene for use in our ray-tracing algorithm. */
class Scene {
  /** Maximum depth for reflection/refraction traces. */
  static constexpr auto MaxTraceDepth = 3;

 public:
  Scene(const Color &backgroundColor,
        const Camera &camera,
        const std::vector<std::shared_ptr<Light>> &lights,
        const std::vector<std::shared_ptr<SceneNode>> &nodes)
      : backgroundColor_(backgroundColor), camera_(camera), lights_(lights), nodes_(nodes) {
  }

  /** Renders the scene to an image of RGBA pixels. */
  auto render(uint32_t width, uint32_t height) const {
    auto image = std::make_unique<Image>(width, height);

    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        const auto cameraRay = project(x, y, width, height);
        const auto color     = trace(cameraRay, 0, MaxTraceDepth);

        image->set(x, y, color.clamp());
      }
    }

    return image;
  }

 private:
  /** Contains information about an intersection in the scene when tracing rays */
  struct Intersection {
    Intersection() : Intersection(nullptr, 0.0f) {}
    Intersection(std::shared_ptr<SceneNode> node, double distance)
        : node(std::move(node)), distance(distance) {}

    std::shared_ptr<SceneNode> node;
    double                     distance;
  };

  /** Samples the color at the resultant object by projecting the given ray into the scene with the given max sample depth. */
  Color trace(const Ray &ray, uint32_t depth, uint32_t maxDepth) const {
    // cap the maximum number of times a ray can bounce within the scene
    if (depth >= maxDepth) {
      return backgroundColor_;
    }

    // project a ray into the scene for each pixel in our resultant image
    const auto intersection = findIntersectingObject(ray);

    // if we're able to locate a valid intersection for this ray
    if (intersection.isValid()) {
      const auto distance = intersection.get().distance;
      const auto node     = intersection.get().node;
      const auto material = node->getMaterial();

      // calculate the hit point, normal and UV on the surface of the object
      const auto hitPoint      = ray.origin + ray.direction * distance;
      const auto surfaceNormal = node->calculateNormal(hitPoint);
      const auto surfaceUV     = node->calculateUV(hitPoint);

      // evaluate color for this pixel based on the material and it's surrounding lights
      auto color = applyDiffuseShading(distance, material, hitPoint, surfaceNormal, surfaceUV);

      // apply reflective surface properties by reflecting a ray about the surface point and sampling the resultant color
      if (material->reflectivity > 0) {
        const auto reflectionRay = Ray::createReflection(surfaceNormal, ray.direction, hitPoint, Epsilon);

        color = color * (1.0 - material->reflectivity);
        color = color + (trace(reflectionRay, depth + 1, maxDepth) * material->reflectivity);
      }

      return color;
    }

    return backgroundColor_;
  }

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

  /** Traces a ray in the scene, attempting to locate the closest object. */
  Optional<Intersection> findIntersectingObject(const Ray &ray) const {
    auto   result   = none<Intersection>();
    double distance = 9999999999.0f;

    // walk through all nodes in the scene
    for (const auto &node : nodes_) {
      const auto intersection = node->intersects(ray);

      // if our ray intersects with the node
      if (intersection.isValid()) {
        const auto hitDistance = intersection.get();

        // and the intersection point is the closest we've located so far
        if (hitDistance < distance) {
          distance = hitDistance;
          result   = some(Intersection(node, distance)); // then record the result
        }
      }
    }

    return result;
  }

  /** Applies lighting to some object's material by evaluating all lights in the scene relative to it's intersection information. */
  Color applyDiffuseShading(const double distance,
                            const Material *material,
                            const Vector &hitPoint,
                            const Vector &surfaceNormal,
                            const UV &surfaceUV) const {
    auto       color  = Color::Black;
    const auto albedo = material->sample(surfaceUV);

    // walk through all lights in the scene
    for (const auto &sceneLight : lights_) {
      const auto lightType = sceneLight->type();

      // HACK: dirty ioc around light types; difficult to get polymorphic behaviour with some much ambient state
      if (lightType == Light::DIRECTIONAL) {
        const auto light = dynamic_cast<DirectionalLight *>(sceneLight.get());

        const auto directionToLight = -light->direction;

        // cast a ray from the intersection point back to the light to see if we're in shadow
        const auto shadowRay = Ray(hitPoint + surfaceNormal * Epsilon, directionToLight);
        const auto inShadow  = findIntersectingObject(shadowRay).isValid();

        // mix light color based on distance and intensity
        const auto lightPower     = surfaceNormal.dot(directionToLight) * (inShadow ? 0.0f : light->intensity);
        const auto lightReflected = albedo / M_PI;
        const auto lightColor     = light->emissive * lightPower * lightReflected;

        color = color + albedo * lightColor;
      } else if (lightType == Light::SPHERICAL) {
        const auto light = dynamic_cast<SphericalLight *>(sceneLight.get());

        const auto directionToLight = (light->position - hitPoint).normalize();
        const auto distanceToLight  = (hitPoint - light->position).magnitude();

        const auto intensity = light->intensity / (4 * M_PI * distanceToLight);

        const auto shadowRay          = Ray(hitPoint + surfaceNormal * Epsilon, directionToLight);
        const auto shadowIntersection = findIntersectingObject(shadowRay);
        const auto inLight            = !shadowIntersection.isValid() || shadowIntersection.get().distance > distanceToLight;

        // mix light color based on distance and intensity
        const auto lightPower     = surfaceNormal.dot(directionToLight) * (inLight ? light->intensity : 0.0);
        const auto lightReflected = albedo / M_PI;
        const auto lightColor     = light->emissive * lightPower * lightReflected;

        color = color + albedo * lightColor;
      }
    }

    return color;
  }

 private:
  Color  backgroundColor_;
  Camera camera_;

  std::vector<std::shared_ptr<Light>>     lights_;
  std::vector<std::shared_ptr<SceneNode>> nodes_;
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
    lights_.push_back(std::shared_ptr<Light>(light));
    return *this;
  }

  SceneBuilder &addNode(SceneNode *node) {
    nodes_.push_back(std::shared_ptr<SceneNode>(node));
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
  Camera camera_;
  Color  backgroundColor_;

  std::vector<std::shared_ptr<Light>>     lights_;
  std::vector<std::shared_ptr<SceneNode>> nodes_;
};

/** Entry point for the ray-tracer. */
int main() {
  // the scene to be rendered by the ray-tracer
  const auto scene = SceneBuilder()
      .setBackgroundColor(Color::Black)
      .setCamera(Camera(70.0))
      .addNode(new Sphere(Vector(5, -1, -15), 2.0, new SolidMaterial(Color::Blue, 0.3)))
      .addNode(new Sphere(Vector(3, 0, -35), 1.0, new SolidMaterial(Color::Green, 0.1)))
      .addNode(new Sphere(Vector(-5.5, 0, -15), 1.0, new TexturedMaterial(Image::load("textures/checkerboard.png"), 0.3)))
      .addNode(new Plane(Vector(0, -4.2, 0), -Vector::UnitY, new SolidMaterial(Color::White, 0.1)))
      .addLight(new DirectionalLight(Vector(-1, -1, 0), Color::White, 0.33f))
      .addLight(new DirectionalLight(Vector(1, -1, 0), Color::White, 0.33f))
      .addLight(new SphericalLight(Vector(0, 3, 0), Color::White, 1.0f))
      .build();

  // render the scene into an in-memory bitmap
  const auto image = scene->render(1024, 768);

  // render the bitmap to a .png file
  image->save("output.png");

  return 0;
}
