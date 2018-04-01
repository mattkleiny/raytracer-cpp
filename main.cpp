#include <memory>
#include <vector>
#include <cassert>

/** Defines a color in RGBA color space. */
union Color {
  Color() : Color(0, 0, 0) {}
  Color(uint8_t red, uint8_t green, uint8_t blue) : red(red), green(green), blue(blue), alpha(255) {}

  union {
    struct {
      uint8_t red;
      uint8_t green;
      uint8_t blue;
      uint8_t alpha;
    };
    uint32_t packed{};
  };

  static Color Black;
  static Color Red;
  static Color Green;
  static Color Blue;
  static Color White;
};

// commonly used colors
Color Color::Black(0, 0, 0);
Color Color::Red(255, 0, 0);
Color Color::Green(0, 255, 0);
Color Color::Blue(0, 0, 255);
Color Color::White(255, 255, 255);

/** Defines a vector in 3-space. */
struct Vector {
  Vector(float x, float y, float z) : x(x), y(y), z(z) {}

  float x, y, z;

  static Vector Zero;
  static Vector UnitX;
  static Vector UnitY;
  static Vector UnitZ;
};

// commonly used vectors
Vector Vector::Zero(0, 0, 0);
Vector Vector::UnitX(1, 0, 0);
Vector Vector::UnitY(0, 1, 0);
Vector Vector::UnitZ(0, 0, 1);

/** Defines a ray in 3-space. */
struct Ray {
  Ray(float distance, const Vector &normal) : distance(distance), normal(normal) {}

  float distance;
  Vector normal;
};

/** Defines the material for some scene node. */
struct Material {
  Material(const Color &diffuse) : diffuse(diffuse), reflectivity(0), transparency(0) {}

  Color diffuse;
  float reflectivity;
  float transparency;
};

/** A bitmappedimage of RGBA pixels. */
class Image {
 public:
  Image(int32_t width, uint32_t height) : width_(width), height_(height) {
    pixels_ = new Color[width * height];
  }

  ~Image() {
    delete[] pixels_;
  }

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

 private:
  int32_t width_;
  int32_t height_;

  Color *pixels_;
};

/** Defines a node for use in scene rendering. */
class SceneNode {
 public:
  virtual bool intersects(const Ray &ray) const =0;
  virtual const Material &material() const =0;
};

/** Defines a sphere in the scene. */
class Sphere : public SceneNode {
 public:
  Sphere(const Vector &center, float radius, const Material &material)
      : center_(center), radius_(radius), material_(material) {}

  bool intersects(const Ray &ray) const override {
    return false;
  }

  const Material &material() const override {
    return material_;
  }

 private:
  Vector center_;
  float radius_;
  Material material_;
};

/** Defines a cube in the scene. */
class Cube : public SceneNode {
 public:
  Cube(const Vector &center, float radius, const Material &material)
      : center_(center), radius_(radius), material_(material) {}

  bool intersects(const Ray &ray) const override {
    return false;
  }

  const Material &material() const override {
    return material_;
  }

 private:
  Vector center_;
  float radius_;
  Material material_;
};

/** Defines a scene for use in our ray-tracing algorithm. */
class Scene {
  const int MaxTraceDepth = 3;

 public:
  Scene(std::initializer_list<SceneNode *> nodes) {
    // wrap nodes into shared pointers
    for (auto node : nodes) {
      nodes_.push_back(std::shared_ptr<SceneNode>(node));
    }
  }

  /** Renders the scene to the given image. */
  std::shared_ptr<Image> render(int width, int height) {
    auto image = std::make_shared<Image>(width, height);

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        auto ray = project(x, y);
        auto color = sample(ray, 0, MaxTraceDepth);

        image->set(x, y, color);
      }
    }

    return image;
  }

 private:
  /** Determines if any scene node intersects the given ray. */
  bool intersects(const Ray &ray) const {
    for (const auto &node : nodes_) {
      if (node->intersects(ray)) {
        return true;
      }
    }
    return false;
  }

  /** Projects a ray into the scene. */
  inline Ray project(int x, int y) const {
    return Ray(0, Vector(0, 0, 0));
  }

  /** Samples the color by projecting the given ray into the scene. */
  Color sample(const Ray &ray, int depth, const int maxDepth) const {
    return Color::Black;
  }

 private:
  std::vector<std::shared_ptr<SceneNode>> nodes_;
};

/** Entry point for the ray-tracer. */
int main() {
  // build the default scene to be rendered by the tracer
  auto scene = Scene{
      new Sphere(Vector(5, -1, -15), 2.0, Color::Red),
      new Sphere(Vector(3, 0, -35), 2.0, Color::Green),
      new Sphere(Vector(-5, 0, -15), 3.0, Color::Blue),
  };
  auto image = scene.render(1920, 1080);

  return 0;
}