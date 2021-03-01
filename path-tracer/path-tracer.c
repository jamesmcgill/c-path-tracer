#include <assert.h>
#include <stdio.h>   // printf
#include <stdint.h>  // uint32_t
#include <stdbool.h> // bool
#include <stdlib.h>
#include <string.h>
#include <time.h>    // timespec_get

#define _USE_MATH_DEFINES
#include <math.h>    // M_PI, cos, sin
#include <float.h>   // FLT_MAX

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

typedef int8_t    i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef int64_t  i64;
typedef uint8_t   u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float    f32;
typedef double   f64;


#ifdef _WIN32
#include "win32-threading.h"
#else
#include "c11-threading.h"
#endif

//------------------------------------------------------------------------------
// Output image details
static const char* const IMAGE_FILENAME = "test.bmp";

enum
{
    NUM_THREADS = 32,

    // Output image details
    IMAGE_WIDTH  = 1920,
    IMAGE_HEIGHT = 1080,

    // Rendering parameters
    NUM_SAMPLES = 200 / 20,
    TILE_WIDTH = 64,        // Multiples of 64 (64 is the assumed cache length)

    // Program constants
    NUM_COLOR_COMPONENTS = 4,
};

static const f32 NUM_SAMPLES_RECIP = 1.0f / NUM_SAMPLES;
//------------------------------------------------------------------------------
typedef struct PxColor PxColor;
struct PxColor
{
    u8 red;
    u8 green;
    u8 blue;
    u8 alpha;
};

//------------------------------------------------------------------------------
// Math
//------------------------------------------------------------------------------
static const f32 degrees_to_radians = (f32)(M_PI) / 180.0f;

i32 imin(const i32 a, const i32 b)
{
    return (a < b) ? a : b;
}

f32 clampf(f32 val)
{
    if (val > 1.0f) { return 1.0f; }
    if (val < 0.0f) { return 0.0f; }
    return val;
}

bool approx_equal_f32(f32 a, f32 b)
{
    // https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
    i32 A = (i32)a;
    i32 B = (i32)b;
    if (A < 0 != B < 0) { return false; }  // Different sign means they don't match

    static const i32 max_ulps_diff = 0x2;

    i32 ulps_diff = abs(A - B);
    return (ulps_diff <= max_ulps_diff);
}

typedef struct Vec3 Vec3;
struct Vec3
{
    f32 x;
    f32 y;
    f32 z;
};

typedef struct Vec2 Vec2;
struct Vec2
{
    f32 x;
    f32 y;
};

Vec3 add_vec3(const Vec3 lhs, const Vec3 rhs)
{
    return (Vec3)
    {
        .x = lhs.x + rhs.x,
        .y = lhs.y + rhs.y,
        .z = lhs.z + rhs.z,
    };
}

Vec3 subtract_vec3(const Vec3 lhs, const Vec3 rhs)
{
    return (Vec3)
    {
        .x = lhs.x - rhs.x,
        .y = lhs.y - rhs.y,
        .z = lhs.z - rhs.z,
    };
}

Vec3 multiply_vec3(const Vec3 lhs, const Vec3 rhs)
{
    return (Vec3)
    {
        .x = lhs.x * rhs.x,
        .y = lhs.y * rhs.y,
        .z = lhs.z * rhs.z,
    };
}

Vec3 scale_vec3(const Vec3 v, const f32 scale)
{
    return (Vec3)
    {
        .x = v.x * scale,
        .y = v.y * scale,
        .z = v.z * scale,
    };
}

Vec3 sqrt_vec3(const Vec3 v)
{
    return (Vec3)
    {
        .x = sqrtf(v.x),
        .y = sqrtf(v.y),
        .z = sqrtf(v.z),
    };
}

f32 dot_vec3(const Vec3 lhs, const Vec3 rhs)
{
    return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z);
}

f32 length_squared_vec3(const Vec3 v)
{
    return dot_vec3(v, v);
}

f32 length_vec3(const Vec3 v)
{
    return sqrtf(length_squared_vec3(v));
}

Vec3 normalize_vec3(const Vec3 v)
{
    f32 inv_len = 1.0f / length_vec3(v);
    return scale_vec3(v, inv_len);
}

Vec3 cross_vec3(const Vec3 lhs, const Vec3 rhs)
{
    return (Vec3)
    {
        .x = (lhs.y * rhs.z) - (lhs.z * rhs.y),
        .y = (lhs.z * rhs.x) - (lhs.x * rhs.z),
        .z = (lhs.x * rhs.y) - (lhs.y * rhs.x),
    };
}

Vec3 clamp_vec3(const Vec3 v)
{
    return (Vec3)
    {
        .x = clampf(v.x),
        .y = clampf(v.y),
        .z = clampf(v.z),
    };
}

//------------------------------------------------------------------------------
Vec3 reflect_vec3(Vec3 v, Vec3 surface_normal)
{
    const f32 double_height = dot_vec3(v, surface_normal) * -2.0f;
    Vec3 reflect_height = scale_vec3(surface_normal, double_height);
    return add_vec3(v, reflect_height);
}

//------------------------------------------------------------------------------
Vec3 refract_theta_vec3(
    Vec3 v,
    f32 in_cos_theta,
    Vec3 surface_normal,
    f32 ni_over_nt)
{
    assert(approx_equal_f32(1.0f, length_squared_vec3(v)) && "v must be normalized");

    const Vec3 in_ortho_len = scale_vec3(surface_normal, in_cos_theta);
    Vec3 ortho = add_vec3(v, in_ortho_len);
    ortho = scale_vec3(ortho, ni_over_nt);

    const f32 parallel_len = -sqrtf( fabsf(1.0f - length_squared_vec3(ortho)) );
    const Vec3 parallel = scale_vec3(surface_normal, parallel_len);

    return add_vec3(ortho, parallel);
}

//------------------------------------------------------------------------------
Vec3 refract_vec3(Vec3 v, Vec3 surface_normal, f32 ni_over_nt)
{
    const Vec3 unit_v = normalize_vec3(v);
    const Vec3 inv_v = scale_vec3(unit_v, -1.0f);
    const f32 cos_theta = fminf(dot_vec3(inv_v, surface_normal),  1.0f);

    return refract_theta_vec3(unit_v, cos_theta, surface_normal, ni_over_nt);
}

//------------------------------------------------------------------------------
// Random
//------------------------------------------------------------------------------
typedef struct xorshift128p_state RandState;
struct xorshift128p_state
{
    uint64_t a, b;
};

/* The state must be seeded so that it is not all zero */
uint64_t xorshift128p(RandState* state)
{
    uint64_t t = state->a;
    uint64_t const s = state->b;
    state->a = s;
    t ^= t << 23;		// a
    t ^= t >> 17;		// b
    t ^= s ^ (s >> 26);	// c
    state->b = t;
    return t + s;
}

RandState create_rand_state(void)
{
    struct xorshift128p_state state;
    state.a = 0xF4F9F632A33FD8CF;
    state.b = 0x59A42E9F51B09B89;
    return state;
}

f32 rand_f32(RandState* state)
{
    u32 r = (u32)xorshift128p(state);
    return r / (f32)UINT32_MAX;
}

f32 rand_f32_range(RandState* state, f32 min, f32 max)
{
    const f32 range = max - min;
    return (rand_f32(state) * range) + min;
}

Vec3 rand_vec3(RandState* state)
{
    return (Vec3)
    {
        .x = rand_f32(state),
        .y = rand_f32(state),
        .z = rand_f32(state),
    };
}

Vec3 rand_vec3_in_range(RandState* state, f32 min, f32 max)
{
    return (Vec3)
    {
        .x = rand_f32_range(state, min, max),
        .y = rand_f32_range(state, min, max),
        .z = rand_f32_range(state, min, max),
    };
}

Vec3 rand_vec3_in_unit_sphere(RandState* state)
{
    Vec3 p;
    while (true)
    {
        p.x = rand_f32_range(state, -1.0f, 1.0f);
        p.y = rand_f32_range(state, -1.0f, 1.0f);
        p.z = rand_f32_range(state, -1.0f, 1.0f);
        if (length_squared_vec3(p) < 1.0f) { break; }
    }
    return p;
}

Vec3 rand_vec3_in_unit_disc(RandState* state)
{
    Vec3 p;
    while (true)
    {
        p.x = rand_f32_range(state, -1.0f, 1.0f);
        p.y = rand_f32_range(state, -1.0f, 1.0f);
        p.z = 0;
        if (length_squared_vec3(p) < 1.0f) { break; }
    }
    return p;
}

//------------------------------------------------------------------------------
// Ray
//------------------------------------------------------------------------------
typedef struct Ray Ray;
struct Ray
{
    Vec3 origin;
    Vec3 direction;
};

Vec3 ray_at_t(const Ray ray, const f32 t)
{
    return add_vec3(ray.origin, scale_vec3(ray.direction, t));
}

//------------------------------------------------------------------------------
// Materials
//------------------------------------------------------------------------------
enum MaterialType
{
    MATERIAL_LAMBERTIAN,
    MATERIAL_METAL,
    MATERIAL_DIELECTRIC
};

typedef struct Material Material;
struct Material
{
    enum MaterialType type;
    union
    {
        struct { Vec3 diff_albedo; };             // Lambertian
        struct { Vec3 metal_albedo; f32 fuzz; };  // Metal
        struct { f32 refract_index; };            // Dielectric
    };
};

//------------------------------------------------------------------------------
// Scene
//------------------------------------------------------------------------------
typedef struct Sphere Sphere;
struct Sphere
{
    Vec3 position;
    f32 radius;
    Material material;
};

enum
{
    GRID_SIZE = 22,
    NUM_SPHERES = 4 + (GRID_SIZE * GRID_SIZE)
};

typedef struct Scene Scene;
struct Scene
{
    Sphere* spheres;
};

//------------------------------------------------------------------------------
Scene create_basic_scene(void)
{
    assert(NUM_SPHERES == 5);

    Scene scene;
    scene.spheres = malloc(sizeof(Sphere[NUM_SPHERES]));

    scene.spheres[0] = (Sphere)
    {
        .position = { 0.0f, -100.5f, -1.0f },
        .radius = 100.0f,
        .material = (Material)
        {
            .type = MATERIAL_LAMBERTIAN,
            .diff_albedo = (Vec3){ 0.8f, 0.8f, 0.0f },
        }
    };
    scene.spheres[1] = (Sphere)
    {
        .position = { 0.0f, 0.0f, -1.0f },
        .radius = 0.5f,
        .material = (Material)
        {
            .type = MATERIAL_LAMBERTIAN,
            .diff_albedo = (Vec3){ 0.1f, 0.2f, 0.5f },
        }
    };
    scene.spheres[2] = (Sphere)
    {
        .position = { 1.0f, 0.0f, -1.0f },
        .radius = 0.5f,
        .material = (Material)
        {
            .type = MATERIAL_METAL,
            .metal_albedo = (Vec3){ 0.8f, 0.6f, 0.2f },
            .fuzz = 0.0f,
        }
    };
    scene.spheres[3] = (Sphere)
    {
        .position = { -1.0f, 0.0f, -1.0f },
        .radius = 0.5f,
        .material = (Material)
        {
            .type = MATERIAL_DIELECTRIC,
            .refract_index = 1.5f,
        }
    };
    scene.spheres[4] = (Sphere)
    {
        .position = { -1.0f, 0.0f, -1.0f },
        .radius = -0.45f,
        .material = (Material)
        {
            .type = MATERIAL_DIELECTRIC,
            .refract_index = 1.5f,
        }
    };

    return scene;
}
//------------------------------------------------------------------------------
Scene create_random_scene(RandState* rand_state)
{
    assert(NUM_SPHERES == 4 + (GRID_SIZE * GRID_SIZE));

    Scene scene;
    scene.spheres = malloc(sizeof(Sphere[NUM_SPHERES]));

    const Material ground_material = (Material)
    {
        .type = MATERIAL_LAMBERTIAN,
        .diff_albedo = { .x = 0.5f, .y = 0.5f, .z = 0.5f },
    };
    const Material glass_material = (Material)
    {
        .type = MATERIAL_DIELECTRIC,
        .refract_index = 1.5f,
    };
    const Material metal_material = (Material)
    {
        .type = MATERIAL_METAL,
        .metal_albedo = { .x = 0.7f, .y = 0.6f, .z = 0.5f },
        .fuzz = 0.0f,
    };
    const Material diffuse_material = (Material)
    {
        .type = MATERIAL_LAMBERTIAN,
        .diff_albedo = { .x = 0.4f, .y = 0.2f, .z = 0.1f },
    };

    const f32 large_radius = 1.0f;
    const f32 small_radius = 0.2f;

    // TODO: Why is this position avoided?
    const Vec3 avoid_pos = { .x = 4.0f, .y = 0.2f, .z = 0.0f };

    for (int u = 0; u < GRID_SIZE; ++u)
    {
        for (int v = 0; v < GRID_SIZE; ++v)
        {
            Vec3 position;
            do
            {
                position = (Vec3)
                {
                    .x = (f32)u - (GRID_SIZE / 2) + rand_f32_range(rand_state, 0.0, 0.9f),
                    .y = small_radius,
                    .z = (f32)v - (GRID_SIZE / 2) + rand_f32_range(rand_state, 0.0, 0.9f),
                };
            } while (length_squared_vec3(subtract_vec3(position, avoid_pos)) <= 0.9f);

            // Materials
            Material material;
            const f32 material_choice = rand_f32(rand_state);
            if (material_choice < 0.8f)
            {
                material.type = MATERIAL_LAMBERTIAN;
                material.diff_albedo = multiply_vec3(
                    rand_vec3(rand_state), rand_vec3(rand_state));
            }
            else if (material_choice < 0.95f)
            {
                material.type = MATERIAL_METAL;
                material.metal_albedo =
                    rand_vec3_in_range(rand_state, 0.5f, 1.0f);
                material.fuzz = rand_f32_range(rand_state, 0.0f, 0.5f);
            }
            else
            {
                material = glass_material;
            }

            scene.spheres[u * GRID_SIZE + v] = (Sphere)
            {
                .position = position,
                .radius = small_radius,
                .material = material,
            };
        } // for v
    } // for u

    // Large Spheres
    scene.spheres[GRID_SIZE * GRID_SIZE + 0] = (Sphere)
    {
        .position = { .x = 0.0f, .y = -1000.0f, .z = 0.0f },
        .radius = 1000.0,
        .material = ground_material,
    };
    scene.spheres[GRID_SIZE * GRID_SIZE + 1] = (Sphere)
    {
        .position = { .x = 4.0f, .y = 1.0f, .z = 0.0f },
        .radius = large_radius,
        .material = metal_material,
    };
    scene.spheres[GRID_SIZE * GRID_SIZE + 2] = (Sphere)
    {
        .position = { .x = 0.0f, .y = 1.0f, .z = 0.0f },
        .radius = large_radius,
        .material = glass_material,
    };
    scene.spheres[GRID_SIZE * GRID_SIZE + 3] = (Sphere)
    {
        .position = {.x = -4.0, .y = 1.0, .z = 0.0 },
        .radius = large_radius,
        .material = diffuse_material,
    };

    return scene;
}

//------------------------------------------------------------------------------
// Collision
//------------------------------------------------------------------------------
typedef struct HitInfo HitInfo;
struct HitInfo
{
    Vec3 point;
    Vec3 surface_normal;
    Material material;
    f32 t;
    bool front_face;
} ;

typedef struct ScatterInfo ScatterInfo;
struct ScatterInfo
{
    Ray ray;
    Vec3 attenuation;
};


//------------------------------------------------------------------------------
bool scatter_lambertian(
    ScatterInfo* scatter_info,
    const Ray ray,
    const HitInfo hit_info,
    RandState* rand_state)
{
    assert(hit_info.material.type == MATERIAL_LAMBERTIAN);

    const Vec3 normal_sphere_pos =
        add_vec3(hit_info.point, hit_info.surface_normal);
    const Vec3 reflect_point =
        add_vec3(normal_sphere_pos, rand_vec3_in_unit_sphere(rand_state));

    scatter_info->ray = (Ray)
    {
        .origin = hit_info.point,
        .direction = subtract_vec3(reflect_point, hit_info.point),
    };
    scatter_info->attenuation = hit_info.material.diff_albedo;

    return true;
}
//------------------------------------------------------------------------------
bool scatter_metal(
    ScatterInfo* scatter_info,
    const Ray ray,
    const HitInfo hit_info,
    RandState* rand_state)
{
    assert(hit_info.material.type == MATERIAL_METAL);

    const Vec3 reflected_dir = reflect_vec3(ray.direction, hit_info.surface_normal);
    const Vec3 fuzz_offset =
        scale_vec3(rand_vec3_in_unit_sphere(rand_state), hit_info.material.fuzz);
    const Vec3 scattered_dir = add_vec3(reflected_dir, fuzz_offset);
    if (dot_vec3(scattered_dir, hit_info.surface_normal) > 0.0f)
    {
        scatter_info->ray = (Ray)
        {
            .origin = hit_info.point,
            .direction = scattered_dir,
        };
        scatter_info->attenuation = hit_info.material.metal_albedo;
        return true;
    }

    return false;
}

//------------------------------------------------------------------------------
f32 schlick(const f32 cos_theta, const f32 refract_index)
{
    const f32 r = (1.0f - refract_index) / (1.0f + refract_index);
    const f32 r0 = r * r;
    return r0 + (1.0f - r0) * powf((1.0f - cos_theta), 5.0f);
}

bool scatter_dielectric(
    ScatterInfo* scatter_info,
    const Ray ray,
    const HitInfo hit_info,
    RandState* rand_state)
{
    assert(hit_info.material.type == MATERIAL_DIELECTRIC);

    // Adjust properties depending on whether ray is entering or exiting surface
    const f32 ni_over_nt = (hit_info.front_face)
        ? 1.0f / hit_info.material.refract_index
        : hit_info.material.refract_index;

    const Vec3 surface_normal = (hit_info.front_face)
        ? hit_info.surface_normal
        : scale_vec3(hit_info.surface_normal, -1.0f);

    // Obtain sin theta to check for total internal reflection
    const Vec3 unit_v = normalize_vec3(ray.direction);
    const f32 cos_theta = fminf( dot_vec3(scale_vec3(unit_v, -1.0f), surface_normal), 1.0f);
    const f32 sin_theta = sqrtf(1.0f - (cos_theta * cos_theta));

    const bool will_reflect = ( (ni_over_nt * sin_theta > 1.0f)
        || (schlick(cos_theta, ni_over_nt) > rand_f32(rand_state)) );

    const Vec3 scattered_dir = (will_reflect)
        ? reflect_vec3(unit_v, surface_normal)
        : refract_theta_vec3(unit_v, cos_theta, surface_normal, ni_over_nt);

    scatter_info->ray = (Ray)
    {
        .origin = hit_info.point,
        .direction = scattered_dir,
    };
    scatter_info->attenuation = (Vec3){.x = 1.0f, .y = 1.0f, .z = 1.0f};
    return true;
}

//------------------------------------------------------------------------------
bool is_in_range(const f32 val, const f32 min, const f32 max)
{
    if (val < min) { return false; }
    if (val > max) { return false; }
    return true;
}

//------------------------------------------------------------------------------
bool select_closest_t(f32* closest_t, f32 t1, f32 t2, f32 t_min, f32 t_max)
{
    const bool is_t1_valid = is_in_range(t1, t_min, t_max);
    const bool is_t2_valid = is_in_range(t2, t_min, t_max);

    if (!is_t1_valid && !is_t2_valid)
    {
        return false;
    }
    else if (!is_t2_valid)
    {
        *closest_t = t1;
    }
    else if (!is_t1_valid)
    {
        *closest_t = t2;
    }
    else if (t1 < t2)
    {
        *closest_t = t1;
    }
    else
    {
        *closest_t = t2;
    }
    return true;
}

//------------------------------------------------------------------------------
bool hit_test(
    HitInfo* hit_info,
    const Sphere sphere,
    const Ray ray,
    const f32 t_min,
    const f32 t_max)
{
    const f32 radius_sq = (sphere.radius * sphere.radius);
    const Vec3 displacement = subtract_vec3(ray.origin, sphere.position);

    const f32 a      = length_squared_vec3(ray.direction);
    const f32 half_b = dot_vec3(displacement, ray.direction);
    const f32 c      = length_squared_vec3(displacement) - radius_sq;

    const f32 discriminant = (half_b * half_b) - (a * c);
    if (discriminant < 0.0f) { return false; }

    const f32 discrim_root = sqrtf(discriminant);
    const f32 t1 = (-half_b - discrim_root) / a;
    const f32 t2 = (-half_b + discrim_root) / a;

    f32 t;
    if (select_closest_t(&t, t1, t2, t_min, t_max))
    {
        const Vec3 point_at_t = ray_at_t(ray, t);
        Vec3 outward_normal = subtract_vec3(point_at_t, sphere.position);
        outward_normal = scale_vec3(outward_normal, 1.0f / sphere.radius);

        hit_info->point = point_at_t;
        hit_info->t = t;
        hit_info->surface_normal = outward_normal;
        hit_info->front_face = dot_vec3(outward_normal, ray.direction) < 0.0f;
        hit_info->material = sphere.material;
        return true;
    }
    return false;
}

//------------------------------------------------------------------------------
// Camera
//------------------------------------------------------------------------------
typedef struct Camera Camera;
struct Camera
{
    Vec3 position;
    Vec3 top_left;
    Vec3 u;
    Vec3 v;
    Vec3 w;
    Vec2 px_scale;
    f32 lens_radius;
};

//------------------------------------------------------------------------------
Camera create_camera(
    Vec3 look_from,
    Vec3 look_at,
    Vec3 vup,
    u32 image_width,
    u32 image_height,
    f32 vertical_fov,
    f32 aspect_ratio,
    f32 aperture,
    f32 focus_dist)
{
    const f32 theta = degrees_to_radians * vertical_fov;
    const f32 h = tanf(theta / 2.0f);
    const f32 frustum_height = 2.0f * h;
    const f32 frustum_width = aspect_ratio * frustum_height;

    const Vec3 look_dir = subtract_vec3(look_from, look_at); // -z is forward
    const Vec3 w = normalize_vec3(look_dir);
    const Vec3 u = normalize_vec3(cross_vec3(vup, w));
    const Vec3 v = cross_vec3(w, u);

    const Vec3 focus_dir = scale_vec3(w, focus_dist);
    const Vec3 half_horiz = scale_vec3(u, frustum_width / 2.0f);
    const Vec3 half_vert = scale_vec3(v, frustum_height / 2.0f);

    // Top left of frustum viewport. Target for first ray.
    // Corresponds to top-left pixel of rendered image
    Vec3 top_left_pos = subtract_vec3(look_from, half_horiz);
    top_left_pos = add_vec3(top_left_pos, half_vert);
    top_left_pos = subtract_vec3(top_left_pos, focus_dir);

    // Scale: screen pixels projection onto frustum view
    const Vec2 px_proj =
    {
        .x = frustum_width / (f32)(image_width - 1),
        .y = frustum_height / (f32)(image_height - 1),
    };

    return (Camera)
    {
        .position = look_from,
        .top_left = top_left_pos,
        .u = u,
        .v = v,
        .w = w,
        .lens_radius = aperture * 0.5f,
        .px_scale = px_proj,
    };
}

//------------------------------------------------------------------------------



//------------------------------------------------------------------------------
// Path tracing
//------------------------------------------------------------------------------
Ray generate_ray(const Camera* camera, const i32 x, const i32 y, RandState* rand_state)
{
    // Random starting position on lens aperture
    const Vec3 rd = scale_vec3(rand_vec3_in_unit_disc(rand_state), camera->lens_radius);
    const Vec3 aperture_offset_u = scale_vec3(camera->u, rd.x);
    const Vec3 aperture_offset_v = scale_vec3(camera->v, rd.y);
    const Vec3 aperture_offset = add_vec3(aperture_offset_u, aperture_offset_v);

    // The point that corresponds to on the frustum plane
    const Vec3 horiz = scale_vec3(camera->u, (x + rand_f32(rand_state)) * camera->px_scale.x);
    const Vec3 vert  = scale_vec3(camera->v, (y + rand_f32(rand_state)) * camera->px_scale.y);
    Vec3 to_pixel = add_vec3(camera->top_left, horiz);
    to_pixel = subtract_vec3(to_pixel, vert);

    const Vec3 ray_start = add_vec3(camera->position, aperture_offset);
    const Vec3 ray_dir = subtract_vec3(to_pixel, ray_start);
    return (Ray)
    {
        .origin = ray_start,
        .direction = ray_dir
    };
}

//------------------------------------------------------------------------------
Vec3 calc_color(
    const Ray ray,
    const Scene* scene,
    RandState* rand_state,
    u32 call_depth)
{
    if (call_depth > 50)
    {
        return (Vec3) {.x = 1.0f, .y = 0.0f, .z = 0.0f};
    }

    // Test all objects in the scene
    bool hit_something = false;
    f32 closest_t = FLT_MAX;
    HitInfo hit_info;
    for (i32 i = 0; i < NUM_SPHERES; ++i)
    {
        HitInfo current_info;
        if (hit_test(&current_info, scene->spheres[i], ray, 0.001f, closest_t))
        {
            hit_something = true;
            closest_t = current_info.t;
            hit_info = current_info;
        }
    }

    if (hit_something)
    {
        bool is_scattered = false;
        ScatterInfo scatter_info;
        switch (hit_info.material.type)
        {
        case MATERIAL_LAMBERTIAN:
            is_scattered = scatter_lambertian(
                &scatter_info, ray, hit_info, rand_state);
            break;
        case MATERIAL_METAL:
            is_scattered = scatter_metal(
                &scatter_info, ray, hit_info, rand_state);
            break;
        case MATERIAL_DIELECTRIC:
            is_scattered = scatter_dielectric(
                &scatter_info, ray, hit_info, rand_state);
            break;
        };

        if (is_scattered)
        {
            Vec3 color = calc_color(
                scatter_info.ray,
                scene,
                rand_state,
                call_depth + 1
            );
            return multiply_vec3(color, scatter_info.attenuation);
        }
        else
        {
            return (Vec3){.x = 0.0f, .y = 0.0f, .z = 0.0f};
        }
    } // hit_something

    // Draw background sky gradient
    const Vec3 unit_direction = normalize_vec3(ray.direction);
    const f32 t = (unit_direction.y + 1.0f) * 0.5f;
    const f32 invT = 1.0f - t;

    const Vec3 white_tone = { 1.0f * invT, 1.0f * invT, 1.0f * invT };
    const Vec3 blue_tone = { 0.5f * t, 0.7f * t, 1.0f * t };
    return add_vec3(white_tone, blue_tone);
}

//------------------------------------------------------------------------------
// Threading
//------------------------------------------------------------------------------
typedef struct Job Job;
struct Job
{
    i32 startX; // replace with buffer index?
    i32 startY;
    i32 width; // could be constant for the program
    i32 height;
    PxColor* pixels;
};

typedef struct JobQueue JobQueue;
struct JobQueue
{
    Job* jobs;
    i32 num_jobs;
    volatile i32 next_job_idx;
    volatile i32 num_completed_jobs;
    Camera camera;
    Scene scene;
};

//------------------------------------------------------------------------------
void run_job(RandState* rand_state, Job job, Scene* scene, Camera* camera)
{
    assert(job.startX + job.width <= IMAGE_WIDTH);
    assert(job.startY + job.height <= IMAGE_HEIGHT);

    for (i32 x = job.startX; x < job.startX + job.width; ++x)
    {
        const f32 xRatio = x / (f32)IMAGE_WIDTH;
        for (i32 y = job.startY; y < job.startY + job.height; ++y)
        {
            const f32 yRatio = y / (f32)IMAGE_HEIGHT;
            Vec3 accumulated_color = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
            for (i32 sample = 0; sample < NUM_SAMPLES; ++sample)
            {
                const Ray ray = generate_ray(camera, x, y, rand_state);
                const Vec3 ray_color = calc_color(ray, scene, rand_state, 0);
                accumulated_color = add_vec3(accumulated_color, ray_color);
            } // sample

            Vec3 color = scale_vec3(accumulated_color, NUM_SAMPLES_RECIP); // Average of samples
            color = sqrt_vec3(color);                                      // Gamma correction
            job.pixels[y * IMAGE_WIDTH + x] = (PxColor)
            {
                .red = (u8)(color.x * UINT8_MAX),
                .green = (u8)(color.y * UINT8_MAX),
                .blue = (u8)(color.z * UINT8_MAX),
                .alpha = UINT8_MAX,
            };
        }
    }
}

void process_jobs_until_queue_empty(JobQueue* job_queue)
{
    RandState rand_state = create_rand_state();
    i32 job_idx = locked_increment_and_return_previous(&job_queue->next_job_idx);

    while (job_idx < job_queue->num_jobs)
    {
        run_job(&rand_state, job_queue->jobs[job_idx], &job_queue->scene, &job_queue->camera);
        i32 num_completed = locked_increment_and_return_previous(&job_queue->num_completed_jobs);
        printf("\r %d%% completed", 100 * (i32)num_completed / job_queue->num_jobs);
        fflush(stdout);

        job_idx = locked_increment_and_return_previous(&job_queue->next_job_idx);
    }
}

//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
int main(void)
{
    PxColor* pixels = malloc(sizeof(PxColor[IMAGE_WIDTH * IMAGE_HEIGHT]));

    // Scene
    struct xorshift128p_state rand_state = create_rand_state();
    Scene scene = create_random_scene(&rand_state);

    // Camera
    const Vec3 camera_pos  = { .x = 13.0f, .y = 2.0f, .z = 3.0f };
    const Vec3 look_at_pos = { .x = 0.0f,  .y = 0.0f, .z = 0.0f };
    const Vec3 v_up        = { .x = 0.0f,  .y = 1.0f, .z = 0.0f };
    const f32 vertical_fov = 120.0f;
    const f32 aspect_ratio = (f32)IMAGE_WIDTH / IMAGE_HEIGHT;
    const f32 aperture     = 0.1f;
    const f32 focus_dist   = 10.0f;  // TODO: Investigate: Changing this affects background gradient, because it effectively shrinks the vertical_fov. Is that expected from focus_dist?

    printf("Configuration: num_threads: %d, num_samples: %d, resolution: %d x %d, tile_width: %d\n\n",
        NUM_THREADS, NUM_SAMPLES, IMAGE_WIDTH, IMAGE_HEIGHT, TILE_WIDTH);
    printf("Reticulating splines...\n");
    struct timespec t0;
    i32 ret = timespec_get(&t0, TIME_UTC);

    const i32 x_steps = (IMAGE_WIDTH + TILE_WIDTH - 1) / TILE_WIDTH;
    const i32 y_steps = (IMAGE_HEIGHT + TILE_WIDTH - 1) / TILE_WIDTH;

    // Create jobs
    JobQueue job_queue;
    job_queue.num_jobs = x_steps * y_steps;
    job_queue.jobs = malloc(sizeof(Job) * job_queue.num_jobs);
    job_queue.next_job_idx = 0;
    job_queue.num_completed_jobs = 0;

    i32 x = 0;
    i32 y = 0;
    for (i32 job_idx = 0; job_idx < job_queue.num_jobs; ++job_idx)
    {
        assert(x < IMAGE_WIDTH);
        assert(y < IMAGE_HEIGHT);
        job_queue.jobs[job_idx] = (Job)
        {
            .startX = x,
            .startY = y,
            .width = imin(TILE_WIDTH, IMAGE_WIDTH - x),
            .height = imin(TILE_WIDTH, IMAGE_HEIGHT - y),
            .pixels = pixels
        };
        x += TILE_WIDTH;
        if (x >= IMAGE_WIDTH)
        {
            x = 0;
            y += TILE_WIDTH;
        }
    }
    job_queue.camera = create_camera(
        camera_pos,
        look_at_pos,
        v_up,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        vertical_fov,
        aspect_ratio,
        aperture,
        focus_dist
    );
    job_queue.scene = scene;

    // Run jobs on threads and wait for completion
    dispatch_jobs(&job_queue, NUM_THREADS);
    while (job_queue.num_completed_jobs < job_queue.num_jobs)
    {
        sleep_for_ms(500);
    }

    free(job_queue.jobs);
    free(scene.spheres);

    // Save the image to a file
    ret = stbi_write_bmp(
        IMAGE_FILENAME,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        NUM_COLOR_COMPONENTS,
        pixels
    );
    free(pixels);

    if (ret == 0)
    {
        printf("\nFAILED writing to file: %s\n", IMAGE_FILENAME);
    }
    else
    {
        printf("\nSuccessfuly written file: %s\n", IMAGE_FILENAME);
    }

    struct timespec t1;
    ret = timespec_get(&t1, TIME_UTC);
    double seconds_elapsed = (double)t1.tv_sec - t0.tv_sec;
    printf("\nRender Time: %.fs\n", seconds_elapsed);
    return 0;
}
