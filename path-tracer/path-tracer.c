#include <assert.h>
#include <stdio.h>   // printf
#include <stdint.h>  // uint32_t
#include <stdbool.h> // bool
#include <stdlib.h>
#include <string.h>
#include <time.h>    // timespec_get

#define _USE_MATH_DEFINES
#include <math.h>    // M_PI, cos, sin

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

//------------------------------------------------------------------------------
// Output image details
static const char* const IMAGE_FILENAME = "test.bmp";

enum {
    // Output image details
    IMAGE_WIDTH  = 1920/4,
    IMAGE_HEIGHT = 1080/4,

    // Rendering parameters
    NUM_SAMPLES = 50,

    // Program constants
    NUM_COLOR_COMPONENTS = 4,
};

//------------------------------------------------------------------------------
typedef struct {
    u8 red;
    u8 green;
    u8 blue;
    u8 alpha;
} PxColor;

//------------------------------------------------------------------------------
// Math
//------------------------------------------------------------------------------
static const f32 degrees_to_radians = (f32)(M_PI) / 180.0f;

f32 clampf(f32 val)
{
    if (val > 1.0f) { return 1.0f; }
    if (val < 0.0f) { return 0.0f; }
    return val;
}

typedef struct {
    f32 x;
    f32 y;
    f32 z;
} Vec3;

typedef struct {
    f32 x;
    f32 y;
} Vec2;

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
// Random
//------------------------------------------------------------------------------
typedef struct xorshift128p_state {
  uint64_t a, b;
} rand_state;

/* The state must be seeded so that it is not all zero */
uint64_t xorshift128p(rand_state* state)
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

rand_state create_rand_state(void)
{
    struct xorshift128p_state state;
    state.a = 0xF4F9F632A33FD8CF;
    state.b = 0x59A42E9F51B09B89;
    return state;
}

f32 rand_f32(rand_state* state)
{
    u32 r = xorshift128p(state);
    return r / (f32)UINT32_MAX;
}

f32 rand_f32_range(rand_state* state, f32 min, f32 max)
{
    const f32 range = max - min;
    return (rand_f32(state) * range) + min;
}

Vec3 rand_vec3_in_unit_sphere(rand_state* state)
{
    Vec3 p;
    while (true)
    {
        p.x = rand_f32_range(state, -1.0, 1.0);
        p.y = rand_f32_range(state, -1.0, 1.0);
        p.z = rand_f32_range(state, -1.0, 1.0);
        if (length_squared_vec3(p) < 1.0) { break; }
    }
    return p;
}

Vec3 rand_vec3_in_unit_disc(rand_state* state)
{
    Vec3 p;
    while (true)
    {
        p.x = rand_f32_range(state, -1.0, 1.0);
        p.y = rand_f32_range(state, -1.0, 1.0);
        p.z = 0;
        if (length_squared_vec3(p) < 1.0) { break; }
    }
    return p;
}

//------------------------------------------------------------------------------
// Ray
//------------------------------------------------------------------------------
typedef struct {
    Vec3 origin;
    Vec3 direction;
} Ray;

Vec3 ray_at_t(const Ray ray, const f32 t)
{
    return add_vec3(ray.origin, scale_vec3(ray.direction, t));
}

//------------------------------------------------------------------------------
typedef struct
{
    Vec3 point;
    Vec3 surface_normal;
//  Material* material_ptr;
    f32 t;
    bool front_face;
} HitInfo;

typedef struct
{
    Ray ray;
    Vec3 attenuation;
} ScatterInfo;

//------------------------------------------------------------------------------
// Scene 
//------------------------------------------------------------------------------
typedef struct {
    Vec3 position;
} Scene;

//------------------------------------------------------------------------------
typedef struct
{
    Vec3 position;
    f32 radius;
//  Material material;
} Sphere;

//----------------------------------------------------------------------------
bool is_in_range(const f32 val, const f32 min, const f32 max)
{
    if (val < min) { return false; }
    if (val > max) { return false; }
    return true;
}

//----------------------------------------------------------------------------
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
    if (discriminant < 0.0) { return false; }

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
        hit_info->front_face = dot_vec3(outward_normal, ray.direction) < 0.0;
        //hit_info->material_ptr = &self.material;
        return true;
    }
    return false;
}


//----------------------------------------------------------------------------
// Camera
//----------------------------------------------------------------------------
typedef struct {
    Vec3 position;
    Vec3 top_left;
    Vec3 u;
    Vec3 v;
    Vec3 w;
    Vec2 px_scale;
    f32 lens_radius;
} Camera;

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
Ray generate_ray(const Camera camera, const i32 x, const i32 y, rand_state* rand_state)
{
    // Random starting position on lens aperture
    const Vec3 rd = scale_vec3(rand_vec3_in_unit_disc(rand_state), camera.lens_radius);
    const Vec3 aperture_offset_u = scale_vec3(camera.u, rd.x);
    const Vec3 aperture_offset_v = scale_vec3(camera.v, rd.y);
    const Vec3 aperture_offset = add_vec3(aperture_offset_u, aperture_offset_v);

    // The point that corresponds to on the frustum plane
    const Vec3 horiz = scale_vec3(camera.u, (x + rand_f32(rand_state)) * camera.px_scale.x);
    const Vec3 vert  = scale_vec3(camera.v, (y + rand_f32(rand_state)) * camera.px_scale.y);
    Vec3 to_pixel = add_vec3(camera.top_left, horiz);
    to_pixel = subtract_vec3(to_pixel, vert);

    const Vec3 ray_start = add_vec3(camera.position, aperture_offset);
    const Vec3 ray_dir = subtract_vec3(to_pixel, ray_start);
    return (Ray) {
        .origin = ray_start,
        .direction = ray_dir
    };
}

//------------------------------------------------------------------------------
Vec3 calc_color(
    const Ray ray,
    const Scene* scene,
    rand_state* rand_state,
    u32 call_depth)
{
    if (call_depth > 50)
    {
        return (Vec3) {.x = 0.0, .y = 0.0, .z = 0.0};
    }

    // TODO: real scene data
    i32 num_spheres = 1;
    Sphere spheres[1] =
    {
        (Sphere){.position = { 0.0f, 0.0f, 0.0f }, .radius = 10.0f }
    };

    // Test all objects in the scene
    bool hit_something = false;
    f32 closest_t = 99999.0f;   // TODO: math.f32_max;
    HitInfo hit_info;
    for (i32 i = 0; i < num_spheres; ++i)
    {
        HitInfo current_info;
        if (hit_test(&current_info, spheres[i], ray, 0.001, closest_t))
        {
            hit_something = true;
            closest_t = current_info.t;
            hit_info = current_info;  // TODO: do I really need to deep copy?
        }
    }

    if (hit_something)
    {
        bool is_scattered = false;
        ScatterInfo scatter_info;

        // TODO: this is temp code. similar to lambertian
        const Vec3 normal_sphere_pos =
            add_vec3(hit_info.point, hit_info.surface_normal);
        const Vec3 reflect_point =
            add_vec3(normal_sphere_pos, rand_vec3_in_unit_sphere(rand_state));

        scatter_info.ray = (Ray)
        {
            .origin = hit_info.point,
            .direction = subtract_vec3(reflect_point, hit_info.point),
        },
        scatter_info.attenuation = (Vec3) { 0.1f, 0.2f, 0.2f },
        is_scattered = true;

/*
        switch (hit_info.material_type)
        {
        case MaterialTag.Lambertian:
            is_scattered = labertian_scatter(
                &scatter_info, ray, &hit_info, rand_state);
            break;
        case MaterialTag.Metal:
            is_scattered = metal_scatter(
                &scatter_info, ray, &hit_info, rand_state);
            break;
        case MaterialTag.Dielectric:
            is_scattered = dielectric_scatter(
                &scatter_info, ray, &hit_info, rand_state);
            break;
        };
*/
        if (is_scattered)
        {
            Vec3 color = calc_color(
                scatter_info.ray,
                scene,
                rand,
                call_depth + 1
            );
            return multiply_vec3(color, scatter_info.attenuation);
        }
        else
        {
            return (Vec3){.x = 0.0, .y = 0.0, .z = 0.0};
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
int main()
{
    const f32 num_samples_recip = 1.0f / NUM_SAMPLES;
    i32 ret;
    PxColor pixels[IMAGE_HEIGHT][IMAGE_WIDTH];

    // Random
    struct xorshift128p_state rand_state = create_rand_state();

    // Scene
    Scene scene;

    // Camera
    const Vec3 camera_pos  = { .x = 13.0f, .y = 2.0f, .z = 3.0f };
    const Vec3 look_at_pos = { .x = 0.0f,  .y = 0.0f, .z = 0.0f };
    const Vec3 v_up        = { .x = 0.0f,  .y = 1.0f, .z = 0.0f };
    const f32 vertical_fov = 120.0f;
    const f32 aspect_ratio = (f32)IMAGE_WIDTH / IMAGE_HEIGHT;
    const f32 aperture     = 0.1f;
    const f32 focus_dist   = 1.0f;  // TODO: Investigate: Changing this affects background gradient, because it effectively shrinks the vertical_fov. Is that expected from focus_dist?

    Camera camera = create_camera(
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

    printf("Reticulating splines...\n");
    struct timespec t0;
    ret = timespec_get(&t0, TIME_UTC);
    
    // Output Image
    for (i32 x = 0; x < IMAGE_WIDTH; ++x)
    {
        const f32 xRatio = x / (f32)IMAGE_WIDTH;
        for (i32 y = 0; y < IMAGE_HEIGHT; ++y)
        {
            const f32 yRatio = y / (f32)IMAGE_HEIGHT;
            Vec3 accumulated_color = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
            for (i32 sample = 0; sample < NUM_SAMPLES; ++sample)
            {
                const Ray ray        = generate_ray(camera, x, y, &rand_state);
                const Vec3 ray_color = calc_color(ray, &scene, &rand_state, 0);
                accumulated_color    = add_vec3(accumulated_color, ray_color);
            } // sample

            Vec3 color = scale_vec3(accumulated_color, num_samples_recip); // Average of samples
            color = sqrt_vec3(color);                                      // Gamma correction
            pixels[y][x] = (PxColor)
            {
                .red   = (u8)(color.x * UINT8_MAX),
                .green = (u8)(color.y * UINT8_MAX),
                .blue  = (u8)(color.z * UINT8_MAX),
                .alpha = UINT8_MAX,
            };
        } // y
    } // x

  // Save the image to a file
    ret = stbi_write_bmp(
        IMAGE_FILENAME,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        NUM_COLOR_COMPONENTS,
        &pixels
    );

    if (ret == 0)
    {
        printf("FAILED writing to file: {}\n");
    }
    else
    {
        printf("Successfuly written file: {}\n");
    }

    struct timespec t1;
    ret = timespec_get(&t1, TIME_UTC);
    double seconds_elapsed = (double)t1.tv_sec - t0.tv_sec;
    printf("Render Time: %.fs\n", seconds_elapsed);
    return 0;
}
