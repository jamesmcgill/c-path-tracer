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
// Ray
//------------------------------------------------------------------------------
typedef struct {
    Vec3 origin;
    Vec3 direction;
} Ray;

//------------------------------------------------------------------------------
// Scene 
//------------------------------------------------------------------------------
typedef struct {
    Vec3 position;
} Scene;

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
Ray generate_ray(const Camera camera, const i32 x, const i32 y)
{
    // Random starting position on lens aperture
    const Vec3 rd = { 1.0f, 1.0f, 1.0f }; // scale_vec3(rand.randomPointFromUnitDisc(), cam.lens_radius);
    const Vec3 aperture_offset_u = scale_vec3(camera.u, rd.x);
    const Vec3 aperture_offset_v = scale_vec3(camera.v, rd.y);
    const Vec3 aperture_offset = add_vec3(aperture_offset_u, aperture_offset_v);

    // The point that corresponds to on the frustum plane
    const Vec3 horiz = scale_vec3(camera.u, (x /*+ rand.float()*/) * camera.px_scale.x);
    const Vec3 vert  = scale_vec3(camera.v, (y /*+ rand.float()*/) * camera.px_scale.y);
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
Vec3 calc_color(const Ray ray, const Scene* scene)
{
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
                const Ray ray        = generate_ray(camera, x, y);
                const Vec3 ray_color = calc_color(ray, &scene);
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
