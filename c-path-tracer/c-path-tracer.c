#include <assert.h>
#include <stdio.h>   // printf
#include <stdint.h>  // uint32_t
#include <stdbool.h> // bool
#include <stdlib.h>
#include <string.h>
#include <time.h>   // timespec_get

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

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
    uint8_t red;
    uint8_t green;
    uint8_t blue;
    uint8_t alpha;
} color_t;

//------------------------------------------------------------------------------
int main()
{
    const float num_samples_recip = 1.0f / NUM_SAMPLES;
    int ret;

    printf("Reticulating splines...\n");
    struct timespec t0;
    ret = timespec_get(&t0, TIME_UTC);
    
    // Output Image
    color_t pixels[IMAGE_HEIGHT][IMAGE_WIDTH];
    for (int x = 0; x < IMAGE_WIDTH; ++x)
    {
        const float xRatio = x / (float)IMAGE_WIDTH;
        for (int y = 0; y < IMAGE_HEIGHT; ++y)
        {
            const float yRatio = y / (float)IMAGE_HEIGHT;

            pixels[y][x] = (color_t)
            {
                .red   = (uint8_t)(xRatio * UINT8_MAX),
                .green = (uint8_t)(yRatio * UINT8_MAX),
                .blue  = 0,
                .alpha = UINT8_MAX,
            };
        }
    }

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
