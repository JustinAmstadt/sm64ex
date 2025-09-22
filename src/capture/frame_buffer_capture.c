#include "frame_buffer_capture.h"
#include "../pc/gfx/gfx_pc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef RAPI_GL
#include <GL/gl.h>
#endif
#ifdef RAPI_GL_LEGACY
#include <GL/gl.h>
#endif

void capture_opengl_framebuffer(const char* filename) {
#if defined(RAPI_GL) || defined(RAPI_GL_LEGACY)
    // Copy dimensions to local variables to avoid any potential issues
    const int width = gfx_current_dimensions.width;
    const int height = gfx_current_dimensions.height;

    if (width <= 0 || height <= 0 || width > 4096 || height > 4096) {
        printf("Invalid dimensions\n");
        return;
    }

    const size_t row_size = (size_t)width * 3;
    const size_t buffer_size = row_size * (size_t)height;

    unsigned char* pixels = malloc(buffer_size);
    if (!pixels) {
        printf("Failed to allocate %zu bytes\n", buffer_size);
        return;
    }

    glFinish();  // Ensure rendering is complete

    // Set pack alignment to 1 byte to avoid padding issues
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

    // Write raw binary file
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Failed to open framebuffer file: %s\n", filename);
        free(pixels);
        return;
    }

    // Write raw framebuffer data as-is (bottom-to-top from OpenGL)
    fwrite(pixels, 1, buffer_size, fp);
    fclose(fp);
    free(pixels);
#else
printf("ERROR: Output capture for anything other than OpenGL is not supported");
#endif
}