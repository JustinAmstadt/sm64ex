# include "frame_buffer_capture.h"

#include <stdio.h>
#include <stdint.h>

void capture_framebuffer(const char* filename, u16* fb, int width, int height) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to open file");
        return;
    }

    // Write PPM header (P3 format - ASCII)
    fprintf(fp, "P3\n%d %d\n255\n", width, height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            u16 pixel = fb[y * width + x];
            int r = (pixel >> 11) & 0x1F;  // 5 bits red
            int g = (pixel >> 6)  & 0x1F;  // 5 bits green
            int b = (pixel >> 1)  & 0x1F;  // 5 bits blue

            r = (r << 3) | (r >> 2); // expand 5 bits to 8 bits
            g = (g << 3) | (g >> 2);
            b = (b << 3) | (b >> 2);

            // Write pixel colors as decimal values (0-31)
            fprintf(fp, "%d %d %d\n", r, g, b);
        }
    }

    fclose(fp);
    printf("Framebuffer saved as %s\n", filename);
}