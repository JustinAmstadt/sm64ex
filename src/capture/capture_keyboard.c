#include <ultra64.h>
#include <stdio.h>

void capture_keyboard_input(FILE *file, u16 button, s8 stick_x, s8 stick_y, const char *framebuffer_path) {
    fprintf(file, "%04X,%d,%d,%s\n", button, stick_x, stick_y, framebuffer_path);
}