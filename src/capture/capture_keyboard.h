#ifndef CAPTURE_H
#define CAPTURE_H

#include <stdio.h>

void capture_keyboard_input(FILE *file, u16 button, s8 stick_x, s8 stick_y, const char *framebuffer_path);

#endif