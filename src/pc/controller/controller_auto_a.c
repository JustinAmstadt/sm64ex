#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <SDL2/SDL.h>

#include <ultra64.h>

#include "controller_api.h"
#include "controller_auto_a.h"

static bool auto_a_initialized = false;
static Uint32 last_press_time = 0;
static bool a_pressed = false;

static void auto_a_init(void) {
    if (!auto_a_initialized) {
        auto_a_initialized = true;
        last_press_time = SDL_GetTicks();
        printf("Auto A controller initialized - will press A every 0.2 seconds\n");
    }
}

static void auto_a_read(OSContPad *pad) {
    if (!auto_a_initialized) {
        auto_a_init();
    }

    Uint32 current_time = SDL_GetTicks();
    Uint32 time_diff = current_time - last_press_time;

    // Every 200ms (0.2 seconds), toggle A press
    if (time_diff >= 200) {
        a_pressed = !a_pressed;
        last_press_time = current_time;

        if (a_pressed) {
            pad->button |= A_BUTTON;
        }
    } else if (a_pressed && time_diff >= 100) {
        // Release A button after 100ms (halfway through the cycle)
        a_pressed = false;
    }

    // Apply A button state
    if (a_pressed) {
        pad->button |= A_BUTTON;
    }
}

static void auto_a_shutdown(void) {
    auto_a_initialized = false;
}

static u32 auto_a_rawkey(void) {
    return VK_INVALID;
}

struct ControllerAPI controller_auto_a = {
    VK_INVALID,
    auto_a_init,
    auto_a_read,
    auto_a_rawkey,
    NULL, // no rumble
    NULL, // no rumble stop
    NULL, // no key binds
    auto_a_shutdown
};