#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <SDL2/SDL.h>

#include <ultra64.h>

#include "controller_api.h"
#include "controller_websocket.h"

static bool websocket_initialized = false;
static bool is_new_key_press = false;
static u32 last_keys_pressed = 0x0;

void input_key_press_from_websocket(u32 key_press) {
    is_new_key_press = true;
    last_keys_pressed = key_press;
}

static void websocket_controller_init(void) {
    if (!websocket_initialized) {
        websocket_initialized = true;
        websocket_thread_init();
        printf("Websocket controller initialized\n");
    }
}

static void websocket_controller_read(OSContPad *pad) {
    if (!websocket_initialized) {
        websocket_thread_init();
    }

    if (is_new_key_press) {
        pad->button |= last_keys_pressed;
        is_new_key_press = false;
        printf("Pressed a button! %d\n", last_keys_pressed);
    }
}

static void websocket_controller_shutdown(void) {
    websocket_initialized = false;
    websocket_server_shutdown();
}

static u32 websocket_controller_rawkey(void) {
    if (last_keys_pressed != 0x0) {
        return last_keys_pressed;
    }

    return VK_INVALID;
}

struct ControllerAPI controller_websocket = {
    VK_INVALID,
    websocket_controller_init,
    websocket_controller_read,
    websocket_controller_rawkey,
    NULL, // no rumble
    NULL, // no rumble stop
    NULL, // no key binds
    websocket_controller_shutdown
};