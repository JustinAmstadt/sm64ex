#ifndef CONTROLLER_WEBSOCKET_H
#define CONTROLLER_WEBSOCKET_H

#include <stdbool.h>

#include "controller_api.h"
#include "../websocket_server.h"

#define VK_INVALID 0xFFFF

extern struct ControllerAPI controller_websocket;
void input_key_press_from_websocket(u32 key_press);

#endif