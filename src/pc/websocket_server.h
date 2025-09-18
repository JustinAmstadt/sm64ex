#ifndef WEBSOCKET_SERVER_H
#define WEBSOCKET_SERVER_H

#include <pthread.h>
#include <libwebsockets.h>
#include <stdbool.h>

int websocket_thread_init(void);
void websocket_server_shutdown(void);
static int callback_action(struct lws *wsi, enum lws_callback_reasons reason, void *user, void *in, size_t len);

#endif