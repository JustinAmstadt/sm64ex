#include <stdio.h>
#include <stdatomic.h>
#include <ultra64.h>

#include "websocket_server.h"
#include "controller/controller_websocket.h"

struct libwebsocket_context *context;
pthread_t thread;
atomic_bool running = 1;

struct lws_protocols protocols[] = {
    {
        "",
        callback_action,
        0,
        1024
    },
    { NULL, NULL, 0, 0 }
};


void websocket_server_context_init() {
    struct lws_context_creation_info info;
    memset(&info, 0, sizeof info);
    info.port = 8080;
    info.protocols = protocols;

    context = lws_create_context(&info);
}

void* websocket_server_thread(void* arg) {
    websocket_server_context_init();

    while (running) {
        lws_service(context, 50);  // 50ms timeout
    }

    lws_cancel_service(context);
    lws_context_destroy(context);
    context = NULL;
    return NULL;
}

int websocket_thread_init(void) {
    printf("Creating websocket thread\n");
    if (pthread_create(&thread, NULL, websocket_server_thread, NULL)) {
        fprintf(stderr, "Error creating thread\n");
        return 1;
    }

    return 0;
}

void websocket_server_shutdown(void) {
    printf("Websocket shutdown called\n");
    running = 0;
    if (context) {
        lws_cancel_service(context);
    }
    pthread_join(thread, NULL); // Block until thread finishes
}

static int callback_action(struct lws *wsi, enum lws_callback_reasons reason, void *user, void *in, size_t len) {
    switch (reason) {
        case LWS_CALLBACK_RECEIVE:
            // ONLY here are 'in' and 'len' valid
            printf("Received %zu bytes: %.*s\n", len, (int)len, (char*)in);
            u32 button = (u32)strtoul(in, NULL, 10);
            input_key_press_from_websocket(button);
            printf("Received button: 0x%04X (%u)\n", button, button);
            break;

        case LWS_CALLBACK_ESTABLISHED:
            // printf("Client connected\n");
            break;

        case LWS_CALLBACK_CLOSED:
            // printf("Client disconnected\n");
            break;

        case LWS_CALLBACK_SERVER_WRITEABLE:
            // Don't print - this happens frequently
            break;
        default:
            // printf("Callback reason: %d\n", reason);
            break;

    }
    return 0;
}