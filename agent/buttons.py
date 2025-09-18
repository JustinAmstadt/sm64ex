from enum import IntFlag

class ControllerButton(IntFlag):
    CONT_A       = 0x8000
    CONT_B       = 0x4000
    CONT_G       = 0x2000
    CONT_START   = 0x1000
    CONT_UP      = 0x0800
    CONT_DOWN    = 0x0400
    CONT_LEFT    = 0x0200
    CONT_RIGHT   = 0x0100
    CONT_L       = 0x0020
    CONT_R       = 0x0010
    CONT_E       = 0x0008
    CONT_D       = 0x0004
    CONT_C       = 0x0002
    CONT_F       = 0x0001

    # Aliases using Nintendo-style button names
    A_BUTTON     = CONT_A
    B_BUTTON     = CONT_B
    Z_TRIG       = CONT_G
    START_BUTTON = CONT_START
    U_JPAD       = CONT_UP
    D_JPAD       = CONT_DOWN
    L_JPAD       = CONT_LEFT
    R_JPAD       = CONT_RIGHT
    L_TRIG       = CONT_L
    R_TRIG       = CONT_R
    U_CBUTTONS   = CONT_E
    D_CBUTTONS   = CONT_D
    L_CBUTTONS   = CONT_C
    R_CBUTTONS   = CONT_F