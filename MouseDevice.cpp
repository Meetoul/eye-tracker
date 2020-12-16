#include "MouseDevice.h"

#include <cstring>

#include <fcntl.h>
#include <unistd.h>

#include <linux/uinput.h>

bool MouseDevice::create() {
    struct uinput_setup usetup;
    mFd = open("/dev/uinput", O_WRONLY | O_NONBLOCK);

    if (mFd == -1) {
        return false;
    }

    /* enable mouse button left and relative events */
    ioctl(mFd, UI_SET_EVBIT, EV_KEY);
    ioctl(mFd, UI_SET_KEYBIT, BTN_LEFT);
    ioctl(mFd, UI_SET_KEYBIT, BTN_RIGHT);

    ioctl(mFd, UI_SET_EVBIT, EV_REL);
    ioctl(mFd, UI_SET_RELBIT, REL_X);
    ioctl(mFd, UI_SET_RELBIT, REL_Y);

    memset(&usetup, 0, sizeof(usetup));
    usetup.id.bustype = BUS_USB;
    usetup.id.vendor = 0x1111;
    usetup.id.product = 0x1111;
    strcpy(usetup.name, "Gaze input device");

    ioctl(mFd, UI_DEV_SETUP, &usetup);
    ioctl(mFd, UI_DEV_CREATE);

    return true;
}

void MouseDevice::destroy() {
    ioctl(mFd, UI_DEV_DESTROY);
    close(mFd);
}

void MouseDevice::clickLeft() { postEvent(EV_KEY, BTN_LEFT); }

void MouseDevice::clickRight() { postEvent(EV_KEY, BTN_RIGHT); }

void MouseDevice::move(int x, int y) {
    postEvent(EV_REL, REL_X, x);
    postEvent(EV_REL, REL_Y, y);
}

void MouseDevice::postEvent(int type, int code, int val) {
    postEventRaw(type, code, val);
    postEventRaw(EV_SYN, SYN_REPORT, 0);
}

void MouseDevice::postEventRaw(int type, int code, int val) {
    struct input_event ie;

    ie.type = type;
    ie.code = code;
    ie.value = val;
    ie.time.tv_sec = 0;
    ie.time.tv_usec = 0;

    write(mFd, &ie, sizeof(ie));
}
