class MouseDevice {
public:
    bool create();
    void destroy();

    void clickLeft();
    void clickRight();

    void move(int x, int y);

private:
    void postEvent(int type, int code, int val = 0);
    void postEventRaw(int type, int code, int val);

    int mFd = -1;
};
