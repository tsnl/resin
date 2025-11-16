import zero


def main():
    instance = zero.Instance(require_render_support=False)

    window = instance.window_manager.create_window("Demo Window", 800, 600)
    window.set_visible(True)
    while not window.should_close:
        instance.window_manager.update()


if __name__ == "__main__":
    main()
