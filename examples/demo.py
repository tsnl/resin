import zero


def main():
    gpu_context = zero.GpuContext(enable_present_support=False)

    # window = zero.Window(width=800, height=600, title="Zero Demo Window")

    # window.show()
    # while not window.should_close():
    #     zero.Window.update_all()


if __name__ == "__main__":
    main()
