class Instance:
    def __init__(
        self,
        require_window_support: bool = True,
        require_render_support: bool = True,
        debug_mode: bool = True,
    ) -> None:
        """
        Creates a new instance of the game engine.

        :param require_window_support: Whether window support is enabled. Required to create a window for rendering.
        :param require_render_support: Whether render support is enabled. Required to render graphics. Independent of window support.
        :param debug_mode: Whether to enable debug mode. Enables additional logging and debugging features, e.g. Vulkan validation layers.
        """

    def update(self):
        """
        Starts the game engine and runs the main loop.
        """
