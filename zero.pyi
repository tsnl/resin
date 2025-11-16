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

    window_manager: WindowManager
    render_manager: RenderManager

class WindowManager:
    """
    Manager to create windows.
    """

    def create_window(
        self,
        title: str = "Untitled Zero App",
        width: int = 800,
        height: int = 600,
    ):
        """
        Creates a new window to read input and write rendered frames.

        :param title: The title of the window.
        :param width: The width of the window.
        :param height: The height of the window.
        """

    def update(self):
        """
        Updates all windows created by this manager.
        """

class RenderManager:
    """
    Manager for all things rendering.
    """
