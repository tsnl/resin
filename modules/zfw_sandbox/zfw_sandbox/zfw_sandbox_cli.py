"""
Universal Paperclips CLI - Phase 1 implementation.

TODO: background image: https://unsplash.com/photos/aerial-view-of-green-trees-and-road-during-daytime-ZeDw8ck4XEM
"""

import argparse
import sys

import zfw

from .bundled_data import BUNDLED_DATA_PATH


KENNEY_SOKOBAN_DATA_PATH = (
    BUNDLED_DATA_PATH
    / "data/KenneyGameAssetsAllInOne-3_3_0/2D assets"
    / "Sokoban Pack"
    / "PNG/Default size"
)

FONT_SIZE_PX = 18


class PaperclipsGame:
    """Universal Paperclips game state and logic."""

    def __init__(self):
        # Business
        self.paperclips: int = 0
        self.funds: float = 0.00
        self.unsold_inventory: int = 0
        self.price_per_clip: float = 0.25
        self.public_demand: float = 100.0  # percentage

        # Marketing
        self.marketing_level: int = 1
        self.marketing_cost: float = 100.00

        # Manufacturing
        self.clips_per_second: int = 0
        self.wire_inches: int = 0
        self.wire_cost: float = 0.0
        self.auto_clippers: int = 0
        self.auto_clippers_cost: float = 0.0

        # Computational Resources
        self.trust: int = 0
        self.processors: int = 1
        self.memory: int = 1
        self.operations: int = 0
        self.max_operations: int = 1000
        self.creativity: int = 0

        # Projects
        self.available_projects: list[str] = []

    def make_paperclip(self):
        """Make a single paperclip manually."""
        raise NotImplementedError("make_paperclip not yet implemented")

    def lower_price(self):
        """Lower the price per clip."""
        raise NotImplementedError("lower_price not yet implemented")

    def raise_price(self):
        """Raise the price per clip."""
        raise NotImplementedError("raise_price not yet implemented")

    def buy_marketing(self):
        """Buy a marketing level."""
        raise NotImplementedError("buy_marketing not yet implemented")

    def buy_wire(self):
        """Buy wire."""
        raise NotImplementedError("buy_wire not yet implemented")

    def buy_auto_clipper(self):
        """Buy an AutoClipper."""
        raise NotImplementedError("buy_auto_clipper not yet implemented")

    def adjust_processors(self, delta: int):
        """Adjust processor count."""
        raise NotImplementedError("adjust_processors not yet implemented")

    def adjust_memory(self, delta: int):
        """Adjust memory count."""
        raise NotImplementedError("adjust_memory not yet implemented")


def main():
    ap = argparse.ArgumentParser(description="Universal Paperclips")
    ap.add_argument(
        "--swapchain-image-count",
        type=int,
        default=3,
        choices=[2, 3],
        help="Number of swapchain images: 2 for double buffering, 3 for triple buffering",
    )
    ap.add_argument("--debug", action="store_true", help="Enable debug settings")
    args = ap.parse_args()

    # Create game state
    game = PaperclipsGame()

    # Create engine
    engine = zfw.Engine(
        app_name="Zero Sandbox",
        debug=args.debug,
        swapchain_image_count=args.swapchain_image_count,
        enable_gui=True,
    )

    # Configure window grid: single column layout with multiple rows
    # Rows: Title, Paperclip Count, Make Button, Business Section, Manufacturing, Computational Resources, Projects
    engine.window.central_widget.set_grid_config(
        num_rows=10,
        num_cols=3,
        row_sizes=(
            80,
            60,
            40,
            150,
            120,
            200,
            -1,
            40,
            40,
            40,
        ),  # Fixed sizes with stretch for projects
        col_sizes=(
            -1,
            -1,
            -1,
        ),  # Three equal columns
    )

    # Row 0: Title
    zfw.GuiWidget(
        parent_widget=engine.window.central_widget,
        row=0,
        col=0,
        col_span=3,
        text="Universal Paperclips",
        style_classes=["label", "h1"],
    )

    # Row 1: Paperclip count display
    paperclips_label = zfw.GuiWidget(
        parent_widget=engine.window.central_widget,
        row=1,
        col=0,
        col_span=3,
        text=f"Paperclips: {game.paperclips:,}",
        style_classes=["label"],
    )

    # Row 2: Make Paperclip button
    make_button = zfw.GuiWidget(
        parent_widget=engine.window.central_widget,
        row=2,
        col=0,
        col_span=3,
        text="Make Paperclip",
        style_classes=["button"],
    )

    # Row 3: Business section (left column)
    business_container = zfw.GuiWidget(
        parent_widget=engine.window.central_widget,
        row=3,
        col=0,
        text="",
        style_classes=["label"],
    )

    # Business header
    zfw.GuiWidget(
        parent_widget=business_container,
        text="Business",
        style_classes=["label"],
    )

    # Available funds
    funds_label = zfw.GuiWidget(
        parent_widget=business_container,
        text=f"Available Funds: $ {game.funds:.2f}",
        style_classes=["label"],
    )

    # Unsold inventory
    inventory_label = zfw.GuiWidget(
        parent_widget=business_container,
        text=f"Unsold Inventory: {game.unsold_inventory}",
        style_classes=["label"],
    )

    # Price controls
    price_container = zfw.GuiWidget(
        parent_widget=business_container,
        text="",
        style_classes=["label"],
    )

    lower_price_button = zfw.GuiWidget(
        parent_widget=price_container,
        text="lower",
        style_classes=["button"],
    )

    raise_price_button = zfw.GuiWidget(
        parent_widget=price_container,
        text="raise",
        style_classes=["button"],
    )

    price_label = zfw.GuiWidget(
        parent_widget=price_container,
        text=f"Price per Clip: $ {game.price_per_clip:.2f}",
        style_classes=["label"],
    )

    # Public demand
    demand_label = zfw.GuiWidget(
        parent_widget=business_container,
        text=f"Public Demand: {game.public_demand:.0f}%",
        style_classes=["label"],
    )

    # Marketing section
    marketing_container = zfw.GuiWidget(
        parent_widget=business_container,
        text="",
        style_classes=["label"],
    )

    marketing_button = zfw.GuiWidget(
        parent_widget=marketing_container,
        text="Marketing",
        style_classes=["button"],
    )

    marketing_label = zfw.GuiWidget(
        parent_widget=marketing_container,
        text=f"Level: {game.marketing_level}",
        style_classes=["label"],
    )

    marketing_cost_label = zfw.GuiWidget(
        parent_widget=marketing_container,
        text=f"Cost: $ {game.marketing_cost:.2f}",
        style_classes=["label"],
    )

    # Row 4: Manufacturing section (left column)
    manufacturing_container = zfw.GuiWidget(
        parent_widget=engine.window.central_widget,
        row=4,
        col=0,
        text="",
        style_classes=["label"],
    )

    # Manufacturing header
    zfw.GuiWidget(
        parent_widget=manufacturing_container,
        text="Manufacturing",
        style_classes=["label"],
    )

    # Clips per second
    cps_label = zfw.GuiWidget(
        parent_widget=manufacturing_container,
        text=f"Clips per Second: {game.clips_per_second}",
        style_classes=["label"],
    )

    # Wire
    wire_container = zfw.GuiWidget(
        parent_widget=manufacturing_container,
        text="",
        style_classes=["label"],
    )

    wire_button = zfw.GuiWidget(
        parent_widget=wire_container,
        text="Wire",
        style_classes=["button"],
    )

    wire_label = zfw.GuiWidget(
        parent_widget=wire_container,
        text=f"{game.wire_inches} inches",
        style_classes=["label"],
    )

    wire_cost_label = zfw.GuiWidget(
        parent_widget=wire_container,
        text=f"Cost: $ {game.wire_cost:.2f}",
        style_classes=["label"],
    )

    # AutoClippers
    autoclipper_container = zfw.GuiWidget(
        parent_widget=manufacturing_container,
        text="",
        style_classes=["label"],
    )

    autoclipper_button = zfw.GuiWidget(
        parent_widget=autoclipper_container,
        text="AutoClippers",
        style_classes=["button"],
    )

    autoclipper_label = zfw.GuiWidget(
        parent_widget=autoclipper_container,
        text=f"{game.auto_clippers}",
        style_classes=["label"],
    )

    autoclipper_cost_label = zfw.GuiWidget(
        parent_widget=autoclipper_container,
        text=f"Cost: $ {game.auto_clippers_cost:.2f}",
        style_classes=["label"],
    )

    # Row 3-4: Computational Resources section (middle column)
    computational_container = zfw.GuiWidget(
        parent_widget=engine.window.central_widget,
        row=3,
        row_span=2,
        col=1,
        text="",
        style_classes=["label"],
    )

    # Computational Resources header
    zfw.GuiWidget(
        parent_widget=computational_container,
        text="Computational Resources",
        style_classes=["label"],
    )

    # Trust
    trust_label = zfw.GuiWidget(
        parent_widget=computational_container,
        text=f"Trust: {game.trust}",
        style_classes=["label"],
    )

    # Processors
    processors_container = zfw.GuiWidget(
        parent_widget=computational_container,
        text="",
        style_classes=["label"],
    )

    processors_button = zfw.GuiWidget(
        parent_widget=processors_container,
        text="Processors",
        style_classes=["button"],
    )

    processors_label = zfw.GuiWidget(
        parent_widget=processors_container,
        text=f"{game.processors}",
        style_classes=["label"],
    )

    # Memory
    memory_container = zfw.GuiWidget(
        parent_widget=computational_container,
        text="",
        style_classes=["label"],
    )

    memory_button = zfw.GuiWidget(
        parent_widget=memory_container,
        text="Memory",
        style_classes=["button"],
    )

    memory_label = zfw.GuiWidget(
        parent_widget=memory_container,
        text=f"{game.memory}",
        style_classes=["label"],
    )

    # Operations
    operations_label = zfw.GuiWidget(
        parent_widget=computational_container,
        text=f"Operations: {game.operations} / {game.max_operations:,}",
        style_classes=["label"],
    )

    # Creativity
    creativity_label = zfw.GuiWidget(
        parent_widget=computational_container,
        text=f"Creativity: {game.creativity}",
        style_classes=["label"],
    )

    # Row 3-6: Projects section (right column)
    projects_container = zfw.GuiWidget(
        parent_widget=engine.window.central_widget,
        row=3,
        row_span=4,
        col=2,
        text="",
        style_classes=["label"],
    )

    # Projects header
    zfw.GuiWidget(
        parent_widget=projects_container,
        text="Projects",
        style_classes=["label", "h2"],
    )

    # Project 1: Improved AutoClippers
    project1_button = zfw.GuiWidget(
        parent_widget=projects_container,
        text="Improved AutoClippers (750 ops)\nIncreases AutoClipper performance 25%",
        style_classes=["button"],
    )

    # Project 2: Improved Wire Extrusion
    project2_button = zfw.GuiWidget(
        parent_widget=projects_container,
        text="Improved Wire Extrusion (1,750 ops)\n50% more wire supply from every spool",
        style_classes=["button"],
    )

    # Project 3: RevTracker
    project3_button = zfw.GuiWidget(
        parent_widget=projects_container,
        text="RevTracker (500 ops)\nAutomatically calculates average revenue\nper second",
        style_classes=["button"],
    )

    # Placeholder refs to avoid unused variable warnings
    _ = (
        paperclips_label,
        make_button,
        funds_label,
        inventory_label,
        lower_price_button,
        raise_price_button,
        price_label,
        demand_label,
        marketing_button,
        marketing_label,
        marketing_cost_label,
        cps_label,
        wire_button,
        wire_label,
        wire_cost_label,
        autoclipper_button,
        autoclipper_label,
        autoclipper_cost_label,
        trust_label,
        processors_button,
        processors_label,
        memory_button,
        memory_label,
        operations_label,
        creativity_label,
        project1_button,
        project2_button,
        project3_button,
    )

    # Main game loop
    while not engine.window.should_close():
        engine.update()
        engine.render()


def print_gpu_debug_info(
    gpu_context: zfw.GpuContext,
    file: zfw.SupportsWrite[str] = sys.stdout,
):
    print("<gpu-debug-info>")
    gpu_context.print_debug_info(out=file)
    print()
    print("</gpu-debug-info>")


if __name__ == "__main__":
    main()
