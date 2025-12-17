"""Universal Paperclips CLI - Phase 1 implementation."""

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
    engine.window.set_grid_config(
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
        parent_node=engine.window,
        row=0,
        col=0,
        col_span=3,
        text="Universal Paperclips",
        archetype="header",
    )

    # Row 1: Paperclip count display
    paperclips_label = zfw.GuiWidget(
        parent_node=engine.window,
        row=1,
        col=0,
        col_span=3,
        text=f"Paperclips: {game.paperclips:,}",
        archetype="label",
    )

    # Row 2: Make Paperclip button
    make_button = zfw.GuiWidget(
        parent_node=engine.window,
        row=2,
        col=0,
        col_span=3,
        text="Make Paperclip",
        archetype="button",
    )

    # Row 3: Business section (left column)
    business_container = zfw.GuiWidget(
        parent_node=engine.window,
        row=3,
        col=0,
        text="",
        archetype="label",
    )

    # Business header
    zfw.GuiWidget(
        parent_node=business_container,
        text="Business",
        archetype="label",
    )

    # Available funds
    funds_label = zfw.GuiWidget(
        parent_node=business_container,
        text=f"Available Funds: $ {game.funds:.2f}",
        archetype="label",
    )

    # Unsold inventory
    inventory_label = zfw.GuiWidget(
        parent_node=business_container,
        text=f"Unsold Inventory: {game.unsold_inventory}",
        archetype="label",
    )

    # Price controls
    price_container = zfw.GuiWidget(
        parent_node=business_container,
        text="",
        archetype="label",
    )

    lower_price_button = zfw.GuiWidget(
        parent_node=price_container,
        text="lower",
        archetype="button",
    )

    raise_price_button = zfw.GuiWidget(
        parent_node=price_container,
        text="raise",
        archetype="button",
    )

    price_label = zfw.GuiWidget(
        parent_node=price_container,
        text=f"Price per Clip: $ {game.price_per_clip:.2f}",
        archetype="label",
    )

    # Public demand
    demand_label = zfw.GuiWidget(
        parent_node=business_container,
        text=f"Public Demand: {game.public_demand:.0f}%",
        archetype="label",
    )

    # Marketing section
    marketing_container = zfw.GuiWidget(
        parent_node=business_container,
        text="",
        archetype="label",
    )

    marketing_button = zfw.GuiWidget(
        parent_node=marketing_container,
        text="Marketing",
        archetype="button",
    )

    marketing_label = zfw.GuiWidget(
        parent_node=marketing_container,
        text=f"Level: {game.marketing_level}",
        archetype="label",
    )

    marketing_cost_label = zfw.GuiWidget(
        parent_node=marketing_container,
        text=f"Cost: $ {game.marketing_cost:.2f}",
        archetype="label",
    )

    # Row 4: Manufacturing section (left column)
    manufacturing_container = zfw.GuiWidget(
        parent_node=engine.window,
        row=4,
        col=0,
        text="",
        archetype="label",
    )

    # Manufacturing header
    zfw.GuiWidget(
        parent_node=manufacturing_container,
        text="Manufacturing",
        archetype="label",
    )

    # Clips per second
    cps_label = zfw.GuiWidget(
        parent_node=manufacturing_container,
        text=f"Clips per Second: {game.clips_per_second}",
        archetype="label",
    )

    # Wire
    wire_container = zfw.GuiWidget(
        parent_node=manufacturing_container,
        text="",
        archetype="label",
    )

    wire_button = zfw.GuiWidget(
        parent_node=wire_container,
        text="Wire",
        archetype="button",
    )

    wire_label = zfw.GuiWidget(
        parent_node=wire_container,
        text=f"{game.wire_inches} inches",
        archetype="label",
    )

    wire_cost_label = zfw.GuiWidget(
        parent_node=wire_container,
        text=f"Cost: $ {game.wire_cost:.2f}",
        archetype="label",
    )

    # AutoClippers
    autoclipper_container = zfw.GuiWidget(
        parent_node=manufacturing_container,
        text="",
        archetype="label",
    )

    autoclipper_button = zfw.GuiWidget(
        parent_node=autoclipper_container,
        text="AutoClippers",
        archetype="button",
    )

    autoclipper_label = zfw.GuiWidget(
        parent_node=autoclipper_container,
        text=f"{game.auto_clippers}",
        archetype="label",
    )

    autoclipper_cost_label = zfw.GuiWidget(
        parent_node=autoclipper_container,
        text=f"Cost: $ {game.auto_clippers_cost:.2f}",
        archetype="label",
    )

    # Row 3-4: Computational Resources section (middle column)
    computational_container = zfw.GuiWidget(
        parent_node=engine.window,
        row=3,
        row_span=2,
        col=1,
        text="",
        archetype="label",
    )

    # Computational Resources header
    zfw.GuiWidget(
        parent_node=computational_container,
        text="Computational Resources",
        archetype="label",
    )

    # Trust
    trust_label = zfw.GuiWidget(
        parent_node=computational_container,
        text=f"Trust: {game.trust}",
        archetype="label",
    )

    # Processors
    processors_container = zfw.GuiWidget(
        parent_node=computational_container,
        text="",
        archetype="label",
    )

    processors_button = zfw.GuiWidget(
        parent_node=processors_container,
        text="Processors",
        archetype="button",
    )

    processors_label = zfw.GuiWidget(
        parent_node=processors_container,
        text=f"{game.processors}",
        archetype="label",
    )

    # Memory
    memory_container = zfw.GuiWidget(
        parent_node=computational_container,
        text="",
        archetype="label",
    )

    memory_button = zfw.GuiWidget(
        parent_node=memory_container,
        text="Memory",
        archetype="button",
    )

    memory_label = zfw.GuiWidget(
        parent_node=memory_container,
        text=f"{game.memory}",
        archetype="label",
    )

    # Operations
    operations_label = zfw.GuiWidget(
        parent_node=computational_container,
        text=f"Operations: {game.operations} / {game.max_operations:,}",
        archetype="label",
    )

    # Creativity
    creativity_label = zfw.GuiWidget(
        parent_node=computational_container,
        text=f"Creativity: {game.creativity}",
        archetype="label",
    )

    # Row 3-6: Projects section (right column)
    projects_container = zfw.GuiWidget(
        parent_node=engine.window,
        row=3,
        row_span=4,
        col=2,
        text="",
        archetype="label",
    )

    # Projects header
    zfw.GuiWidget(
        parent_node=projects_container,
        text="Projects",
        archetype="label",
    )

    # Project 1: Improved AutoClippers
    project1_button = zfw.GuiWidget(
        parent_node=projects_container,
        text="Improved AutoClippers (750 ops)\nIncreases AutoClipper performance 25%",
        archetype="button",
    )

    # Project 2: Improved Wire Extrusion
    project2_button = zfw.GuiWidget(
        parent_node=projects_container,
        text="Improved Wire Extrusion (1,750 ops)\n50% more wire supply from every spool",
        archetype="button",
    )

    # Project 3: RevTracker
    project3_button = zfw.GuiWidget(
        parent_node=projects_container,
        text="RevTracker (500 ops)\nAutomatically calculates average revenue\nper second",
        archetype="button",
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

    gpu_context.print_debug_info(out=file)
    print()
    print("</gpu-debug-info>")


if __name__ == "__main__":
    main()
