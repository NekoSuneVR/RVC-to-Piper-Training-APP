from __future__ import annotations

import argparse

from piper_builder_gui import PiperBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Piper model builder and optionally start one build step.")
    parser.add_argument(
        "--action",
        choices=("open", "install", "dataset", "train", "export", "all"),
        default="open",
        help="Builder action to run after the window opens.",
    )
    args = parser.parse_args()

    app = PiperBuilder()
    actions = {
        "install": app.install_trainer,
        "dataset": app.build_dataset,
        "train": app.train,
        "export": app.export,
        "all": app.build_all,
    }
    action = actions.get(args.action)
    if action is not None:
        # Give Tk a moment to finish drawing the builder before a long task starts.
        app.after(350, action)
    app.mainloop()


if __name__ == "__main__":
    main()
