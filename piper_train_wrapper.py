from __future__ import annotations

"""Run Piper training without the optional MOS checkpoint callback.

Piper's current training CLI registers two ModelCheckpoint callbacks: one for
val_mel and one for the optional val_mos/UTMOS metric. On some Lightning
versions, if the optional MOS predictor cannot be loaded, val_mos is never
logged and ModelCheckpoint raises instead of skipping the callback. The Studio
only needs reliable val_mel/last checkpoints, so remove the optional callback
before delegating to Piper's normal CLI.
"""

from piper.train import __main__ as piper_train_main


def main() -> None:
    callbacks = list(getattr(piper_train_main, "_DEFAULT_CALLBACKS", []))
    piper_train_main._DEFAULT_CALLBACKS = [
        callback
        for callback in callbacks
        if getattr(callback, "monitor", None) != "val_mos"
    ]
    piper_train_main.main()


if __name__ == "__main__":
    main()
