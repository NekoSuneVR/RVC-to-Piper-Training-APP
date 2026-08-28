# Make your first voice

## Get these files first

1. **A Piper base voice** — the first-launch installer downloads a British-English medium voice automatically. If you replace it later, download both files with the same name:
   - `voice-name.onnx`
   - `voice-name.onnx.json`
2. **Your RVC voice model**:
   - `voice-name.pth`
   - `voice-name.index` if the model includes one (recommended, but optional)

Official Piper voices are available at:

https://huggingface.co/rhasspy/piper-voices/tree/main

For British English, this is the relevant folder:

https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_GB

A medium-quality, neutral voice is a good first base because RVC will replace its voice character. Always check the model card and licence before using a voice.

## Set up the app

1. Extract the ZIP to a normal folder.
2. Double-click **Start RVC Piper Studio.cmd**. First launch downloads and installs all program components plus a default Piper voice. RVC is large, so this can take time.
3. In **1 Models & Setup**, the Piper fields are already filled. Browse to your RVC `.pth` and optional `.index` files.
5. Click **Save and check setup**. The app will tell you exactly which item is missing.

## Tune the RVC voice

1. Open **2 Tune & Create**.
2. Leave **Pitch detection** on `rmvpe`.
3. Start at pitch `0`, similarity `75%`, and sound protection `33%`.
4. Click **Generate pitch test** and listen.
5. Move pitch in small steps of 1–2 semitones and generate another test.
6. Compare the files in the output folder. Their filenames contain the pitch setting.

Good starting points—not fixed rules:

- Same general vocal range: `0`
- Deeper result: `-2` to `-6`
- Higher result: `+2` to `+6`
- Large voice-range change: try `-12` or `+12`, then adjust back toward zero

Pitch changes the musical/vocal range sent to RVC; it does not train or rewrite the model. Extreme values may sound artificial.

## Create your audio

Once the pitch test sounds right, type your own words and click **Generate my audio**. The result plays automatically and is saved under `generated`.
