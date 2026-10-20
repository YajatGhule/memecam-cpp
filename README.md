# Meme Cam (C++)

A C++20 port of [Meme Cam](https://github.com/YajatGhule/memecam): your webcam with motion-capture style markers (glowing face / hand /
body landmarks, corner-bracket boxes, HUD) that pops up a meme while you make the matching gesture
(thumbs up, nerd finger, Absolute Cinema, 67, ...). The Python app stays the reference implementation; this
fork matches its gestures and rendering, but is built for **one person**: it tracks whoever is closest to the
camera (two hands, one face, one body), which keeps every model in its fast tracking mode.

Two executables are built:

- `memecam`   - the original 11 gestures (`config/config.toml`)
- `memecamv2` - the same plus the v2 gestures (Godfather, looksmaxxing; `config/config_v2.toml` is layered on top)

## Windows

Download `MemeCam-cpp-Windows.zip` from [Releases](https://github.com/YajatGhule/memecam-cpp/releases/latest),
unzip, and double-click `memecam.exe` (or `memecamv2.exe`). No install needed. If Windows shows "Windows
protected your PC", click **More info** → **Run anyway**.

## Install (Linux)

```sh
curl -fsSL https://raw.githubusercontent.com/YajatGhule/memecam-cpp/main/install.sh | bash
```

It installs the build tools and OpenCV with your package manager (pacman / apt / dnf, asks for sudo), clones
the repo to `~/.local/share/memecam-cpp`, downloads LiteRT and the MediaPipe models, builds, and adds
`memecam-cpp` / `memecamv2-cpp` launchers (in `~/.local/bin`) plus app-menu entries. The first build compiles
LiteRT from source: ~20-40 minutes and ~2 GB of disk. Re-run the same line to update. Tested on Arch;
Debian / Ubuntu need 24.04 or newer (CMake >= 3.28). macOS / Windows: use the
[Python version](https://github.com/YajatGhule/memecam), which has one-line installers for both.

## Build

Requirements: CMake >= 3.28, a C++20 compiler (GCC 16 tested), OpenCV 5 (with highgui, videoio, and
optionally freetype for the bold captions), unzip. Everything else is fetched by CMake (toml++, CLI11,
nlohmann/json, Catch2) or built from source (LiteRT).

```sh
tools/build_litert.sh                 # clone the LiteRT (TensorFlow Lite C++) sources into third_party/tensorflow
tools/extract_models.sh [DIR]         # extract the .tflite models from MediaPipe's .task bundles into models/
                                      # (downloads the .task files when DIR doesn't have them)
cmake -B build -G Ninja               # or the default generator
cmake --build build -j6
ctest --test-dir build                # logic + parity tests (need no LiteRT)
```

LiteRT is compiled as part of the main build (first build takes a while, cached afterwards). Without it you
can still build the rendering, logic and tests: `cmake -B build-core -DMEMECAM_WITH_VISION=OFF`.

`tools/extract_models.sh` reads `gesture_recognizer.task`, `face_landmarker.task` and
`pose_landmarker_lite.task` from `~/Projects/meme-cam/models` (or the directory you pass; downloaded from
Google's MediaPipe model storage when missing) and unpacks the palm / hand-landmark / gesture / face / pose
`.tflite` files into `models/`.

## Run

```sh
build/memecam                       # webcam, window "Meme Cam"
build/memecamv2
build/memecam --video clip.webm --snapshot out.png --frames 30 --force nerd   # headless render test
build/memecam --video clip.webm --realtime        # play a clip through the live path
build/memecam --bench 10            # headless: display FPS vs tracking FPS
build/memecam --video clip.webm --events          # print every meme switching on/off in a clip
```

Flags: `--camera N`, `--no-pose` (faster), `--config FILE`, `--video FILE`, `--realtime`, `--no-mirror`,
`--snapshot FILE --frames N`, `--force MEME` (repeatable), `--bench SECONDS`, `--events`, `--verbose` (show LiteRT /
OpenCV / Qt logs).

Keys: `d` markers and boxes, `f` meme overlays, `m` mute sounds, `i` gesture debug scores, `r` record to
`~/Videos/memecam-*.mp4`, `c` recalibrate head pose, `q` / `Esc` quit.

Settings live in `config/config.toml`: camera, tracking device, `[person]` (minimum face width, so people in
the background are ignored; dropout bridging time), `[display] meme_on_face` (put every meme over your face), the gesture
thresholds, and one `[memes.*]` table per meme (asset, title, placement, height, hold, trim `frames`,
`loop_from`, ...).

## How it works

- `src/vision/` - MediaPipe-equivalent hand, face and pose pipelines (detector -> landmark model -> ROI
  tracking) on LiteRT, using the same model files; behind the `Tracker` interface (`vision/tracker.h`).
  The detectors only run when tracking is lost; every 15 (face) / 30 (body) frames the detector also checks
  for a clearly bigger face/body than the tracked one and switches to it, so it never sticks to someone in
  the background.
- `src/tracking.*` - attaches hands and body to the face (and drops background faces), motion map.
- `src/gestures.*` - all gesture detectors and the `Trigger` state machine (1:1 with `gestures.py`). A meme
  shows only while its gesture is active (hold 0.3 s to start, 0.3 s linger after).
- `src/crowd.*` - the tracked `Person`'s detectors and meme triggers (kept through short tracking dropouts).
- `src/overlays.*` - GIF/PNG `Media` (trim, `loop_from` rocking loop), alpha blend, fit/panel caches,
  `over_face`, `warp_on_face`, the black-and-white Cinema effect.
- `src/mocap.*` - glowing markers, brackets, labels, HUD.
- `src/compose.*` - draws the active memes and markers onto a frame.
- `src/app.cpp` - a grabber thread (newest frame only), a tracking thread, and the display loop that shows
  every camera frame with the latest tracking result; recording and the key handling.

Tracking check against the Python app: `tools/dump_mediapipe.py` (run with the Python repo's environment)
dumps MediaPipe's landmarks for a clip; `build/vision_parity dump.json...` runs the C++ tracker on the same
frames. On 15 recorded clips (~3000 frames): face mesh within 0.03 % of the face width (median), hands
within 0.5 % of the hand size (median, 99 % handedness / gesture-label agreement), body within ~4 % of the
shoulder width (MediaPipe's smoothing amplifies tiny differences on the off-screen legs). End to end,
`--events` on each of the 20 recorded clips fires exactly the clip's meme and nothing else (v1 and v2).

Rendering check against the Python app: `tools/dump_frame.py` (run in the Python repo) writes a rendered
frame plus its tracking state; `build/render_check <prefix>` redraws it with the C++ renderer and writes
`<prefix>_cmp.png` (python | c++ | diff x4) and the pixel difference. All memes match to within rounding
(only the Cinema caption differs by a few pixels from the font rasteriser).

## Meme sounds

Each meme's sound plays while the meme is on screen (from the start each time it switches on, a quick fade
when it switches off); `m` mutes, `--mute` starts muted, `[audio]` in the config sets the volume. The sounds
are `assets/sounds/<meme>.wav` (git-ignored: they are copyrighted clips); `tools/fetch_sounds.sh` downloads the
default set with yt-dlp + ffmpeg. Use your own by dropping a WAV there or setting `sound = "file.wav"` in a
`[memes.*]` table.

## Meme assets

The GIFs and images in `assets/` are copyrighted by their respective owners (films, anime, memes). They are
included for personal use only; do not redistribute them or publish them in a public repository without
checking the rights.
