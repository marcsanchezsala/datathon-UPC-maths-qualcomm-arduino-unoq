# Math Matrix Copilot — App Lab for Arduino UNO Q

Interactive copilot that shows **answers, confidence, thinking animations, and feedback** on the 8×13 LED matrix.

## What It Shows on the LED Matrix

| Feature | Description |
|---|---|
| **Letters A-E** | Answer choice displayed as 5×7 character glyph |
| **✓ Checkmark** | Shown for correct answers (green feedback) |
| **✕ Cross** | Shown for incorrect answers |
| **Confidence Bar** | 5 horizontal segments (filled/empty) |
| **Thinking Animation** | Pulsing `?` during model inference |
| **Mode Indicator** | Small label on the left: `RAG`, `T1`, `T2`, `ERR` |
| **Flash Effect** | Quick full-matrix flash when result is ready |
| **Status Text** | Short text like `HELLO!` scrolled across |
| **Heartbeat** | Idle pulse at bottom showing app is alive |

## Files

```
unoq-app-matrix/
├── app.yaml                          # App Lab config (name, bricks)
├── sketch/
│   ├── sketch.ino                    # STM32: receives frames via Bridge
│   └── sketch.yaml                   # Platform config (arduino:zephyr)
├── python/
│   └── main.py                       # Linux QRB2210: builds frames, API endpoints
├── assets/
│   └── index.html                    # Web UI (buttons, confidence slider, status)
└── README.md                         # This file
```

## How to Deploy on the Board

### Step 1: Push the app to the board
```bash
# Opció A: via SCP (Wi-Fi / LAN)
ssh arduino@192.168.1.85 "mkdir -p ~/ArduinoApps/math-matrix"
scp -r unoq-app-matrix/* arduino@192.168.1.85:~/ArduinoApps/math-matrix/

# Opció B: via ADB (cable USB)
adb shell "mkdir -p /home/arduino/ArduinoApps/math-matrix"
adb push unoq-app-matrix/. /home/arduino/ArduinoApps/math-matrix/
```

### Step 2: Start the app
```bash
ssh arduino@192.168.1.85 "arduino-app-cli app start user:math-matrix"
```

### Step 3: Open the web UI
```bash
# Over USB
adb forward tcp:7000 tcp:7000
# Then open http://127.0.0.1:7000 in your browser
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/answer/{letter}` | GET | Solve answer A-E (calls model) |
| `/feedback/{letter}/{true/false}` | GET | Show check or cross on matrix |
| `/confidence/{0-5}` | GET | Set confidence bar level |
| `/status` | GET | Get current app state |
| `/clear` | GET | Turn off all LEDs |
| `/demo/all` | GET | Run full demo of all features |

## Web UI Features

- **Answer buttons** A-E: calls the model and shows result
- **Manual feedback**: ✓/✕ with check/cross on LED matrix
- **Confidence slider**: 0-5, drives the LED confidence bar
- **Status card**: Shows busy/ready, mode (RAG/T1/T2/ERR), confidence
- **Clear & Demo buttons**: utilities for testing

## LED Animation Flow (solve answer)

```
1. Show "T1" mode indicator on left
2. Start thinking animation (pulsing ?)
3. (In background) call model server
4. Flash matrix on/off (2×)
5. Show answer letter (5×7 font, centered)
6. Show confidence bar (5 segments)
```

## Notes

- The model server URLs (`http://127.0.0.1:8081`) must match your deployed servers
- The heartbeat animation shows the app is alive when idle
- No internet access needed: everything runs locally on the board
- Flash rate max 3 Hz (board rule)