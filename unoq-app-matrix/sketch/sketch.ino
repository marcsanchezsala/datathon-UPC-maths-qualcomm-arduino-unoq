// Arduino UNO Q — LED Matrix Driver
// Runs on the STM32 microcontroller.
// Receives frame data from Python via Bridge.call("draw", ...)
// and continuously draws it on the 8×13 LED matrix.

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <vector>
#include <cstring>

Arduino_LED_Matrix matrix;

// Frame buffer: 8 rows × 13 columns
// Each cell: 0 = off, 1-7 = brightness (3-bit grayscale)
static const uint8_t FRAME_SIZE = 8 * 13;
static uint8_t frame[FRAME_SIZE] = {0};

// Receive frame data from Python and store in buffer
void draw(std::vector<uint8_t> newFrame) {
  size_t copyLen = std::min(newFrame.size(), (size_t)FRAME_SIZE);
  memcpy(frame, newFrame.data(), copyLen);
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);   // 3-bit: 0-7 brightness
  matrix.clear();
  memset(frame, 0, FRAME_SIZE);

  Bridge.begin();
  Bridge.provide("draw", draw);  // Python calls Bridge.call("draw", ...)
}

void loop() {
  matrix.draw(frame);
  delay(16);  // ~60 FPS refresh
}