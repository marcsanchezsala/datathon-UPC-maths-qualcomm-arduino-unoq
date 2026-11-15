// Arduino UNO Q — LED Matrix Driver
// Runs on the STM32 microcontroller.
// Python sends a frame with Bridge.call("draw", text): 104 characters '0'..'7',
// one per LED (8 rows x 13 columns, row by row), 0 = off, 7 = brightest.
// A plain String is used because every App Lab version can pass it over the Bridge.

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <cstring>

Arduino_LED_Matrix matrix;

static const uint8_t FRAME_SIZE = 8 * 13;
static uint8_t frame[FRAME_SIZE] = {0};

void draw(String s) {
  for (size_t i = 0; i < FRAME_SIZE; i++) {
    char c = i < s.length() ? s[i] : '0';
    frame[i] = (c >= '0' && c <= '7') ? c - '0' : 0;
  }
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);   // 3-bit: 0-7 brightness
  matrix.clear();
  memset(frame, 0, FRAME_SIZE);

  Bridge.begin();
  Bridge.provide("draw", draw);
}

void loop() {
  matrix.draw(frame);
  delay(16);
}
