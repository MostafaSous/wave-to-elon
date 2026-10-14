// Wave to Elon - Arduino Uno side.
// Waits for the letter 'W' over USB serial, then waves the servo back and forth.
//
// Wiring (SG90 / MG90S style servo):
//   Brown/Black  -> GND
//   Red          -> 5V
//   Orange/Yellow (signal) -> pin 9

#include <Servo.h>

const int SERVO_PIN = 9;
const int REST_ANGLE = 90;   // hand straight up
const int LEFT_ANGLE = 50;
const int RIGHT_ANGLE = 130;
const int WAVES = 3;         // how many back-and-forths
const int STEP_DELAY = 6;    // ms per degree; lower = faster wave

Servo hand;
int current = REST_ANGLE;

void moveTo(int target) {
  int step = target > current ? 1 : -1;
  while (current != target) {
    current += step;
    hand.write(current);
    delay(STEP_DELAY);
  }
}

void wave() {
  for (int i = 0; i < WAVES; i++) {
    moveTo(LEFT_ANGLE);
    moveTo(RIGHT_ANGLE);
  }
  moveTo(REST_ANGLE);
}

void setup() {
  Serial.begin(9600);
  hand.attach(SERVO_PIN);
  hand.write(REST_ANGLE);
  Serial.println("READY");
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'W') {
      wave();
      while (Serial.available()) Serial.read();  // ignore waves sent while we were busy
      Serial.println("DONE");
    }
  }
}
