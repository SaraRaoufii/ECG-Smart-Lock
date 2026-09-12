#include <Arduino.h>
#include <SPI.h>
#include "max30003.h"

// تعریف پین Chip Select (می‌توانید به دلخواه تغییر دهید)
const int CS_PIN = 5; 

MAX30003 max30003(CS_PIN);

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Initializing MAX30003...");

  // راه‌اندازی ارتباط SPI و تنظیمات اولیه سنسور
  SPI.begin(18, 19, 23, CS_PIN); // SCK, MISO, MOSI, CS
  max30003.max30003_sw_reset();
  delay(100);
  
  max30003.max30003_begin();
  Serial.println("MAX30003 Initialized successfully!");
}

void loop() {
  // خواندن داده‌های ECG از رجیسترهای ماژول
  uint32_t ecg_sample = max30003.getECGMeasurement();

  // چاپ داده در خروجی سریال (برای دیدن مستقیم روی Serial Plotter یا مانیتور)
  Serial.print("ECG:");
  Serial.println(ecg_sample);

  delay(10); // تنظیم سرعت نمونه‌برداری
}