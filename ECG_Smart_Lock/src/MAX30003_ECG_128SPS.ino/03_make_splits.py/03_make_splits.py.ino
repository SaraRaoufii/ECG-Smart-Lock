#include <SPI.h>

#define CS_PIN 10

#define STATUS_REG   0x01
#define SW_RST       0x08
#define SYNCH        0x09
#define FIFO_RST     0x0A

#define CNFG_GEN     0x10
#define CNFG_EMUX    0x14
#define CNFG_ECG     0x15

#define ECG_FIFO     0x21


void writeRegister(byte reg, uint32_t data) {
  digitalWrite(CS_PIN, LOW);

  SPI.transfer(reg << 1);
  SPI.transfer((data >> 16) & 0xFF);
  SPI.transfer((data >> 8) & 0xFF);
  SPI.transfer(data & 0xFF);

  digitalWrite(CS_PIN, HIGH);
}


uint32_t readRegister(byte reg) {
  uint32_t data = 0;

  digitalWrite(CS_PIN, LOW);

  SPI.transfer((reg << 1) | 0x01);

  data |= ((uint32_t)SPI.transfer(0x00)) << 16;
  data |= ((uint32_t)SPI.transfer(0x00)) << 8;
  data |= SPI.transfer(0x00);

  digitalWrite(CS_PIN, HIGH);

  return data;
}


void setup() {

  Serial.begin(115200);

  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);

  SPI.begin();

  delay(500);

  SPI.beginTransaction(
    SPISettings(1000000, MSBFIRST, SPI_MODE0)
  );

  writeRegister(SW_RST, 0x000000);
  delay(100);

  writeRegister(CNFG_EMUX, 0x000000);

  // ECG = 128 SPS
  writeRegister(CNFG_ECG, 0x805000);

  writeRegister(CNFG_GEN, 0x000000);

  delay(20);

  writeRegister(SYNCH, 0x000000);
  delay(20);

  writeRegister(FIFO_RST, 0x000000);
  delay(20);

  // Enable ECG
  writeRegister(CNFG_GEN, 0x080000);

  delay(20);

  SPI.endTransaction();
}


void loop() {

  SPI.beginTransaction(
    SPISettings(1000000, MSBFIRST, SPI_MODE0)
  );

  uint32_t raw = readRegister(ECG_FIFO);

  SPI.endTransaction();

  int32_t ecg = (raw >> 6) & 0x3FFFF;

  // Sign extension برای 18-bit
  if (ecg & 0x20000) {
    ecg |= 0xFFFC0000;
  }

  Serial.println(ecg);

  delay(8);
}
