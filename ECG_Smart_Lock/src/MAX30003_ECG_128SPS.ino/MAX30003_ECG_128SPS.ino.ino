#include <SPI.h>

#define CS_PIN 10

// =====================================================
// MAX30003 Registers
// =====================================================

#define STATUS_REG   0x01
#define MNGR_INT     0x04

#define SW_RST       0x08
#define SYNCH        0x09
#define FIFO_RST     0x0A

#define CNFG_GEN     0x10
#define CNFG_EMUX    0x14
#define CNFG_ECG     0x15

#define ECG_FIFO     0x21


// =====================================================
// SPI Write - 24 bit register
// =====================================================

void writeRegister(byte reg, uint32_t data)
{
  digitalWrite(CS_PIN, LOW);

  SPI.transfer(reg << 1);

  SPI.transfer((data >> 16) & 0xFF);
  SPI.transfer((data >> 8) & 0xFF);
  SPI.transfer(data & 0xFF);

  digitalWrite(CS_PIN, HIGH);
}


// =====================================================
// SPI Read - 24 bit register
// =====================================================

uint32_t readRegister(byte reg)
{
  uint32_t data = 0;

  digitalWrite(CS_PIN, LOW);

  SPI.transfer((reg << 1) | 0x01);

  data |= ((uint32_t)SPI.transfer(0x00)) << 16;
  data |= ((uint32_t)SPI.transfer(0x00)) << 8;
  data |= SPI.transfer(0x00);

  digitalWrite(CS_PIN, HIGH);

  return data;
}


// =====================================================
// MAX30003 Initialization
// =====================================================

void setup()
{
  Serial.begin(115200);

  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);

  SPI.begin();

  delay(500);

  SPI.beginTransaction(
    SPISettings(
      1000000,
      MSBFIRST,
      SPI_MODE0
    )
  );


  // ---------------------------------------------------
  // 1. Software Reset
  // ---------------------------------------------------

  writeRegister(SW_RST, 0x000000);

  delay(100);


  // ---------------------------------------------------
  // 2. ECG input configuration
  //
  // OPENP = 0
  // OPENN = 0
  //
  // ECG inputs connected
  // ---------------------------------------------------

  writeRegister(CNFG_EMUX, 0x000000);

  delay(10);


  // ---------------------------------------------------
  // 3. ECG configuration
  //
  // 128 Samples Per Second
  // Digital High Pass Filter enabled
  // Digital Low Pass Filter enabled
  // ---------------------------------------------------

  writeRegister(CNFG_ECG, 0x805000);

  delay(10);


  // ---------------------------------------------------
  // 4. General configuration
  //
  // ECG disabled while preparing
  // ---------------------------------------------------

  writeRegister(CNFG_GEN, 0x000000);

  delay(20);


  // ---------------------------------------------------
  // 5. FIFO interrupt threshold
  //
  // EFIT = 00000
  //
  // EINT becomes active when at least
  // ONE ECG record is available.
  // ---------------------------------------------------

  writeRegister(MNGR_INT, 0x000000);

  delay(10);


  // ---------------------------------------------------
  // 6. Synchronize ECG timing
  // ---------------------------------------------------

  writeRegister(SYNCH, 0x000000);

  delay(20);


  // ---------------------------------------------------
  // 7. Reset ECG FIFO
  // ---------------------------------------------------

  writeRegister(FIFO_RST, 0x000000);

  delay(20);


  // ---------------------------------------------------
  // 8. Enable ECG
  // ---------------------------------------------------

  writeRegister(CNFG_GEN, 0x080000);

  delay(50);


  SPI.endTransaction();
}


// =====================================================
// Main Loop
// =====================================================

void loop()
{
  SPI.beginTransaction(
    SPISettings(
      1000000,
      MSBFIRST,
      SPI_MODE0
    )
  );


  // ---------------------------------------------------
  // Read STATUS
  // ---------------------------------------------------

  uint32_t status = readRegister(STATUS_REG);


  // ---------------------------------------------------
  // D23 = EINT
  //
  // EINT = ECG FIFO has enough unread records
  // ---------------------------------------------------

  bool fifoReady = (status & 0x800000UL) != 0;


  // ---------------------------------------------------
  // D22 = EOVF
  //
  // FIFO overflow
  // ---------------------------------------------------

  bool fifoOverflow = (status & 0x400000UL) != 0;


  // ---------------------------------------------------
  // If FIFO overflow occurred:
  // reset FIFO and resynchronize
  // ---------------------------------------------------

  if (fifoOverflow)
  {
    writeRegister(FIFO_RST, 0x000000);

    delay(5);

    writeRegister(SYNCH, 0x000000);

    delay(5);
  }


  // ---------------------------------------------------
  // Read FIFO ONLY when data exists
  // ---------------------------------------------------

  if (fifoReady && !fifoOverflow)
  {
    uint32_t raw = readRegister(ECG_FIFO);


    // -------------------------------------------------
    // ETAG = bits [5:3]
    // -------------------------------------------------

    uint8_t etag = (raw >> 3) & 0x07;


    // -------------------------------------------------
    // ETAG = 000
    // Valid Sample
    //
    // ETAG = 010
    // Valid Sample + End Of FIFO
    // -------------------------------------------------

    if (etag == 0 || etag == 2)
    {
      // -----------------------------------------------
      // ECG = bits [23:6]
      // 18-bit signed value
      // -----------------------------------------------

      int32_t ecg = (raw >> 6) & 0x3FFFF;


      // -----------------------------------------------
      // Sign extension
      // -----------------------------------------------

      if (ecg & 0x20000)
      {
        ecg |= 0xFFFC0000;
      }


      // -----------------------------------------------
      // Send ONLY the ECG value
      // -----------------------------------------------

      Serial.println(ecg);
    }
  }


  SPI.endTransaction();
}
