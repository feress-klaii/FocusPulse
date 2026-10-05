/*
  FocusPulse — ESP32 Firmware
  Reads: motion (MPU6050), light (LDR), air quality (MQ2 — stand-in for MQ135), sound (sound sensor)
  Sends: JSON payload via HTTP POST every SAMPLE_INTERVAL_MS
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <ArduinoJson.h>

// ---- CONFIG ----
const char* WIFI_SSID     = "Wokwi-GUEST";
const char* WIFI_PASSWORD = "";
const char* SERVER_URL    = "http://host.wokwi.internal:8000/ingest";

const int PIN_LDR   = 34; // light sensor (analog)
const int PIN_MQ2   = 35; // air quality proxy (analog)
const int PIN_SOUND = 32; // sound level (analog)

const unsigned long SAMPLE_INTERVAL_MS = 5000;

Adafruit_MPU6050 mpu;
unsigned long lastSample = 0;

void sampleAndSend(); // forward declaration
void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println("\nConnected. IP: " + WiFi.localIP().toString());
}

void setup() {
  Serial.begin(115200);
  delay(500);

  connectWiFi();

  if (!mpu.begin()) {
    Serial.println("MPU6050 not found. Check wiring.");
    while (1) delay(10);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  pinMode(PIN_LDR, INPUT);
  pinMode(PIN_MQ2, INPUT);
  pinMode(PIN_SOUND, INPUT);

  Serial.println("FocusPulse firmware ready.");
}

void loop() {
  if (millis() - lastSample >= SAMPLE_INTERVAL_MS) {
    lastSample = millis();
    sampleAndSend();
  }
}

void sampleAndSend() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  float accelMag = sqrt(a.acceleration.x * a.acceleration.x +
                         a.acceleration.y * a.acceleration.y +
                         a.acceleration.z * a.acceleration.z);

  int lightRaw = analogRead(PIN_LDR);
  int airRaw   = analogRead(PIN_MQ2);
  int soundRaw = analogRead(PIN_SOUND);

  StaticJsonDocument<256> doc;
  doc["timestamp_ms"] = millis();
  doc["accel_x"] = a.acceleration.x;
  doc["accel_y"] = a.acceleration.y;
  doc["accel_z"] = a.acceleration.z;
  doc["accel_magnitude"] = accelMag;
  doc["gyro_x"] = g.gyro.x;
  doc["gyro_y"] = g.gyro.y;
  doc["gyro_z"] = g.gyro.z;
  doc["light_raw"] = lightRaw;
  doc["air_quality_raw"] = airRaw;
  doc["sound_raw"] = soundRaw;

  String payload;
  serializeJson(doc, payload);
  Serial.println("Payload: " + payload);

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");
    int code = http.POST(payload);
    Serial.println("POST response code: " + String(code));
    http.end();
  } else {
    Serial.println("WiFi disconnected, skipping send.");
  }
}