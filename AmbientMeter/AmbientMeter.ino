/*
 * Carbon Dioxide Parts Per Million Meter
 * CO2PPM
 * 
 * learnelectronics
 * 26 MAR 2017
 * 
 * Reference: 
 * www.youtube.com/c/learnelectronics
 * arduino0169@gmail.com
 */

/*
 * Atmospheric CO2 Level..............400ppm
 * Average indoor co2.............350-450ppm
 * Maxiumum acceptable co2...........1000ppm
 * Dangerous co2 levels.............>2000ppm
 */






//---------------------------------------------------------------------------------------------------------------
//                                                  LIBRARIES
//---------------------------------------------------------------------------------------------------------------
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>


//---------------------------------------------------------------------------------------------------------------
//                                                   DEFINES
//---------------------------------------------------------------------------------------------------------------
#define anInput     A0                        //analog feed from MQ135
#define digTrigger   2                        //digital feed from MQ135
#define co2Zero     55                        //calibrated CO2 0 level
#define led          9                        //led on pin 9
#define BME_SCK 8                            //puertos del sensor
#define BME_MISO 7
#define BME_MOSI 6
#define BME_CS 5
#define SEALEVELPRESSURE_HPA (1013.25)        //pressure constante (sea level)


//---------------------------------------------------------------------------------------------------------------
//                                                LIBRARY CALL
//---------------------------------------------------------------------------------------------------------------

Adafruit_BME280 bme(BME_CS);

//---------------------------------------------------------------------------------------------------------------
//                                                  SETUP
//---------------------------------------------------------------------------------------------------------------

unsigned long delayTime;

void setup() {
  
  pinMode(anInput,INPUT);                     //MQ135 analog feed set for input
  pinMode(digTrigger,INPUT);                  //MQ135 digital feed set for input
  pinMode(led,OUTPUT);                        //led set for output
  Serial.begin(9600);                        //serial comms for debuging

  Serial.print("Initializing SD card...");

  // see if the card is present and can be initialized:
  if (!SD.begin(chipSelect)) {
    Serial.println("Card failed, or not present");
    // don't do anything more:
    while (1);
  }

  Serial.println("card initialized.");

  
while(!Serial);    // time to get serial running
    Serial.println(F("BME280 test"));

    unsigned status;
    
    // default settings
    // (you can also pass in a Wire library object like &Wire2)
    status = bme.begin();  
    if (!status) {
        Serial.println("Could not find a valid BME280 sensor, check wiring, address, sensor ID!");
        Serial.print("SensorID was: 0x"); Serial.println(bme.sensorID(),16);
        Serial.print("        ID of 0xFF probably means a bad address, a BMP 180 or BMP 085\n");
        Serial.print("   ID of 0x56-0x58 represents a BMP 280,\n");
        Serial.print("        ID of 0x60 represents a BME 280.\n");
        Serial.print("        ID of 0x61 represents a BME 680.\n");
        while (1);
    }
    
    Serial.println("-- Default Test --");
    delayTime = 1000;

    Serial.println();
  
}
//---------------------------------------------------------------------------------------------------------------
//                                               MAIN LOOP
//---------------------------------------------------------------------------------------------------------------
void loop() {
  String dataString = "";
  //---------------------------------------CO2 SENSOR--------------------------------------------------------
  int co2now[10];                               //int array for co2 readings
  int co2raw = 0;                               //int for raw value of co2
  int co2comp = 0;                              //int for compensated co2 
  int co2ppm = 0;                               //int for calculated ppm
  int zzz = 0;                                  //int for averaging
  
  
  for (int x = 0;x<10;x++){                   //samplpe co2 10x over 2 seconds
      co2now[x]=analogRead(A0);
      delay(10);
    }
  
  for (int x = 0;x<10;x++){                     //add samples together
      zzz=zzz + co2now[x];
    }
    
    co2raw = zzz/10;                            //divide samples by 10
    co2comp = co2raw - co2Zero;                 //get compensated value
    co2ppm = map(co2comp,0,1023,400,5000);      //map value for atmospheric levels
  
    dataString += co2ppm;
    dataString += ",";
    Serial.print("El CO2 es de ->  ");
    Serial.print(co2ppm);
    Serial.println(" PPM");
    if(co2ppm>999){                             //if co2 ppm > 1000
      digitalWrite(led,HIGH);                   //turn on led
    }
    
    else {                                       //if not
      digitalWrite(led,LOW);                    //turn off led
    }

    
   //-----------------------------------------TEMP MEASURE----------------------------------------------
    
    Serial.print("Temperature = ");
    float temp = bme.readTemperature();
    dataString += String(temp);
    dataString += ",";
    Serial.print(temp);
    Serial.println(" *C");
    

   //-----------------------------------------PRESSURE MEASURE----------------------------------------------

    Serial.print("Pressure = ");
    float pressure = bme.readPressure() / 100.0F;
    dataString += pressure;
    dataString += ",";
    Serial.print(pressure);
    Serial.println(" hPa");


   //-----------------------------------------ALTITUDE MEASURE----------------------------------------------
    Serial.print("Approx. Altitude = ");
    float altitude = bme.readAltitude(SEALEVELPRESSURE_HPA);
    dataString += altitude;
    dataString += ",";
    Serial.print(aktitude);
    Serial.println(" m");

   //-----------------------------------------HUMIDITY MEASURE----------------------------------------------

    Serial.print("Humidity = ");
    float humidity = bme.readHumidity();
    dataString += humidity;
    dataString += ",";
    Serial.print(humidity);
    Serial.println(" %");


    //---------------------------------------STORE THE DATA -------------------------------------------------

    File dataFile = SD.open("dataset.txt", FILE_WRITE);

    if(dataFile){
      dataFile.println(dataString);
      dataFile.close();

      Serial.println(dataString);
      
    }

    else{
      Serial.println("Error opening dataset.txt");
    }
    

}
