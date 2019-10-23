/*
 * Carbon Dioxide Parts Per Million Meter
 * CO2PPM
 * 
 * learnelectronics
 * 26 MAR 2017
 * 
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


//---------------------------------------------------------------------------------------------------------------
//                                                   DEFINES
//---------------------------------------------------------------------------------------------------------------
#define anInput     A0                        //analog feed from MQ135
#define digTrigger   2                        //digital feed from MQ135
#define co2Zero     55                        //calibrated CO2 0 level
#define led          9                        //led on pin 9


//---------------------------------------------------------------------------------------------------------------
//                                                LIBRARY CALL
//---------------------------------------------------------------------------------------------------------------


//---------------------------------------------------------------------------------------------------------------
//                                                  SETUP
//---------------------------------------------------------------------------------------------------------------
void setup() {
  
  pinMode(anInput,INPUT);                     //MQ135 analog feed set for input
  pinMode(digTrigger,INPUT);                  //MQ135 digital feed set for input
  pinMode(led,OUTPUT);                        //led set for output
  Serial.begin(9600);                         //serial comms for debuging
  
}
//---------------------------------------------------------------------------------------------------------------
//                                               MAIN LOOP
//---------------------------------------------------------------------------------------------------------------
void loop() {
  
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


  Serial.print("El CO2 es de ->  ");
  Serial.print(co2ppm);
  Serial.println(" PPM");
  if(co2ppm>999){                             //if co2 ppm > 1000
    digitalWrite(led,HIGH);                   //turn on led
  }
  else{                                       //if not
    digitalWrite(led,LOW);                    //turn off led
  }
}
