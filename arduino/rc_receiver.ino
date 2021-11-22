byte PWM_PIN_SPEED = A0;
byte PWM_PIN_STEER = A1;
int speed_val;
int steer_val;

void setup() {
  pinMode(PWM_PIN_SPEED, INPUT);
  pinMode(PWM_PIN_STEER, INPUT);
  Serial.begin(9600);
}
 
void loop() {
  speed_val = pulseIn(PWM_PIN_SPEED, HIGH);
  steer_val = pulseIn(PWM_PIN_STEER, HIGH);
  Serial.write(speed_val / 256);
  Serial.write(speed_val % 256);
  Serial.write(steer_val / 256);
  Serial.write(steer_val % 256);
  Serial.write(0); 
}
