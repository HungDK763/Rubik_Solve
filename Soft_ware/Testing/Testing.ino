
#include <Arduino.h>
#include <LiquidCrystal.h>

/* Speed*/
#define cycle_speed 300
#define speed_solve 10000
uint32_t time_solve = 0; 
uint8_t flag_count = 0; 
/*
 * ===== LCD =====
 */
#define RS_p 16
#define EN_p 17
#define D4_p 29
#define D5_p 27
#define D6_p 25
#define D7_p 23

LiquidCrystal lcd(RS_p, EN_p, D4_p, D5_p, D6_p, D7_p);

/*
 * ===== MOTOR PIN =====
 */
// STEP (mapping theo mặt Rubik)
#define X_STEP_PIN 54   // D (Down - mặt vàng)
#define Y_STEP_PIN 60   // F (Front - mặt xanh lá)
#define Z_STEP_PIN 46   // L (Left - mặt cam)
#define E0_STEP_PIN 26  // R (Right - mặt đỏ)
#define E1_STEP_PIN 36  // B (Back - mặt xanh dương)
#define T1_STEP_PIN 35  // U (Up - mặt trắng)

// DIR
#define X_DIR_PIN 55
#define Y_DIR_PIN 61
#define Z_DIR_PIN 48
#define E0_DIR_PIN 28
#define E1_DIR_PIN 34
#define T1_DIR_PIN 37

// ENABLE
#define X_EN_PIN 38
#define Y_EN_PIN 56
#define Z_EN_PIN 62
#define E0_EN_PIN 24
#define E1_EN_PIN 30
#define T1_EN_PIN 19

/*
 * ===== CONFIG =====
 */
#define MOTOR_ENABLE  LOW
#define MOTOR_DISABLE HIGH

// Motor: 1.8deg, microstep 1/16
#define STEP_PER_REV 200
#define MICROSTEP    4
#define TOTAL_STEP_PER_REV (STEP_PER_REV * MICROSTEP)

// Rubik angle
#define STEP_90  (TOTAL_STEP_PER_REV / 4)
#define STEP_180 (TOTAL_STEP_PER_REV / 2)

/*
 * ===== ARRAY =====
 */
const uint8_t stepPins[6] = {
  X_STEP_PIN, Y_STEP_PIN, Z_STEP_PIN,
  E0_STEP_PIN, E1_STEP_PIN, T1_STEP_PIN
};

const uint8_t dirPins[6] = {
  X_DIR_PIN, Y_DIR_PIN, Z_DIR_PIN,
  E0_DIR_PIN, E1_DIR_PIN, T1_DIR_PIN
};

const uint8_t enPins[6] = {
  X_EN_PIN, Y_EN_PIN, Z_EN_PIN,
  E0_EN_PIN, E1_EN_PIN, T1_EN_PIN
};

/*
 * ===== RUBIK MOVE ENUM =====
 */
typedef enum {
  MOVE_STOP = 0,

  MOVE_U_CW, MOVE_U_CCW, MOVE_U_180,
  MOVE_D_CW, MOVE_D_CCW, MOVE_D_180,
  MOVE_L_CW, MOVE_L_CCW, MOVE_L_180,
  MOVE_R_CW, MOVE_R_CCW, MOVE_R_180,
  MOVE_F_CW, MOVE_F_CCW, MOVE_F_180,
  MOVE_B_CW, MOVE_B_CCW, MOVE_B_180

} RubikMove_t;

/*
 * ===== INIT =====
 */
void motor_init() {
  for (int i = 0; i < 6; i++) {
    pinMode(stepPins[i], OUTPUT);
    pinMode(dirPins[i], OUTPUT);
    pinMode(enPins[i], OUTPUT);

    digitalWrite(enPins[i], MOTOR_DISABLE);
  }
}

/*
 * ===== STEP CORE =====
 */
void motor_move(uint8_t id, bool dir, uint32_t steps, uint16_t period_us)
{
  digitalWrite(dirPins[id], dir);
  digitalWrite(enPins[id], MOTOR_ENABLE);

  for (uint32_t i = 0; i < steps; i++)
  {
    digitalWrite(stepPins[id], HIGH);
    delayMicroseconds(period_us);

    digitalWrite(stepPins[id], LOW);
    delayMicroseconds(period_us);
  }

  digitalWrite(enPins[id], MOTOR_DISABLE);
}

/*
 * ===== RUN MOVE =====
 */
void run_move(RubikMove_t move)
{
  uint8_t id = 0;
  bool dir = HIGH;
  uint32_t steps = STEP_90;

  switch (move)
  {
    // ===== D (Down - mặt vàng) ->  motor 0 (X) =====
    case MOVE_D_CW:   id = 0; dir = HIGH; steps = STEP_90; break;
    case MOVE_D_CCW:  id = 0; dir = LOW;  steps = STEP_90; break;
    case MOVE_D_180:  id = 0; dir = HIGH; steps = STEP_180; break;

    // ===== F (Front - mặt xanh lá) ->  motor 1 (Y) =====
    case MOVE_F_CW:   id = 1; dir = HIGH; steps = STEP_90; break;
    case MOVE_F_CCW:  id = 1; dir = LOW;  steps = STEP_90; break;
    case MOVE_F_180:  id = 1; dir = HIGH; steps = STEP_180; break;

    // ===== L (Left - mặt cam) -> motor 2 (Z) ===== ngược
    case MOVE_L_CW:   id = 2; dir = LOW; steps = STEP_90; break;
    case MOVE_L_CCW:  id = 2; dir = HIGH;  steps = STEP_90; break;
    case MOVE_L_180:  id = 2; dir = LOW; steps = STEP_180; break;

    // ===== R (Right - mặt đỏ) -> motor 3 (E0) ===== ngươc
    case MOVE_R_CW:   id = 3; dir = LOW; steps = STEP_90; break;
    case MOVE_R_CCW:  id = 3; dir = HIGH;  steps = STEP_90; break;
    case MOVE_R_180:  id = 3; dir = LOW; steps = STEP_180; break;

    // ===== B (Back - mặt xanh dương) -> motor 4 (E1) =====
    case MOVE_B_CW:   id = 4; dir = HIGH; steps = STEP_90; break;
    case MOVE_B_CCW:  id = 4; dir = LOW;  steps = STEP_90; break;
    case MOVE_B_180:  id = 4; dir = HIGH; steps = STEP_180; break;

    // ===== U (Up - mặt trắng) -> motor 5 (T1) ===== ngược
    case MOVE_U_CW:   id = 5; dir = LOW; steps = STEP_90; break;
    case MOVE_U_CCW:  id = 5; dir = HIGH;  steps = STEP_90; break;
    case MOVE_U_180:  id = 5; dir = LOW; steps = STEP_180; break;

    default: return;
  }
  motor_move(id, dir, steps, cycle_speed);
}

/*
 * ===== LCD =====
 */
void lcd_update(uint32_t val)
{
  lcd.setCursor(0, 1);
  lcd.print("Count: ");
  lcd.print(val);
  lcd.print("   ");
}

/*
 * ===== SETUP =====
 */
void setup()
{
  Serial.begin(115200);

  motor_init();

  lcd.begin(16, 2);
  lcd.setCursor(0, 0);
  lcd.print("<<Rubik Ready>>");
  lcd.setCursor(0, 1);
  lcd.print("*Time:          ");
}

/*
* ===== Rubik slove ======. 
*/
#define MAX_MOVES 40

RubikMove_t moveQueue[MAX_MOVES];
uint8_t moveCount = 0;
uint8_t moveIndex = 0;

bool isRunning = false;


RubikMove_t parse_move(String token)
{
  token.trim();

  if (token == "D")  return MOVE_D_CW;
  if (token == "D'") return MOVE_D_CCW;
  if (token == "D2") return MOVE_D_180;

  if (token == "U")  return MOVE_U_CW;
  if (token == "U'") return MOVE_U_CCW;
  if (token == "U2") return MOVE_U_180;

  if (token == "F")  return MOVE_F_CW;
  if (token == "F'") return MOVE_F_CCW;
  if (token == "F2") return MOVE_F_180;

  if (token == "B")  return MOVE_B_CW;
  if (token == "B'") return MOVE_B_CCW;
  if (token == "B2") return MOVE_B_180;

  if (token == "L")  return MOVE_L_CW;
  if (token == "L'") return MOVE_L_CCW;
  if (token == "L2") return MOVE_L_180;

  if (token == "R")  return MOVE_R_CW;
  if (token == "R'") return MOVE_R_CCW;
  if (token == "R2") return MOVE_R_180;

  return MOVE_STOP;
}


void parse_data(String input)
{
  moveCount = 0;

  int start = input.indexOf("start:");
  int end   = input.indexOf(":end");

  if (start == -1 || end == -1) return;

  String data = input.substring(start + 6, end);

  while (data.length() > 0 && moveCount < MAX_MOVES)
  {
    int sep = data.indexOf(';');

    String token;
    if (sep != -1) {
      token = data.substring(0, sep);
      data = data.substring(sep + 1);
    } else {
      token = data;
      data = "";
    }

    RubikMove_t mv = parse_move(token);
    if (mv != MOVE_STOP) {
      moveQueue[moveCount++] = mv;
    }
  }

  moveIndex = 0;
}

void parse_control(String input)
{
  if (input.indexOf("control: start") != -1) {
    isRunning = true;
  }

  if (input.indexOf("control: stop") != -1) {
    isRunning = false;
  }
}

void serial_read()
{
  if (Serial.available())
  {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("control:"))
      parse_control(cmd);

    else if (cmd.startsWith("start:"))
      parse_data(cmd);
  }
}

void run_queue()
{
  if (!isRunning) return;

  if (moveIndex >= moveCount)
  {
    isRunning = false;
    return;
  }

  run_move(moveQueue[moveIndex]);  // chạy xong 1 move (blocking)
  moveIndex++;
  delayMicroseconds(speed_solve); 
}





/*
 * ===== LOOP =====
 */
void loop()
{
  // run_move(MOVE_D_CW);
  // delay(5);

  // run_move(MOVE_U_CW);
  // delay(5);

  // run_move(MOVE_F_CW);
  // delay(5);
  
  // run_move(MOVE_B_CW);
  // delay(5);

  // run_move(MOVE_L_CW);
  // delay(5);

  // run_move(MOVE_R_CW);
  // delay(5);

  serial_read();   // nhận lệnh khi idle

  if (isRunning)
  {
    if(flag_count == 1){
      time_solve = millis(); 
      flag_count = 0; 
    }
    run_queue();
  }

  if(isRunning ==  0 && flag_count == 0){
    lcd.setCursor(8, 1);
    lcd.print((float)(millis() - time_solve)/1000);
    flag_count = 1; 
  }
  //print
  
  // static uint32_t cnt = 0;
  // lcd_update(cnt++);


  // delay(5000); 
}