/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Rubik Solver STM32 (DMA + IDLE + Stepper)
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "dma.h"
#include "usart.h"
#include "lcd.h"
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
typedef struct {
    GPIO_TypeDef* port;
    uint16_t pin;
} GPIO_Pin_t;

typedef enum {
  MOVE_STOP = 0,
  MOVE_U_CW, MOVE_U_CCW, MOVE_U_180,
  MOVE_D_CW, MOVE_D_CCW, MOVE_D_180,
  MOVE_L_CW, MOVE_L_CCW, MOVE_L_180,
  MOVE_R_CW, MOVE_R_CCW, MOVE_R_180,
  MOVE_F_CW, MOVE_F_CCW, MOVE_F_180,
  MOVE_B_CW, MOVE_B_CCW, MOVE_B_180
} RubikMove_t;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define RX_SIZE 128
#define MAX_MOVES 40

#define cycle_speed 300
#define speed_solve 10000

#define STEP_PER_REV 200
#define MICROSTEP    4
#define TOTAL_STEP_PER_REV (STEP_PER_REV * MICROSTEP)

#define STEP_90  (TOTAL_STEP_PER_REV / 4)
#define STEP_180 (TOTAL_STEP_PER_REV / 2)
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
TIM_HandleTypeDef htim1;

/* USER CODE BEGIN PV */

/* UART DMA */
uint8_t rxBuf[RX_SIZE];
uint8_t rxData[RX_SIZE];

/* Queue */
RubikMove_t moveQueue[MAX_MOVES];
uint8_t moveCount = 0;
uint8_t moveIndex = 0;
bool isRunning = false;

/* GPIO mapping (SỬA THEO BOARD) */
GPIO_Pin_t stepPins[6] = {
    {GPIOA, GPIO_PIN_0},
    {GPIOA, GPIO_PIN_1},
    {GPIOA, GPIO_PIN_2},
    {GPIOA, GPIO_PIN_3},
    {GPIOA, GPIO_PIN_4},
    {GPIOA, GPIO_PIN_5}
};

GPIO_Pin_t dirPins[6] = {
    {GPIOA, GPIO_PIN_6},
    {GPIOA, GPIO_PIN_7},
    {GPIOA, GPIO_PIN_8},
    {GPIOA, GPIO_PIN_9},
    {GPIOA, GPIO_PIN_10},
    {GPIOA, GPIO_PIN_11}
};

GPIO_Pin_t enPins[6] = {
    {GPIOB, GPIO_PIN_0},
    {GPIOB, GPIO_PIN_1},
    {GPIOB, GPIO_PIN_2},
    {GPIOB, GPIO_PIN_10},
    {GPIOB, GPIO_PIN_11},
    {GPIOB, GPIO_PIN_12}
};

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_TIM1_Init(void);

/* USER CODE BEGIN PFP */
void delay_us(uint16_t us);
void motor_move(uint8_t id, uint8_t dir, uint32_t steps, uint16_t period_us);
void run_move(RubikMove_t move);
void run_queue(void);
void parse_data(char* input);
void parse_control(char* input);
RubikMove_t parse_move(char* token);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* ===== Delay microsecond ===== */
void delay_us(uint16_t us)
{
    __HAL_TIM_SET_COUNTER(&htim1, 0);
    while (__HAL_TIM_GET_COUNTER(&htim1) < us);
}

/* ===== Motor ===== */
void motor_move(uint8_t id, uint8_t dir, uint32_t steps, uint16_t period_us)
{
    HAL_GPIO_WritePin(dirPins[id].port, dirPins[id].pin,
                      dir ? GPIO_PIN_SET : GPIO_PIN_RESET);

    HAL_GPIO_WritePin(enPins[id].port, enPins[id].pin, GPIO_PIN_RESET);

    for (uint32_t i = 0; i < steps; i++)
    {
        HAL_GPIO_WritePin(stepPins[id].port, stepPins[id].pin, GPIO_PIN_SET);
        delay_us(period_us);
        HAL_GPIO_WritePin(stepPins[id].port, stepPins[id].pin, GPIO_PIN_RESET);
        delay_us(period_us);
    }

    HAL_GPIO_WritePin(enPins[id].port, enPins[id].pin, GPIO_PIN_SET);
}

/* ===== Run move ===== */
void run_move(RubikMove_t move)
{
    uint8_t id = 0, dir = 1;
    uint32_t steps = STEP_90;

    switch (move)
    {
        case MOVE_D_CW: id=0; break;
        case MOVE_D_CCW: id=0; dir=0; break;
        case MOVE_D_180: id=0; steps=STEP_180; break;

        case MOVE_F_CW: id=1; break;
        case MOVE_F_CCW: id=1; dir=0; break;
        case MOVE_F_180: id=1; steps=STEP_180; break;

        case MOVE_L_CW: id=2; dir=0; break;
        case MOVE_L_CCW: id=2; break;
        case MOVE_L_180: id=2; dir=0; steps=STEP_180; break;

        case MOVE_R_CW: id=3; dir=0; break;
        case MOVE_R_CCW: id=3; break;
        case MOVE_R_180: id=3; dir=0; steps=STEP_180; break;

        case MOVE_B_CW: id=4; break;
        case MOVE_B_CCW: id=4; dir=0; break;
        case MOVE_B_180: id=4; steps=STEP_180; break;

        case MOVE_U_CW: id=5; dir=0; break;
        case MOVE_U_CCW: id=5; break;
        case MOVE_U_180: id=5; dir=0; steps=STEP_180; break;

        default: return;
    }

    motor_move(id, dir, steps, cycle_speed);
}

/* ===== Parse ===== */
RubikMove_t parse_move(char* t)
{
    if (!strcmp(t,"D")) return MOVE_D_CW;
    if (!strcmp(t,"D'")) return MOVE_D_CCW;
    if (!strcmp(t,"D2")) return MOVE_D_180;

    if (!strcmp(t,"U")) return MOVE_U_CW;
    if (!strcmp(t,"U'")) return MOVE_U_CCW;
    if (!strcmp(t,"U2")) return MOVE_U_180;

    if (!strcmp(t,"F")) return MOVE_F_CW;
    if (!strcmp(t,"F'")) return MOVE_F_CCW;
    if (!strcmp(t,"F2")) return MOVE_F_180;

    if (!strcmp(t,"B")) return MOVE_B_CW;
    if (!strcmp(t,"B'")) return MOVE_B_CCW;
    if (!strcmp(t,"B2")) return MOVE_B_180;

    if (!strcmp(t,"L")) return MOVE_L_CW;
    if (!strcmp(t,"L'")) return MOVE_L_CCW;
    if (!strcmp(t,"L2")) return MOVE_L_180;

    if (!strcmp(t,"R")) return MOVE_R_CW;
    if (!strcmp(t,"R'")) return MOVE_R_CCW;
    if (!strcmp(t,"R2")) return MOVE_R_180;

    return MOVE_STOP;
}

void parse_data(char* input)
{
    moveCount = 0;

    char* start = strstr(input, "start:");
    char* end   = strstr(input, ":end");
    if (!start || !end) return;

    start += 6;

    char buffer[128];
    strncpy(buffer, start, end - start);
    buffer[end - start] = 0;

    char* token = strtok(buffer, ";");
    while (token && moveCount < MAX_MOVES)
    {
        RubikMove_t mv = parse_move(token);
        if (mv != MOVE_STOP)
            moveQueue[moveCount++] = mv;

        token = strtok(NULL, ";");
    }

    moveIndex = 0;
}

void parse_control(char* input)
{
    if (strstr(input, "control: start")) isRunning = true;
    if (strstr(input, "control: stop")) isRunning = false;
}

/* ===== DMA IDLE CALLBACK ===== */
void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *huart, uint16_t Size)
{
    if (huart->Instance == USART1)
    {
        memcpy(rxData, rxBuf, Size);
        rxData[Size] = 0;

        parse_control((char*)rxData);
        parse_data((char*)rxData);

        HAL_UARTEx_ReceiveToIdle_DMA(&huart1, rxBuf, RX_SIZE);
        __HAL_DMA_DISABLE_IT(huart1.hdmarx, DMA_IT_HT);
    }
}

/* ===== Queue ===== */
void run_queue(void)
{
    if (!isRunning) return;

    if (moveIndex >= moveCount)
    {
        isRunning = false;
        return;
    }

    run_move(moveQueue[moveIndex++]);
    delay_us(speed_solve);
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  */
int main(void)
{
  HAL_Init();
  SystemClock_Config();

  MX_GPIO_Init();
  MX_TIM1_Init();
  MX_DMA_Init();
  MX_USART1_UART_Init();
  MX_LCD_Init();

  /* USER CODE BEGIN 2 */
  HAL_TIM_Base_Start(&htim1);

  HAL_UARTEx_ReceiveToIdle_DMA(&huart1, rxBuf, RX_SIZE);
  __HAL_DMA_DISABLE_IT(huart1.hdmarx, DMA_IT_HT);

  lcd_clear();
  lcd_setCursor(0,0);
  lcd_print("Rubik Ready");
  /* USER CODE END 2 */

  while (1)
  {
    /* USER CODE BEGIN WHILE */
    run_queue();
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    /* USER CODE END 3 */
  }
}
