#include <webots/Robot.hpp>
#include <webots/Motor.hpp>
#include <webots/Keyboard.hpp>

#define SPEED 6.28  // rad/s

using namespace webots;

int main() {
  Robot robot;

  int timeStep = robot.getBasicTimeStep();

  // Get motors
  Motor *leftMotor  = robot.getMotor("left wheel motor");
  Motor *rightMotor = robot.getMotor("right wheel motor");

  // Velocity control
  leftMotor->setPosition(INFINITY);
  rightMotor->setPosition(INFINITY);
  leftMotor->setVelocity(0.0);
  rightMotor->setVelocity(0.0);

  // Keyboard
  Keyboard keyboard;
  keyboard.enable(timeStep);

  while (robot.step(timeStep) != -1) {
    int key = keyboard.getKey();

    double leftSpeed = 0.0;
    double rightSpeed = 0.0;

    switch (key) {
      case 'W':
        leftSpeed = SPEED;
        rightSpeed = SPEED;
        break;

      case 'S':
        leftSpeed = -SPEED;
        rightSpeed = -SPEED;
        break;

      case 'A':
        leftSpeed = -SPEED;
        rightSpeed = SPEED;
        break;

      case 'D':
        leftSpeed = SPEED;
        rightSpeed = -SPEED;
        break;

      default:
        break;
    }

    leftMotor->setVelocity(leftSpeed);
    rightMotor->setVelocity(rightSpeed);
  }

  return 0;
}
