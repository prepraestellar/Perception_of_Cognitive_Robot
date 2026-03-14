#include <webots/Robot.hpp>
#include <webots/Camera.hpp>

using namespace webots;

int main(int argc, char **argv) {
  Robot *robot = new Robot();

  int timeStep = robot->getBasicTimeStep();

  Camera *camera = robot->getCamera("camera");
  if (!camera) {
    std::cout << "Camera not found on Sojourner!" << std::endl;
  } else {
    std::cout << "Camera found on Sojourner!" << std::endl;
  }
  if (camera)
    camera->enable(timeStep);

  // Main simulation loop
  while (robot->step(timeStep) != -1) {
    // You can read camera data here
    // const unsigned char *image = camera->getImage();
  }

  delete robot;
  return 0;
}
