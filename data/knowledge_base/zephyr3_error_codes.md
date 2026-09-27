# Zephyr-3 Error Codes and Troubleshooting

## E-101: Low Battery Critical
The battery has fallen below 5 percent and the robot could not reach a charging station. Manually push the robot to a charger using the tow handle after pressing the "Free Wheel" button on the rear panel.

## E-204: Payload Overweight
The load cells measured a payload above 1,200 kg. The robot will refuse to move. Remove weight from the pallet until the status light turns green.

## E-417: Lidar Calibration Drift
The lidar's measured distances disagree with known fiducial marker positions by more than 3 centimetres. First clean the lidar window. If the error persists, run the "Auto-Calibrate" routine from the Nimbus Fleet Console, which takes about 4 minutes and requires the robot to be parked on a calibration marker (marker ID 0 to 9).

## E-520: Wheel Motor Overheat
A drive motor exceeded 85 degrees Celsius. The robot stops and waits for the motor to cool below 60 degrees Celsius before resuming. Repeated E-520 errors usually indicate worn wheel bearings; schedule a technician visit.

## E-777: Fleet Connection Lost
The robot has lost its connection to the Nimbus Fleet Console for more than 30 seconds. It finishes its current task, parks in the nearest safe zone, and waits. Check the warehouse Wi-Fi access points on the 5 GHz band; the Zephyr-3 does not support 2.4 GHz networks.
