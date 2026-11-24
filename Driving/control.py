"""
control.py -- pedal control. Pure python/numpy, so it is testable without CARLA.

SpeedController replaces the longitudinal half of CARLA's VehiclePIDController.
Measured on the first M2 run in CARLA: that PID (K_P = 1 per km/h, so 1 km/h of
error saturates the pedal) switched pedals 208 times a minute and braked in 21%
of cruising ticks. On a car model fitted from that same log (50 ms delay, 0.15 s
engine lag), this controller gives ~1 switch a minute and no cruising brake.
Steering stays with VehiclePIDController's lateral PID.
"""
import numpy as np

BRAKE_FULL_DECEL = 6.0    # m/s^2 at brake = 1.0, 20-30 km/h; from the model fitted to the first
                          # M2 CARLA log (~3.7 brake + engine braking/drag). R^2 0.69: approximate


class SpeedController:
    """PI speed control in m/s with anti-windup and a coast band.

    Replaces the longitudinal half of VehiclePIDController. That one uses
    K_P = 1 per km/h, so 1 km/h of error already saturates the pedal; in the
    first M2 run it switched throttle 0 -> 0.75 -> brake about every 0.6 s while
    cruising. Here: small errors give small pedal changes, a true integrator
    finds the cruise throttle, and a small overspeed coasts instead of braking."""

    def __init__(self, kp=0.5, ki=0.3, kff=0.1, dt=0.05, max_throttle=0.75, max_brake=0.5,
                 coast_band=0.15):
        self.kp, self.ki, self.kff, self.dt = kp, ki, kff, dt
        self.max_throttle, self.max_brake, self.coast_band = max_throttle, max_brake, coast_band
        self.i_max = 0.6 / ki if ki > 0 else 0.0
        self.i = 0.0

    def step(self, v_target, v, a_cmd, hold):
        if hold or v_target < 0.05:
            self.i = 0.0                       # no wind-up while held at a stop
        e = v_target - v
        u = self.kp * e + self.ki * self.i + self.kff * a_cmd
        throttle = brake = 0.0
        if u >= 0.0:
            throttle = min(u, self.max_throttle)
        elif u < -self.coast_band:
            brake = min(-u - self.coast_band, self.max_brake)
        saturated = (u >= self.max_throttle and e > 0) or (brake >= self.max_brake and e < 0)
        if not saturated:
            self.i = float(np.clip(self.i + e * self.dt, -self.i_max, self.i_max))
        return throttle, brake


def shape_control(c, dec):
    """VehiclePIDController steers and tracks v_target; the decision layer
    overrides the pedals where it must."""
    if dec.emergency:
        c.throttle, c.brake = 0.0, 1.0
    elif dec.hold_brake:
        c.throttle, c.brake = 0.0, max(c.brake, 0.6)
    elif dec.a_cmd < -3.0:          # harder than the PID's max_brake can deliver
        c.throttle, c.brake = 0.0, max(c.brake, min(1.0, -dec.a_cmd / BRAKE_FULL_DECEL))
    if c.brake > 0.0:
        c.throttle = 0.0
    return c
