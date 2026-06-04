import numpy as np
import enum

class MovementDirection(enum.Enum):
    X = 0
    Z = 1

class BatteryError(Exception):
    """Battery without energy"""
    pass

class EnergyComsuption:
    def __init__(self, mass: float = 1.5, ## Kg
                 payload: float = 0.0, ## kg, battery more payload
                 power_efficiency: float = 1.0, ### from the article original value is 0.5
                 lift_drag_ration: float = 3.0, ### benchmarket 
                 external_power: float = 0.0, # Watts
                 battery_charging_constant: float = 0.01,
                 battery_capacity: float = 5000.0,  ## mAh
                 battery_voltage: float=14.0,  ## Volts)
                 battery_initial_status: float = 1.0,
                 X_ref_area: float = 0.015, ## m^2
                 Z_ref_area: float = 0.0335, ## m^2
                 X_drag_coef: float = 0.8,
                 Z_drag_coef: float = 0.8,
                ):
        self.mass = mass
        self.payload = payload
        self.external_power = external_power

        self.POWER_EFFICIENCY = power_efficiency
        self.RATIO = lift_drag_ration
        self.BATTERY_CHARGING_CONSTANT = battery_charging_constant
        self.BATTERY_VOLTAGE = battery_voltage
        
        self.GRAVITY = 9.81
        self.AIR_DENSITY = 1.225
        self.RADIUS_PROP = 0.127
        self.NUMBER_ROTORS = 4
        self.disk_area = self.NUMBER_ROTORS*np.pi*(self.RADIUS_PROP**2)

        self.X_ref_area = X_ref_area
        self.Z_ref_area = Z_ref_area
        self.X_drag_coef = X_drag_coef
        self.Z_drag_coef = Z_drag_coef

        self.battery_total_energy = battery_capacity*self.BATTERY_VOLTAGE*3.6 ## the energy in joules
        self.battery_current_energy = battery_initial_status*self.battery_total_energy
        self.battery_status = battery_initial_status

    def air_resistence(self, movement_direction, drone_speed):
        if movement_direction == MovementDirection.X.value:
            area = self.X_ref_area
            coef = self.X_drag_coef
        elif movement_direction == MovementDirection.Z.value:
            area = self.Z_ref_area
            coef = self.Z_drag_coef
        else: 
            return ValueError
        
        drag = 0.5*self.AIR_DENSITY*coef*area*drone_speed**2
        #print(f"Drag: {drag}")
        
        return drag

    def get_inclination_angle(self, drone_speed, movement_direction):
        total_mass = self.mass + self.payload

        if movement_direction == MovementDirection.X.value:
            drag = self.air_resistence(movement_direction, drone_speed)
            theta = np.arctan(drag/(total_mass*self.GRAVITY))
            #print(f"Theta = {theta}")
        elif movement_direction == MovementDirection.Z.value:
            theta = 0               
        else:
            return ValueError
        return theta

    def get_trust_force(self, theta, drone_speed, movement_direction):
        total_mass = self.mass + self.payload
        
        if movement_direction == MovementDirection.X.value:
            hover_trust = (total_mass*self.GRAVITY)/(np.cos(theta))
            
            total_trust = hover_trust

        elif movement_direction == MovementDirection.Z.value:
            drag = self.air_resistence(movement_direction, drone_speed)
            
            total_trust = total_mass*self.GRAVITY + drag
        else: 
            return ValueError

        return total_trust 


    def get_total_power(self, trust, drone_speed, movement_direction):
        # Reference:
        # Practical Endurance Estimation for Minimizing
        # Energy Consumption of Multirotor Unmanned
        # Aerial Vehicles
        total_mass = self.mass + self.payload
 
        # 1. Induced Velocity at Hover
        # Derived from T = 2 * rho * A * v_h^2
        v_h = np.sqrt(trust / (2 * self.AIR_DENSITY * self.disk_area))

        # 2. Induced Velocity in Forward Flight (Glauert approximation)
        v_i = trust / (2 * self.AIR_DENSITY * self.disk_area * np.sqrt(drone_speed**2 + v_h**2))

        # 3. Induced Power (
        induced_power = trust * v_i

        # 4. Parasite Power
        drag = self.air_resistence(movement_direction, drone_speed)
        parasite_power = drag * drone_speed

        # This values is changed to match the information presented in the holybro site:
        ### Flight time: ~18 minutes hover with no additional payload. Tested with 5000mAh Battery.
        return (induced_power + parasite_power) / 0.6

    
    def get_power_consumed(self, drone_speed, movement_direction):
        total_mass = self.mass + self.payload
        
        theta = self.get_inclination_angle(drone_speed, movement_direction)
        trust = self.get_trust_force(theta, drone_speed, movement_direction)        
        total_power = self.get_total_power(trust, drone_speed, movement_direction)
        
        #print(f"THe hover power is: {hover_power}")
        return self.external_power + total_power

   
    def change_external_power(self, new_external_charge):
        self.external_charge = new_external_charge

    ###Assuming the battery voltage doesn't change with the time during discharging

    def get_energy_consumed(self, power_consumed, delta_time):
        return power_consumed*delta_time

    def discharge_battery(self, energy_consumed):
        self.battery_current_energy -= energy_consumed

        if self.battery_current_energy <= 0.0:
            self.battery_current_energy = 0.0 
            self.update_battery_status()       
            raise BatteryError("Drone has no energy.")


    def charge_battery(self, delta_time):
        self.battery_status += self.BATTERY_CHARGING_CONSTANT*delta_time

        self.battery_status = min(1.0, self.battery_status)
    
    def get_current_battery_energy(self):
        return self.battery_current_energy
    
    def get_battery_status(self):
        return self.battery_status

    def update_battery_status(self):
        self.battery_status = (self.battery_current_energy/self.battery_total_energy)
    
    
    def manage_battery_during_fly(self, duration, drone_speed, movement_direction):
        power = self.get_power_consumed(drone_speed, movement_direction)

        energy_consumed = self.get_energy_consumed(power, duration)

        self.discharge_battery(energy_consumed)

        self.update_battery_status()

        return self.battery_status
    
