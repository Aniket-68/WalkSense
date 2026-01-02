# interaction/aux_controller.py

from interaction.haptics import Haptics
from interaction.buzzer import Buzzer
from interaction.led import LED

class AuxController:
    def __init__(self):
        self.haptics = Haptics()
        self.buzzer = Buzzer()
        self.led = LED()

    def trigger_haptic(self, intensity):
        self.haptics.vibrate(intensity)

    def trigger_buzzer(self, pattern):
        self.buzzer.beep(pattern)

    def trigger_led(self, color):
        self.led.on(color)
