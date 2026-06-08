# src/config.py
DATA_FOLDER = "D:/IT/3rd Semester/ML/ultrasonic_project/data"
T_CELSIUS = 20.0
C_SPEED = 331.3 + 0.606 * T_CELSIUS
SAMPLING_RATE = 20
DT_FRAME = 1 / SAMPLING_RATE

CLASS_KEYWORDS = {
    'Medium': ['medium'],
    'Fast': ['fast'],
    'Diagonal': ['dia'],
}
EXCLUDE_KEYWORDS = ['stationary', 'human', 'slow']