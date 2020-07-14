import cv2
import numpy as np
import socket
import sys
import time
import threading
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522

stop = False

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def send_img():
    width, height = 480, 360
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        try:
            ret, frame = cap.read()
#            frame = cv2.flip(frame, -1)
            frame = cv2.resize(frame, (width, height))
            encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
            result, imgencode = cv2.imencode('.jpg', frame, encode_param)
            data = np.array(imgencode)
            stringData = data.tobytes()

            sock.send( str(len(stringData)).encode().ljust(16));
            sock.send( stringData );

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            break
    sock.close()
    cap.release()
    cv2.destroyAllWindows()

def recv_data():
    global stop
    while True:
        try:
            if stop is True:
                motor_move(b's')
            else:
                data = sock.recv(1)
                motor_move(data)
        except:
            break
    sock.close()
    GPIO.cleanup()

def setPinConfig(EN, INA, INB):
    GPIO.setup(EN, GPIO.OUT)
    GPIO.setup(INA, GPIO.OUT)
    GPIO.setup(INB, GPIO.OUT)
    pwm = GPIO.PWM(EN, 100)
    pwm.start(0)
    return pwm

def setUltraConfig():
    GPIO.setup(17, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(18, GPIO.IN)

def setMotorContorl(pwm, INA, INB, speed, stat):
    pwm.ChangeDutyCycle(speed)

    if stat == FORWARD:
        GPIO.output(INA, HIGH)
        GPIO.output(INB, LOW)

    elif stat == BACKWORD:
        GPIO.output(INA, LOW)
        GPIO.output(INB, HIGH)

    elif stat == STOP:
        GPIO.output(INA, LOW)
        GPIO.output(INB, LOW)


def setMotor(ch, speed, stat):
    if ch == CH1:
        setMotorContorl(pwmA, IN1, IN2, speed, stat)
    else:
        setMotorContorl(pwmB, IN3, IN4, speed, stat)

def motor_move(data):
    print(data)
    try:
        if data == b'f':
            setMotor(CH1, 18, FORWARD)
            setMotor(CH2, 18, FORWARD)
        elif data == b'b':
            setMotor(CH1, 18, BACKWORD)
            setMotor(CH2, 18, BACKWORD)
        elif data == b'r':
            setMotor(CH1, 18, STOP)
            setMotor(CH2, 36, FORWARD)
        elif data == b'l':
            setMotor(CH1, 36, FORWARD)
            setMotor(CH2, 18, STOP)
        elif data == b's':
            setMotor(CH1, 18, STOP)
            setMotor(CH2, 18, STOP)
    except Exception as e:
        print(e)

def is_mfrc():
    while True:
        try:
            ids = reader.read_id_no_block()
            if ids:
                print('mfrc stop')
                yield True
                #time.sleep(3)
            else:
                yield False
        except:
             break

def is_ultra():
    while True:
        try:
            GPIO.output(17, True)
            time.sleep(0.00001)
            GPIO.output(17, False)
        
            while GPIO.input(18) == 0:
                _start = time.time()
            
            while GPIO.input(18) == 1:
                _stop = time.time()
            
            time_interval = _stop - _start
            distance = time_interval * 17000
            distance = round(distance, 2)
        
            if distance < 15:
                print('ultra stop')
                yield True
                #time.sleep(3)
            else:
                yield False
        except:
            break
        
def sensor():
    global stop
    while True:
        try:
            mfrc_val = next(is_mfrc())
            ultra_val = next(is_ultra())
            if mfrc_val is True or ultra_val is True:
                stop = True
                time.sleep(4)
                stop = False
        except:
            break

ip = '172.30.1.17'
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (ip, 2000)
connected = False
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

reader = SimpleMFRC522()

STOP  = 0
FORWARD  = 1
BACKWORD = 2

CH1 = 0
CH2 = 1

OUTPUT = 1
INPUT = 0

HIGH = 1
LOW = 0

#PWM PIN
ENA = 26  #37 pin
ENB = 20   #27 pin

#GPIO PIN
IN1 = 19  #37 pin
IN2 = 13  #35 pin
IN3 = 6   #31 pin
IN4 = 12   #29 pin

while not connected:
    try:
        sock.connect(server_address)
        connected = True
    except Exception as e:
        pass

pwmA = setPinConfig(ENA, IN1, IN2)
pwmB = setPinConfig(ENB, IN3, IN4)
setUltraConfig()

t1 = threading.Thread(target=send_img)
t2 = threading.Thread(target=recv_data)
t3 = threading.Thread(target=sensor)

t1.start()
t2.start()
t3.start()
