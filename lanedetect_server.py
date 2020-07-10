import cv2
import numpy as np 
import socket
import sys
from time import sleep

# ROI
def ROI(frame, vertices, color3 = (255, 255, 255), color1 = 255):
    mask = np.zeros_like(frame) # img 크기와 같은 이미지(배열)
    if len(frame.shape) > 2: # 컬러 이미지(3채널)일 때 
        color = color3
    else: # 흑백 이미지(1채널)일 때 
        color = color1

    # mask에 vertices 점들로 이뤄진 다각형 부분을 color로 채움
    cv2.fillPoly(mask, vertices, color)
    # 이미지와 ROI를 합성
    ROI_img = cv2.bitwise_and(frame, mask) 
    return ROI_img

# 선 그리기 함수
def draw_line(frame, lines, color = [0, 0, 255], thickness = 10):
    cv2.line(frame, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)

# 대표선 추출 함수
def get_fitline(frame, lines):
    def reject_outliers(data, m=2):
        return data[(abs(data - np.mean(data, 0)) < 2 * np.std(data, 0)).prod(-1)]
    lines = reject_outliers(lines)
    line = np.mean(lines, 0).astype(np.int16)
    x = line[0] + (line[2] - line[0]) // 2
    y = line[1] + (line[3] - line[1]) // 2
    return line, x, y

    
def intersaction(L_line, R_line):
    m1 = (L_line[3] - L_line[1]) / (L_line[2] - L_line[0])
    m2 = (R_line[3] - R_line[1]) / (R_line[2] - R_line[0])
    x = int((R_line[1] - L_line[1] + L_line[0] * m1 - R_line[0] * m2) / (m1 - m2))
    y = int(m1 * (x - L_line[0]) + L_line[1]) 
    return (x, y)

def align_lines(lines):
    t_line = np.zeros_like(lines)
    for i, line in enumerate(lines):
        if line[1] <= line[3]:
            t_line[i] = line
        else:
            t_line[i,3] = line[1]
            t_line[i,1] = line[3]
            t_line[i,2] = line[0]
            t_line[i,0] = line[2]
    return t_line

def detect_lane(img):
    (height, width, k) = img.shape
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # 흑백 이미지로 변환
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0) # Blur 효과
    kernel = np.ones((5,5), np.uint8)
    dilation_img = cv2.dilate(blur_img, kernel, 1) # 팽창 연산
    canny_img = cv2.Canny(dilation_img, 200, 300) # 외곽선 검출
    vertices = np.array([[(0, height * 3 / 2), 
                          (0, height - 30), 
                          (width, height - 30), 
                          (width, height * 3 / 2)]], dtype = np.int32)
    ROI_img = ROI(canny_img, vertices)

    line_arr = cv2.HoughLinesP(ROI_img, 1, np.pi / 180, 10, 12, 10) # 직선 검 출
    line_arr = align_lines(np.squeeze(line_arr, 1)) # 불필요한 값 정리
    for i in range(0, len(line_arr)):
        l = line_arr[i, :]
        cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 4)
        cv2.circle(img, (l[0], l[1]), 5, (0, 255, 0), -1)
    slope_degree = np.arctan2(line_arr[:, 3]-line_arr[:, 1], 
                              line_arr[:, 2]-line_arr[:, 0]) * 180 / np.pi
    # 수평 기울기 제한
    line_arr = line_arr[np.abs(slope_degree) < 150] 
    slope_degree = slope_degree[np.abs(slope_degree) < 150]
    # 수직 기울기 제한
    line_arr = line_arr[np.abs(slope_degree) > 30] 
    slope_degree = slope_degree[np.abs(slope_degree) > 30]

    L_lines = line_arr[line_arr[:, 0] < (width // 2 + 30)]
    R_lines = line_arr[line_arr[:, 0] > (width // 2 - 30)]
    
    if len(L_lines) == 0 or len(R_lines) == 0 : # 한 쪽 차선만 인식될 때
        lines = line_arr
        fit_line, x, y = get_fitline(img, lines)
        draw_line(img, fit_line) 
        angle = np.arctan2(
            fit_line[1] - fit_line[3], 
            fit_line[0] - fit_line[2]) * 180 / np.pi
        is_double_line = False
    else:
        left_fit_line, lx, ly = get_fitline(img, L_lines) 
        right_fit_line, rx, ry = get_fitline(img, R_lines)
        center_point = int(lx + (rx - lx) / 2), int(max(ly, ry)) # 양쪽 차선 중심점
        inter_point = intersaction(left_fit_line, right_fit_line)
        angle = np.arctan2(inter_point[1]-center_point[1] , inter_point[0]-center_point[0]) * 180 / np.pi
        if inter_point[0] < center_point[0]: 
            angle = angle + 180
        draw_line(img, [inter_point[0], inter_point[1], center_point[0], center_point[1]])
        is_double_line = True
    return img, angle, len(L_lines), len(R_lines)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


HOST=''
PORT=2000
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM) 
s.bind((HOST,PORT))
s.listen(1)
print('Socket now listening..')

#연결, conn에는 소켓 객체, addr은 소켓에 바인드 된 주소 
conn,addr=s.accept()
print(f'connected by {addr}')
iteration = 0.
datalist = []

while True:
    try:
        length = recvall(conn, 16)
        if length:
            stringData = recvall(conn, int(length))
            data = np.frombuffer(stringData, dtype = 'uint8')
            frame = cv2.imdecode(data, 1)
            signal = None
            angle = None
            try: 
                frame, angle, L, R = detect_lane(frame) 
                if np.abs(angle) <= 110 and np.abs(angle) >= 70:
                    signal = b'f'
                    iteration = 0
                elif np.abs(angle) < 60 and (R == 0):
                    # datalist.append('r')
                    signal = b'r'
                    iteration = 0
                    # print("Right")
                elif np.abs(angle) > 120 and (L == 0):
                    # datalist.append('l')
                    signal = b'l'
                    iteration = 0
            except:
                iteration += 1
                if iteration > 30:
                    signal = b'b'
                    # angle = None
                pass

            # if len(datalist) > 3:
            #     s = datalist.count('s')
            #     r = datalist.count('r')
            #     l = datalist.count('l')
            #     max_val = max(s,r,l)
            #     if max_val == s:
            #         signal = 's'
            #     elif max_val == r:
            #         signal = 'r'
            #     elif max_val == l:
            #         signal = 'l'
            
            if signal is not None:
                # print(signal.decode())
                if angle is not None:
                    # cv2.putText(frame, signal+str(angle), (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255,0))
                    print(signal.decode() + str(angle))
                cv2.imshow('SERVER',frame)
                conn.send( signal )
                # datalist = []
                
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                conn.close()
                break
    # except TypeError:
        # print('no data')
        # pass
    except Exception as e:
        print(e)
        pass
conn.close()
cv2.destroyAllWindows()
