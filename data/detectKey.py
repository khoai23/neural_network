import win32api, win32con
import time, random
# from msvcrt import getch

def keyWasUnPressed():
	print("enabling joystick...")
	#enable joystick here

def keyWasPressed():
	print("disabling joystick...")
	#disable joystick here

def isKeyPressed(key):
	#"if the high-order bit is 1, the key is down; otherwise, it is up."
	return (win32api.GetKeyState(key) & (1 << 7)) != 0

# char = ord(getch())
# print(char)
# key = ord('+')
key = win32con.VK_UP
keyUp = 56
q = 0x51
rightKey = 0x27

while True:
	if isKeyPressed(key):
		waitTime = 0.3
		print('KeyPress detected, timeWait', waitTime)
		time.sleep(0.5)
		win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,0,0)
		win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,0,0)
		time.sleep(waitTime)
		#win32api.keybd_event(rightKey,0,0 | 2,0)
		win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEDOWN,0,0)
		win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEUP,0,0)
		time.sleep(waitTime * 0.1)
		#win32api.keybd_event(rightKey,0,1 | 2,0)
		time.sleep(1.2 - waitTime)
		win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
		win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
	
	time.sleep(0.01)