import pyautogui
import time
import pyperclip
from tqdm import tqdm

time.sleep(10)

pyautogui.PAUSE = 0.06

with open("result.txt", "r", encoding="utf-8") as f:
    for command in tqdm(f):
        pyperclip.copy(command)
        pyautogui.press('t')
        pyautogui.hotkey('ctrl', 'v')
        pyautogui.press('enter')

print("Done")