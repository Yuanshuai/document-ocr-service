import cv2
import pyzbar
from pyzbar.pyzbar import decode


def identify_qrcode(img_input):
    img = cv2.imread(img_input) if isinstance(img_input, str) else img_input.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    decoded = decode(gray, symbols=[pyzbar.pyzbar.ZBarSymbol.QRCODE])
    # for d in decoded:
    #     print(d.data.decode())
    if decoded:
        for obj in decoded:
            rect = obj.rect
            cv2.rectangle(img, (rect.left, rect.top), (rect.left + rect.width, rect.top + rect.height), (255, 255, 255),
                          -1)
    else:
        return img
    return img
