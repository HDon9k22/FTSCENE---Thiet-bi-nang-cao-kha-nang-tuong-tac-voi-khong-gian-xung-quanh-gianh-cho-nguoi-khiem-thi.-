import speech_recognition as sr

r = sr.Recognizer()
print("�ang ki?m tra micro...")

# In ra danh s�ch micro
for i, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"{i}: {name}")

index = int(input("Nh?p s? micro b?n mu?n test: "))
with sr.Microphone(device_index=index) as source:
    print("N�i g? �� b?ng ti?ng Vi?t...")
    audio = r.listen(source)
    print("�ang nh?n d?ng...")

try:
    text = r.recognize_google(audio, language="vi-VN")
    print("B?n v?a n�i:", text)
except Exception as e:
    print("Kh�ng nh?n ��?c gi?ng:", e)
