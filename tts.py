
from gtts import gTTS
import os
tts = gTTS(text='आइतोनिक्समा आज उपस्थित हुन पाउदा म निकै नै खुसि छु । ', lang='ne')
tts.save("good.mp3")
os.system("mpg123 good.mp3")