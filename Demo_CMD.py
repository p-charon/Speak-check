from __future__ import print_function
import json
import six
import paddlehub as hub
import sys

test_text = sys.argv[1]

def speak_check(test_text):
    porn_detection_lstm = hub.Module(name="porn_detection_lstm")
    input_dict = {"text": test_text}
    results = porn_detection_lstm.detection(data=input_dict,use_gpu=True, batch_size=1)
    for index, text in enumerate(test_text):
        results[index]["text"] = text
    for index, result in enumerate(results):
        if six.PY2:
            print(
                json.dumps(results[index], encoding="utf8", ensure_ascii=False))
        else:
            #print(results[index])
            print("您的发言为:",test_text[0])
    lst = list(results[0].values())
    if lst[1]==1:
        print("您的发言涉嫌违规，已被屏蔽！")
    else:
        print("已发送！")

speak_check([test_text])