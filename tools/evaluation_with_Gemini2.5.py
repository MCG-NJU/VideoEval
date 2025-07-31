import json
import os
import os
import random
import time
import json
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from google.genai.types import HttpOptions
from loguru import logger
from google.genai.errors import ClientError


TIMEOUT = 10 * 60 * 1000 # in milliseconds
client = genai.Client(
    http_options=HttpOptions(timeout=TIMEOUT)
)
anno_file = "VidTAB/annotations/Animal_Behavior/val.txt"
video_root = "Animal_Behavior"

label_dict = {9: 'Chirping', 10: 'Include feeding', 4: 'Animal makes no or minimal movement (i.e.', 6: 'Animal grooms (e.g. licks fur) itself\n', 2: 'Flying', 11: 'Animal makes large jumping movement from one spot to another (e.g. from lower to higher grounds)', 0: 'Animal moves from one spot to another. Include insects crawling. Include behaviours that cannot be described in other locomotive terms in the Movement category. \n', 7: 'Running', 1: 'Animal swims in the water (e.g. fish)', 
8: 'Animal moves from one spot to another in a slow pace\n', 5: 'Animal locates a stimulus of potential interest',
 3: 'Different from attending. Attention to the stimulus / stimuli is not fixated. Animal may continuously move its head to scan its surrounding\n'}

gemini_type = 'flash'
retry_times = 20
sleep_time_base = 5

with open(f"Animal_Behavior/result_gemini_{gemini_type}.jsonl", "w") as fw:
    with open(anno_file, "r") as f:
        for line in f:
            video_name = line.strip().split()[0]
            answer = label_dict[int(line.strip().split()[1])]
            video_file_name = os.path.join(video_root, video_name)
            video_bytes = open(video_file_name, 'rb').read()

            _retry_times = 0
            while _retry_times < retry_times:
                try:
                    response = client.models.generate_content(
                        model=f'gemini-2.5-{gemini_type}',
                        contents=types.Content(
                            parts=[
                                types.Part(
                                    inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                                ),
                                types.Part(text=f"Classify this video, select from {list(label_dict.values())}, only give one answer!")
                            ]
                        )
                    )
                    assert response is not None
                    print(f"{video_name}<|>{answer}<|>{response.text}\n")
                    fw.write(f"{video_name}<|>{answer}<|>{response.text}\n") #{response.text}
                    break
                except Exception as e:
                    print(e)
                    if isinstance(e, ClientError):
                        err_details = e.details['error']
                        if err_details['code'] == 429 and err_details['details'][0]['violations'][0]['quotaId'] == 'GenerateRequestsPerDayPerProjectPerModel':
                            os._exit(0)
                    sleep_time = sleep_time_base * random.random() * 2 + sleep_time_base
                    time.sleep(sleep_time)
                    logger.error(f'{video_name} error: {e}, sleep {sleep_time} seconds')
                    _retry_times += 1
                time.sleep(1)
            time.sleep(1)