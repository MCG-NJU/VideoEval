
# VidTAB

## Action Recognition in Dark

You could download all videos from ARID at https://opendatalab.com/OpenDataLab/Action_Recognition_in_the_Dark.

You just need to use the mp4 video in the video folder and then use the [annotations](https://github.com/leexinhao/VideoEval/tree/main/VidTAB/annotations/AR_in_Dark) we provided.

## Action Recognition in Long Video

You could download all videos from Breakfast at https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/.

You just need to use the mp4 video in the video folder and then use the [annotations](https://github.com/leexinhao/VideoEval/tree/main/VidTAB/annotations/AR_in_Long) we provided.

## Medical Surgery
You could download all videos from SurgicalActions160 at http://ftp.itec.aau.at/datasets/SurgicalActions160/index.html. 

You just need to use the mp4 video in the video folder and then use the [annotations](https://github.com/leexinhao/VideoEval/tree/main/VidTAB/annotations/Medical_Surgery)  we provided.

## Animal Behavior

You could download all videos from Animal Kingdom at https://forms.office.com/pages/responsepage.aspx?id=drd2NJDpck-5UGJImDFiPVRYpnTEMixKqPJ1FxwK6VZUQkNTSkRISTNORUI2TDBWMUpZTlQ5WUlaSyQlQCN0PWcu.

You just need to use the mp4 video in the video folder and then use the [annotations](https://github.com/leexinhao/VideoEval/tree/main/VidTAB/annotations/Animal_Behavior)  we provided.

## Harmful_Content

You could download all videos from MOB at https://drive.google.com/file/d/1Zjib-WaF5hk3wVrj5eW6ewdpMFcn45Wo/view.

Merge folders benign and malicious and then use the  [annotations](https://github.com/leexinhao/VideoEval/tree/main/VidTAB/annotations/Quality_Access) we provided.

## Fake Face

You could download all videos from FaceForensics++ at https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform?pli=1.

Then

```bash
cd yourpath/FaceForensics++
mkdir videos
mv faceforensics_videos/original_sequences/youtube/c23 videos/pos
mkdir videos/neg
python get_negs_samples.py
```

`get_negs_samples.py` is 

```python
import os
import shutil

video_list = os.listdir('videos/pos')
assert len(video_list) == 1000, len(video_list) 

for i in range(0, 1000):
    for method in ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]:
    	shutil.copy(f"faceforensics_videos/manipulated_sequences/{method}/c23/videos/{video_list[i]}", f"videos/neg/{video_list[i][:-4]}-{method}.mp4")
```

And then use the  [annotations](https://github.com/leexinhao/VideoEval/tree/main/VidTAB/annotations/Fake_Face) we provided.


## Emotion Analysis

You could download all videos from CAER at https://drive.google.com/file/d/1JsdbBkulkIOqrchyDnML2GEmuwi6E_d2/view

You just need to use the mp4 video in the video folder and then use the [annotations](https://github.com/leexinhao/VideoEval/tree/main/VidTAB/annotations/Emotion_Analysis) we provided.

## Quality Access

You could download all videos from DOVER at  https://huggingface.co/datasets/teowu/DIVIDE-MaxWell/resolve/main/videos.zip.

You just need to use the mp4 video in the video folder and then use the [annotations](https://github.com/leexinhao/VideoEval/tree/main/VidTAB/annotations/Quality_Access) we provided.

# VidEB

todo