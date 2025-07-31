
threshold = 0.4
scores = []
with open("regress_anno.txt", "r") as f:
    for line in f:
        video_id, score = line.strip().split()
        scores.append((video_id, float(score)))

scores = sorted(scores, key=lambda x:x[1])

low_quality = scores[:int(len(scores) * threshold)] 
high_quality = scores[-int(len(scores) * threshold):]

with open("classify_anno.txt", "w") as f:
    for video_id, score in low_quality:
        f.write(f"{video_id} 0\n")
    for video_id, score in high_quality:
        f.write(f"{video_id} 1\n")