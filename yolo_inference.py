from ultralytics import YOLO 

model = YOLO('d:/major 2/models/last.pt')

results = model.predict('d:/major 2/input_videos/match.mp4',save=True)
print(results[0])

for box in results[0].boxes:
    print(box)