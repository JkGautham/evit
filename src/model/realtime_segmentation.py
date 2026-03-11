import torch
import cv2
import numpy as np
from ref import EViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(r"E:/model_full.pth", map_location=device)
model = model.to(device)
model.eval()

# class labels
class_names = [
"background","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
"traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep",
"cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
"frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
"surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
"banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
"chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
"keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock",
"vase","scissors","teddy bear","hair drier","toothbrush"
]

colors = np.random.randint(0,255,(81,3),dtype=np.uint8)
colors[0] = [0,0,0]

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]

    # ---------- MODEL INPUT ----------
    img = cv2.resize(frame,(512,512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = img.transpose(2,0,1)

    img = torch.from_numpy(img).float().unsqueeze(0).to(device)

    # ---------- MODEL INFERENCE ----------
    with torch.no_grad():
        output = model(img)

    pred = torch.argmax(output,dim=1)
    mask_small = pred.squeeze().cpu().numpy()

    # ---------- SCALE AFTER MODEL ----------
    mask = cv2.resize(
        mask_small,
        (orig_w,orig_h),
        interpolation=cv2.INTER_NEAREST
    )

    # ---------- COLOR SEGMENTATION ----------
    mask_rgb = colors[mask]

    overlay = cv2.addWeighted(frame,0.7,mask_rgb,0.3,0)

    # ---------- BOUNDING BOXES ----------
    for cls in np.unique(mask):

        if cls == 0:
            continue

        class_mask = (mask == cls).astype(np.uint8)

        contours,_ = cv2.findContours(
            class_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:

            if cv2.contourArea(cnt) < 800:
                continue

            x,y,w,h = cv2.boundingRect(cnt)

            color = colors[cls].tolist()

            cv2.rectangle(overlay,(x,y),(x+w,y+h),color,2)

            cv2.putText(
                overlay,
                class_names[cls],
                (x,y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    cv2.imshow("EViT Real-time Segmentation", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()