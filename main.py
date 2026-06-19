import torch
from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline
from diffusers.utils.export_utils import export_to_video
import cv2
def main():
    pipe = CogVideoXPipeline.from_pretrained(
     r"./models--THUDM--CogVideoX-2b/snapshots/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01",
    torch_dtype=torch.float16,
    local_files_only=True,
    )
    pipe.enable_model_cpu_offload()
    video = pipe(
        prompt="A cute small dog, anime style, warm indoor lighting, detailed illustration, cozy living room",
        num_frames=40
    ).frames[0] # type: ignore
    export_to_video(video, "demo.mp4", fps=4) # type: ignore
    print(torch.__version__)
    print("Hello from videogen!")
    cap = cv2.VideoCapture("demo.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video", frame)
        cv2.waitKey(250)

if __name__ == "__main__":
    main()


    
