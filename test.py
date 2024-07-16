from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os
import warnings

warnings.filterwarnings("ignore")

# Load model
model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")

# Set parameters
TEXT_PROMPT = "license plate"
BOX_THRESHOLDS = [0.1, 0.3, 0.5]
TEXT_THRESHOLD = 0.10

# Folder containing images
IMAGE_FOLDER = "../../../../ssd_scratch/furqan/Datasets/IDD_Segmentation/leftImg8bit"
ANNOTATED_FOLDER = "../../../../ssd_scratch/IDD_Privacy/annotated_images/"
BOX_COORDINATES_FOLDER = "../../../../ssd_scratch/IDD_Privacy/box_coordinates/"

# Create output folders for box coordinates if they don't exist
for threshold in BOX_THRESHOLDS:
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(ANNOTATED_FOLDER, f"box_{threshold}", split), exist_ok=True)
        os.makedirs(os.path.join(BOX_COORDINATES_FOLDER, f"box_{threshold}", split), exist_ok=True)

# Iterate over all image files in the folder
for threshold in BOX_THRESHOLDS:
    for split in ["train", "val", "test"]:
        split_image_folder = os.path.join(IMAGE_FOLDER, split)
        split_annotated_folder = os.path.join(ANNOTATED_FOLDER, f"box_{threshold}", split)
        split_box_coordinates_folder = os.path.join(BOX_COORDINATES_FOLDER, f"box_{threshold}", split)

        for subdir in os.listdir(split_image_folder):
            subdir_path = os.path.join(split_image_folder, subdir)
            if os.path.isdir(subdir_path):
                # Ensure subdirectories exist in the annotated and box coordinates folders
                os.makedirs(os.path.join(split_annotated_folder, subdir), exist_ok=True)
                os.makedirs(os.path.join(split_box_coordinates_folder, subdir), exist_ok=True)

                for filename in os.listdir(subdir_path):
                    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Check if file is an image
                        # Load image
                        image_path = os.path.join(subdir_path, filename)
                        image_source, image = load_image(image_path)

                        # Predict
                        boxes, logits, phrases = predict(
                            model=model,
                            image=image,
                            caption=TEXT_PROMPT,
                            box_threshold=threshold,
                            text_threshold=TEXT_THRESHOLD,
                            device="cuda"  # Change to "cuda" if using GPU
                        )

                        if len(boxes) > 0:  # Check if there are any detected boxes
                            # Save box coordinates to txt file
                            coordinates_filename = f"box_coordinates_{filename}.txt"
                            coordinates_path = os.path.join(split_box_coordinates_folder, subdir, coordinates_filename)
                            with open(coordinates_path, "w") as txt_file:
                                for box in boxes:
                                    x_min, y_min, x_max, y_max = box
                                    txt_file.write(f"x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}\n")

                            # Annotate image
                            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

                            # Save annotated image
                            annotated_path = os.path.join(split_annotated_folder, subdir, f"annotated_{filename}")
                            cv2.imwrite(annotated_path, annotated_frame)

            print(f"Box Threshold: {threshold} for {subdir_path} data is saved")