import cv2
import os
from model.insightface import FaceAnalysis

class FaceDetection:
    def __init__(self):
        # Initialize the FaceAnalysis detector
        self.app = FaceAnalysis(allowed_modules=['detection'], name='buffalo_l', root='model/insightface/')
        self.app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.45)
        
    # Function to detect faces using RetinaFace
    def detect_faces(self, image_path, output_folder):
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read image {image_path}")
            return
        
        # Convert the image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.app.get(rgb_image)
        
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Draw rectangles around the detected faces
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
        
        # Extract the filename from the input image path
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, f'detected_{filename}')
        
        # Save the image with detected faces
        cv2.imwrite(output_path, image)
        
        print(f"Result saved as {output_path}")

    # Function to process all images in the input folder
    def process_images(self, input_folder, output_folder):
        # Get a list of all files in the input folder
        files = os.listdir(input_folder)
        for file in files:
            file_path = os.path.join(input_folder, file)
            # Check if the file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                print(f"Processing {file_path}")
                self.detect_faces(file_path, output_folder)
            else:
                print(f"Skipping non-image file: {file_path}")


if __name__ == "__main__":
    # Test the function with an example input folder and output folder
    input_folder = 'input'  # Replace with your input folder path
    output_folder = 'output'  # Replace with your desired output folder path
    FaceDetection().process_images(input_folder, output_folder)