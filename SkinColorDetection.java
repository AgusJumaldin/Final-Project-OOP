import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;
import java.util.*;

public class SkinColorDetection {
    public static void main(String[] args) {
        // Load OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load Haar Cascade for face detection
        String cascadePath = "haarcascade_frontalface_default.xml"; // Ensure this file exists
        CascadeClassifier faceDetector = new CascadeClassifier(cascadePath);

        // Open the webcam
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println("Error: Camera not found!");
            return;
        }

        Mat frame = new Mat();
        while (true) {
            // Capture frame from webcam
            camera.read(frame);
            if (frame.empty()) {
                System.out.println("Error: Empty frame captured!");
                break;
            }

            // Convert to grayscale for face detection
            Mat gray = new Mat();
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

            // Detect faces
            MatOfRect faces = new MatOfRect();
            faceDetector.detectMultiScale(gray, faces, 1.1, 4, 0, new Size(100, 100), new Size());

            // Loop through detected faces
            for (Rect rect : faces.toArray()) {
                // Draw rectangle around the face
                Imgproc.rectangle(frame, rect, new Scalar(0, 255, 0), 3);

                // Extract the face region
                Mat faceROI = frame.submat(rect);

                // Convert face region to YCrCb color space
                Mat ycrcb = new Mat();
                Imgproc.cvtColor(faceROI, ycrcb, Imgproc.COLOR_BGR2YCrCb);

                // Define skin color range in YCrCb
                Scalar lowerYCrCb = new Scalar(0, 133, 77);
                Scalar upperYCrCb = new Scalar(255, 173, 127);

                // Create a mask to detect skin pixels
                Mat skinMask = new Mat();
                Core.inRange(ycrcb, lowerYCrCb, upperYCrCb, skinMask);

                // Compute the average skin color
                Scalar avgSkinColor = Core.mean(faceROI, skinMask);

                // Determine the skin undertone
                String undertone = detectUndertone(avgSkinColor);

                // Get color recommendations
                String recommendedColors = getColorRecommendation(undertone);

                // Display the detected skin tone and color suggestions
                Imgproc.putText(frame, "Skin Tone: " + undertone, new Point(rect.x, rect.y - 30),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(255, 255, 255), 2);
                Imgproc.putText(frame, "Colors: " + recommendedColors, new Point(rect.x, rect.y - 10),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(255, 255, 255), 2);
            }

            // Show the frame with detected faces, skin tone, and recommendations
            HighGui.imshow("Real-Time Skin Tone & Color Recommendation", frame);

            // Exit on 'ESC' key press
            if (HighGui.waitKey(30) == 27) {
                break;
            }
        }

        // Release resources
        camera.release();
        HighGui.destroyAllWindows();
    }

    // Function to classify skin undertone based on average color
    private static String detectUndertone(Scalar avgColor) {
        double r = avgColor.val[2]; // Red component in BGR
        double g = avgColor.val[1]; // Green component
        double b = avgColor.val[0]; // Blue component

        // Convert to YCrCb color space for better skin tone analysis
        double y = 0.299 * r + 0.587 * g + 0.114 * b;
        double cr = (r - y) * 0.713 + 128; // Red chrominance
        double cb = (b - y) * 0.564 + 128; // Blue chrominance

        // Adjusted thresholds for better undertone detection
        if (cr >= 150) return "Warm";      // Higher Cr (more red) → Warm Undertone
        else if (cr <= 130) return "Cool";  // Lower Cr (less red, more blue) → Cool Undertone
        else return "Neutral";              // Middle range → Neutral Undertone
    }


    // Function to get color recommendations based on undertone
    private static String getColorRecommendation(String undertone) {
        switch (undertone) {
            case "Cool":
                return "Blues, Purples, Emerald Green";
            case "Warm":
                return "Earthy Tones, Coral, Gold";
            case "Neutral":
                return "Jewel Tones, Mauve, Charcoal";
            default:
                return "Any color!";
        }
    }
}
