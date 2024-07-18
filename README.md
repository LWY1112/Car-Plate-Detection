# Car Object Detection and License Plate Recognition

This Python script utilizes YOLOv3 for car object detection and EasyOCR for license plate recognition. It processes images from an input directory, detects cars, extracts license plates, and verifies them against a predefined list of authorized plates.

## Installation

1. Clone the repository:

2. Install dependencies:
-pip install -r requirements.txt


## Usage

- Ensure your input directory (`Car Image/`) contains images of cars with visible license plates.
- Run the script:

## License Plate Authorization

The script uses a predefined dictionary (`allowed_plates`) to authorize known license plates. To authorize a new license plate, edit the `allowed_plates` dictionary in `carmain.py` and add a new entry.

## Outputs

- Detected objects (cars) are displayed with bounding boxes.
- Recognized license plates are extracted and verified against the `allowed_plates` dictionary.
- Unauthorized plates trigger a denial message.

## File Structure

- `carmain.py`: Main script for car object detection and license plate recognition.
- `model/`: Directory containing YOLOv3 configuration (`darknet-yolov3.cfg`) and weights (`model.weights`).
- `Car Image/`: Directory for input images of cars with visible license plates.

## Credits

- **YOLOv3:** Object detection model.
- **EasyOCR:** Optical Character Recognition library.

## Contributing

Contributions are welcome! Feel free to open issues or pull requests for improvements or bug fixes.

## Contact

For questions or suggestions, contact [your@email.com](mailto:your@email.com).

## References

- (https://www.youtube.com/watch?v=73REqZM1Fy0&list=PLb49csYFtO2Fh9gPJsJMu1PciV_6rAlj2&index=15)
