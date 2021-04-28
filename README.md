# File Descriptions

## Input Images
- **R/G/B.jpg**: Originally photographed images of the letters
- **R/G/B-small.jpg**: Shrinked photographed images of the letters
- **R/G/B-smallest.jpg**: More shrinked photographed images of the letters

*R-inked/-small/-smallest.jpg* is the photo containing "R" but with intentionally added noise.
<br /><br />

## Image-input Programs
- **LetterDetectorWithOpenCV.py:** Letter detection algorithim on an input image using OpenCV. This was the first program I wrote.
- **LetterDetectionWithBlobCounting.py:** Similar to "LetterDetectorWithOpenCV.py" but instead of using OpenCV's contour counting, I successfully implemented blob counting. Additional small revisions, such as adding more comments, are also included.
- **LetterDetectionWithBlobCountingAndBlobCleaning.py:** Similar to "LetterDetectionWithBlobCounting.py," but instead of using OpenCV's contour area calculations to filter out noise, I unsuccessfully tried to filter out noise by eliminating all blobs except the largest one (the letter). If I had more time to debug and look up documentation/Python rules, and if this was in C++, I would have pursed to implement it successfully.
<br /><br />

## Video-input Programs
- **LetterDetectorWithOpenCV - Video.py:** Similar to "LetterDetectorWithOpenCV.py," but with video input instead of image input. Successfully tested with 30 FPS.
- **LetterDetectorWithBlobCounting - Video.py:** Similar to "LetterDetectionWithBlobCounting.py," but with video input instead of image input. Because of algorithmic inefficiency, frame rate has to be really low.
<br /><br />

## Output Image Folders
- **B/G/R-inked-small with Blob Counting:** Output images at each stage of the letter detection algorithim when "B/G/R-small.jpg" is fed into "LetterDetectionWithBlobCounting.py."
<br /><br />

## Miscellaneous
- **Recursion Issue Error.png:** Memory Error when I implemented a DFS-approach for blob purging in my "LetterDetectionWithBlobCountingAndBlobCleaning.py" with a large image as input.
- **RGB Letters.docx:** Word document with which I used to print the letters I photographed.
<br /><br />

### *Presentation Link: https://docs.google.com/presentation/d/1OiV0zqMdQdK8OEd8cOdTDSG8wcabW4MM8qVu2mUQaaA/edit?usp=sharing*