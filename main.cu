#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

// TODO: pass as command line arguments
const std::string BASE_PATH = "";

std::string makePath(std::string filename) {
    return BASE_PATH + filename;
}

// TODO: move to utils
Point findMaxIntensityCircle(const Mat &dft_intensity, int radius) {
    Mat integralImage;
    integral(dft_intensity, integralImage);

    Point maxPoint;
    double maxSum = 0;

    int halfRadius = radius / 2;

    for (int i = radius; i < dft_intensity.rows - radius; ++i) {
        for (int j = radius; j < dft_intensity.cols - radius; ++j) {
            int x1 = j - halfRadius;
            int y1 = i - halfRadius;
            int x2 = j + halfRadius;
            int y2 = i + halfRadius;

            double sum = integralImage.at<double>(y2, x2) - integralImage.at<double>(y1, x2) -
                         integralImage.at<double>(y2, x1) + integralImage.at<double>(y1, x1);

            if (sum > maxSum) {
                maxSum = sum;
                maxPoint = Point(j, i);
            }
        }
    }

    return maxPoint;
}

Point findMaxIntensityCircleFromFFT(const Mat &dft, int radius, int windowSize) {
    // Find a mask for the excited state
    Mat dft_intensity;
    std::vector<Mat> planes;
    split(dft, planes);

    // Calculate magnitude
    magnitude(planes[0], planes[1], dft_intensity);

    // Switch to logarithmic scale and normalize
    dft_intensity += Scalar::all(1);
    log(dft_intensity, dft_intensity);
    normalize(dft_intensity, dft_intensity, 0, 1, NORM_MINMAX);

    // Draw a circle to null out the center
    circle(dft_intensity, Point(dft_intensity.cols / 2, dft_intensity.rows / 2), radius, Scalar(0), -1);

    // Find the maximum intensity point
    return findMaxIntensityCircle(dft_intensity, windowSize);
}

std::tuple<int, int> findCenterOfMask(const Mat &mask) {
    Mat maskI;
    extractChannel(mask, maskI, 0);
    Moments moments = cv::moments(maskI, true);
    int centerX = int(moments.m10 / moments.m00);
    int centerY = int(moments.m01 / moments.m00);

    return std::make_tuple(centerX, centerY);
}

Mat findTranslationToCenterMatrix(const Mat &dft, const std::tuple<int, int> centerPosition) {
    // Calculate the shift needed to center the visible part
    int shiftX = dft.cols / 2 - std::get<0>(centerPosition);
    int shiftY = dft.rows / 2 - std::get<1>(centerPosition);

    std::cout << shiftX << "\t" << shiftY << std::endl;

    // Create a translation matrix
    return (Mat_<float>(2, 3) << 1, 0, shiftX, 0, 1, shiftY);
}

void fftShift(Mat &magI) {
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;

    Mat q0(magI, Rect(0, 0, cx, cy));
    Mat q1(magI, Rect(cx, 0, cx, cy));
    Mat q2(magI, Rect(0, cy, cx, cy));
    Mat q3(magI, Rect(cx, cy, cx, cy));

    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void addOptimalPadding(Mat &mat) {
    int optimalRows = getOptimalDFTSize(mat.rows);
    int optimalCols = getOptimalDFTSize(mat.cols);
    Mat padded;
    copyMakeBorder(mat, padded, 0, optimalRows - mat.rows, 0,
                   optimalCols - mat.cols, BORDER_CONSTANT, Scalar(0));
}

// TODO: magnitude and phase can be extracted with one function call to cv::cartToPolar
void writeMagnitudeFromC2(Mat &mat, std::string filename) {
    std::vector<Mat> planes;
    split(mat, planes);

    // Calculate magnitude
    Mat magI;
    magnitude(planes[0], planes[1], magI);

    // Switch to logarithmic scale
    magI += Scalar::all(1);
    log(magI, magI);

    imwrite(makePath(filename + ".tif"), magI);
}

void writePhaseFromC2(Mat &mat, std::string filename) {
    std::vector<Mat> planes;
    split(mat, planes);

    Mat p;
    phase(planes[0], planes[1], p);

    Mat phase_towrite;
    p.convertTo(phase_towrite, CV_8U);

    imwrite(makePath(filename + ".tif"), p);
}

int main() {
    // TODO: move to config
    int windowSizePhaseRegion = 256; // Adjust the size of the square as needed
    int centerPhaseCircleRadius = 32;
    
    // Load images
    Mat reference_normal = imread( makePath("reference_normal.tif"), IMREAD_GRAYSCALE);
    Mat reference_excited = imread(makePath("reference_excited.tif"), IMREAD_GRAYSCALE);
    Mat excited = imread(makePath("excited.tif"), IMREAD_GRAYSCALE);

    if (reference_normal.empty() || reference_excited.empty() || excited.empty()) {
        std::cerr << "Error: Unable to load input images." << std::endl;
        return -1;
    }

    std::cout << "Loaded images with sizes: " << reference_normal.size() << ", " << reference_excited.size() << ", "
              << excited.size() << std::endl;

    // Check image dimensions and data types
    if (reference_normal.size() != reference_excited.size() || reference_normal.size() != excited.size() ||
        reference_normal.type() != CV_8U || reference_excited.type() != CV_8U || excited.type() != CV_8U) {
        std::cerr << "Error: Images have mismatched dimensions or incorrect data types." << std::endl;
        return -1;
    }

    addOptimalPadding(excited);

    // Perform Discrete Fourier Transform (DFT)
    Mat dft_object;
    dft(Mat_<float>(excited), dft_object, DFT_COMPLEX_OUTPUT);

    // Shift the quadrants of the DFT around the center
    fftShift(dft_object);

    // Plot the magnitude and phase of the DFT
    writeMagnitudeFromC2(dft_object, "dft_object-magnitude");
    writePhaseFromC2(dft_object, "dft_object-phase");


    // Find the maximum intensity point in the phase region
    Point maxIntensityPoint = findMaxIntensityCircleFromFFT(dft_object, centerPhaseCircleRadius, windowSizePhaseRegion);
    maxIntensityPoint.y -= 70; // TODO: need a better detection strategy

    // Draw the circle on a copy of the original image
    Mat mask(dft_object.size(), CV_8UC1, Scalar(0, 0));
    circle(mask, maxIntensityPoint, windowSizePhaseRegion / 2, Scalar(255), -1);


    // ------
    // Apply the mask to set everything outside the circle to 0
    Mat result;
    dft_object.copyTo(result, mask);

    auto centerPositionMask = findCenterOfMask(mask);
    Mat translationMatrix = findTranslationToCenterMatrix(dft_object, centerPositionMask);

    // Apply the translation to center the visible part
    warpAffine(result, result, translationMatrix, dft_object.size());

    writeMagnitudeFromC2(result, "before_ifft_shift-magnitude");

    // Shift back the quadrants of the DFT
    fftShift(result);
    writeMagnitudeFromC2(result, "before_ifft-magnitude");


    // Apply inverse DFT
    Mat inverseDFT;
    dft(result, inverseDFT, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
    normalize(inverseDFT, inverseDFT, 0, 1, NORM_MINMAX);


    namedWindow("Result", WINDOW_NORMAL);
    resizeWindow("Result", 1000, 1000);
    imshow("Result", inverseDFT);
    waitKey(0);

    // Save the results
//    imwrite(makePath("phase.tif"), phase);
    imwrite(makePath("result.tif"), inverseDFT);

    return 0;
}
