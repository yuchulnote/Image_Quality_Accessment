import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# Constants
MASK_THRESHOLD: int = 5
LOW_THRESHOLD: int = 50
HIGH_THRESHOLD: int = 230

INPUT_IMAGE_PATH: str = (
    "C:/Users/PETNOW/Desktop/ivan/Robert_ivan/result_his/dataset/output-positive"
)
STD_HAE_THRESHOLD: int = 50

OUTPUT_PATH = "C:/Users/PETNOW/Desktop/ivan/Robert_ivan/result_his/output/Positive"


def calculate_metrics(img, mask):
    mean = np.mean(img[mask])
    var = np.var(img[mask])
    std = np.std(img[mask])
    print("mean, var, std")
    print(mean, var, std)
    return mean, var, std


def apply_histogram_equalization(img, mask):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img[mask].reshape(-1, 1))  # Flatten the masked part, apply CLAHE
    cl1 = cl1.reshape(img[mask].shape)  # Reshape back to original size
    return cl1


def draw_graph(img, image_file):
    mask = img > MASK_THRESHOLD

    # Create an empty image filled with zeros
    equ = np.zeros_like(img)

    # Apply mask to flatten the image, apply histogram equalization, and then reshape it back
    flat_pixels = img[mask].flatten()
    equalized_pixels = cv2.equalizeHist(flat_pixels)
    equ[mask] = equalized_pixels.reshape(img[mask].shape)

    cl1 = apply_histogram_equalization(img, mask)
    cl2 = apply_histogram_equalization(equ, mask)

    mean, var, std = calculate_metrics(img, mask)
    std_hae = np.std(cl2)
    std_clahe = np.std(cl1)

    with open("./result.txt", mode="a") as f:
        print(
            f"{image_file},{mean},{var},{std},{std_hae},{std_clahe},{std_hae-std},{std_clahe-std}",
            file=f,
        )

    img_display = img.copy()
    equ_display = equ.copy()
    cl1_display = np.zeros_like(img)
    cl2_display = np.zeros_like(equ)
    cl1_display[mask] = cl1
    cl2_display[mask] = cl2

    display_images_and_histograms(
        img_display,
        equ_display,
        cl1_display,
        cl2_display,
        img,
        equ,
        cl1,
        cl2,
        mask,
        image_file,
    )
    return mean, var, std, std_hae


def display_images_and_histograms(
    img_display,
    equ_display,
    cl1_display,
    cl2_display,
    img,
    equ,
    cl1,
    cl2,
    mask,
    image_file,
):
    plt.figure(figsize=(12, 12))

    titles = [
        f"Original {os.path.basename(image_file)}",
        "Histogram Equalized Image",
        "CLAHE Image",
        "Adaptive Equalized Image",
    ]
    images = [img_display, equ_display, cl1_display, cl2_display]
    equalized_images = [img.ravel(), equ.ravel(), cl1, cl2]

    for i in range(4):
        plt.subplot(4, 2, 2 * i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])

        plt.subplot(4, 2, 2 * i + 2)
        plt.hist(equalized_images[i], bins=256)
        plt.title(f"Histogram of {titles[i]}")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/processed_{os.path.basename(image_file)}")
    # plt.show()
    plt.close()


def show_hist_v(img_path):
    img = cv2.imread(img_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    mask = v > 5  # create a mask where non-black pixels (in value channel) are
    v = v[mask]  # apply the mask
    histr = cv2.calcHist([v], [0], None, [256], [0, 256])

    # Plot the histogram and the image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image")
    ax2.plot(histr)
    ax2.set_title("Value Histogram")

    # Save the figure
    output_path = f"./shadow-result/Positive/{os.path.basename(img_path)}"
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    total_pixels = np.count_nonzero(mask)  # count total non-black pixels
    low_threshold = np.count_nonzero(v < 50)
    high_threshold = np.count_nonzero(v > 230)
    percenet_low = low_threshold / total_pixels * 100
    percenet_high = high_threshold / total_pixels * 100
    max_Value = np.max(v)

    print(
        "Total Pixels - {}\n Pixels More than High Threshold - {} \n Pixels Less than Low Threshold - {} \n Pixels percentage less than Low Threshold - {} \n Pixel percentage more than High Threshold - {} \n".format(
            total_pixels, high_threshold, low_threshold, percenet_low, percenet_high
        )
    )

    return total_pixels, high_threshold, low_threshold, percenet_low, percenet_high


def process_images(image_list):
    rf = open("./logic-result.txt", mode="w")
    print("image_file,is_shadow,is_glare", file=rf)
    dataset_size = len(image_list)
    tp = 0
    glare_tp = 0
    for image_file in image_list:
        is_shadow = True
        is_glare = True

        print(os.path.basename(image_file))
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        mean, var, std, std_hae = draw_graph(image, image_file)
        print("std hae : ", std_hae)
        (
            total_pixels,
            high_threshold,
            low_threshold,
            percent_low,
            percent_high,
        ) = show_hist_v(image_file)

        if std_hae > 55 and percent_low > 30:
            tp += 1
            print("SHADOW")
            is_shadow = True
        else:
            is_shadow = False
            print("NOT SHADOW")

        if std_hae > 65 and percent_high > 0.1 and percent_low > 40:
            print("GLARE")
            is_glare = True
            glare_tp += 1
        else:
            print("NOT GLARE")
            is_glare = False
        print(f"{os.path.basename(image_file)},{is_shadow},{is_glare}", file=rf)

        print("-" * 100)
    if dataset_size == 0:
        print(image_list)
        print("dataset_size is zero")
    else:
        print(
            f"TP: {tp}, FP: {dataset_size - tp}, Total: {dataset_size}, Accuracy: {tp/dataset_size}"
        )
        print(
            f"TP: {glare_tp}, FP: {dataset_size - glare_tp}, Total: {dataset_size}, Accuracy: {glare_tp/dataset_size}"
        )


image_list = glob.glob(f"{INPUT_IMAGE_PATH}/**")
process_images(image_list)

# Initialize result.txt
with open("./result.txt", mode="w") as f:
    print(
        f"image_file,mean,variance,std,std(hae),std(clahe),std(hae)-std,std(clahe)-std",
        file=f,
    )
