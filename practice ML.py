import cv2
import numpy as np

# Load the image
image = cv2.imread("F:\\WhatsApp Image 2025-04-08 at 01.43.04_f0004115.jpg")
resized_image = cv2.resize(image, (500, 650))

# Apply filters
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(resized_image, (11, 11), 0)
edges = cv2.Canny(resized_image, 100, 200)
negative = cv2.bitwise_not(resized_image)

# Dictionary to hold all filters
filtered_images = {
    "Negative": negative,
    "Edges": edges,
    "Original": resized_image,
    "Blurred": blur,
    "Grayscale": gray,
}

# Show one by one
for title, img in filtered_images.items():
    cv2.imshow(title, img)
    print(f"Showing: {title}")
    cv2.waitKey(0)  # Wait for any key press
    cv2.destroyAllWindows()  # Close the current window before next

cv2.imwrite("gray_image.jpg", gray)
cv2.imwrite("blurred_image.jpg", blur)
cv2.imwrite("edges_image.jpg", edges)
cv2.imwrite("negative_image.jpg", negative)

