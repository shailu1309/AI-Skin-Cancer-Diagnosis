
# AI-enabled Skin Cancer Diagnosis Assistant

This project focuses on building an **AI-based web application** that uses **deep learning** to classify various types of skin cancer lesions. It addresses the need for accurate, real-time, and early diagnosis of skin cancer, particularly focusing on the differentiation between melanoma and non-melanoma skin lesions. The backbone of this system is a **Convolutional Neural Network (CNN)**, trained to classify pigmented skin lesions into seven categories based on dermoscopic images. 

## Introduction

Skin cancer is among the most common types of cancer worldwide, with increasing numbers diagnosed every year. **Early diagnosis** can drastically improve survival rates. Traditionally, dermatologists rely on **dermoscopy** for diagnosis, which is time-consuming and heavily dependent on the expertise of the practitioner. This project uses **deep learning techniques** to assist medical professionals by providing automated classification of skin lesions, thereby offering a **quick and reliable second opinion**.

## Why Convolutional Neural Networks (CNN)?

The project places a heavy emphasis on **Convolutional Neural Networks (CNN)** over **Artificial Neural Networks (ANN)** due to the nature of the task—image classification. Traditional machine learning and ANN approaches depend heavily on **manual feature extraction** from images, which can be error-prone and time-consuming. CNNs are particularly suitable for this task because:
1. **Automatic Feature Extraction**: CNNs learn to extract hierarchical features directly from raw images without human intervention.
2. **Spatial Hierarchy**: The convolution operation allows CNNs to capture local patterns (like edges, textures) which are crucial for image analysis, while deeper layers capture more complex structures.
3. **Efficiency**: CNNs reduce the dimensionality of input images while maintaining relevant information through techniques like pooling, making them computationally efficient.

Compared to **ANNs**, which struggle with large image inputs, CNNs are much more suited to the high-dimensional nature of image data. This makes CNNs the preferred choice for skin cancer diagnosis, where subtle visual cues differentiate between cancerous and non-cancerous lesions.

---

## Project Structure

### Key Objectives:
1. Develop a robust **CNN-based classification model** capable of differentiating between seven types of skin lesions.
2. Build a **web application** that enables users to upload dermoscopic images for real-time automated diagnosis.
3. Use **transfer learning** with pre-trained models like **ResNet50**, **MobileNetV2**, and **EfficientNetV2** to improve the accuracy and reduce training time.

---

## Experiments and Results

### Comparison between CNN and ANN models:
| Model               | Feature Extraction | Image Input Size | Complexity | Accuracy | AUC-ROC |
|---------------------|--------------------|------------------|------------|----------|---------|
| **ANN**             | Manual             | Low              | Medium     | 70.23%   | 0.82    |
| **CNN**             | Automatic          | High (128x128)   | High       | 80.45%   | 0.95    |
| **CNN with Transfer Learning** | Automatic  | High (224x224)   | High       | 82.89%   | 0.96    |

The performance of CNN models clearly surpasses that of traditional ANNs, thanks to their ability to automatically learn and extract relevant features from raw image data.

### Experiments on CNN Architectures:

Several deep learning models were trained and fine-tuned using **transfer learning**. Below is the summary of experiments conducted:

| Experiment | Model Used                    | Input Size | Architecture             | Validation Accuracy | AUC-ROC | Top 2 Accuracy | Top 3 Accuracy |
|------------|-------------------------------|------------|--------------------------|---------------------|---------|----------------|----------------|
| 1          | CNN (Custom)                  | 128x128    | Simple CNN               | 68.5%               | 0.82    | 80.15%         | 90.34%         |
| 2          | CNN (Custom, Dropout Layers)  | 224x224    | CNN with Dropout         | 70.23%              | 0.84    | 82.1%          | 92.5%          |
| 3          | **ResNet50 (Transfer Learning)** | 224x224  | Pre-trained ResNet50     | 75.34%              | 0.85    | 80.12%         | 91.34%         |
| 4          | **EfficientNetV2 (Fine-tuned)** | 240x240  | Transfer Learning         | 80.45%              | 0.9517  | 87.99%         | 94.05%         |
| 5          | **MobileNetV2 (Fine-tuned)**  | 224x224    | Transfer Learning         | 77.89%              | 0.89    | 85.23%         | 92.78%         |
| 6          | ResNet50 (Fine-tuned)         | 224x224    | Transfer Learning         | 78.9%               | 0.9     | 86.54%         | 92.5%          |
| 7          | InceptionResNetV2 (Fine-tuned)| 299x299    | Transfer Learning         | 79.23%              | 0.92    | 85.45%         | 93.8%          |

### Model Evaluation:
- **EfficientNetV2** emerged as the best performing model with an accuracy of **80.45%** and **AUC-ROC score of 0.9517**. This model was fine-tuned from a pre-trained network, enabling it to generalize well on new skin lesion images.
- **MobileNetV2** and **ResNet50** also performed well, with **ResNet50** showing strong results in both its pre-trained and fine-tuned forms.

### Performance Metrics:
| Metric                | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| **Accuracy**          | Proportion of correct predictions out of all predictions made.               |
| **AUC-ROC**           | Area Under the Receiver Operating Characteristic curve for multi-class classification. |
| **Precision**         | Proportion of true positives out of all positive predictions.                |
| **Recall**            | Proportion of true positives out of all actual positives.                    |
| **F1 Score**          | Harmonic mean of precision and recall, balancing both.                       |
| **Top 2 Accuracy**    | Proportion of instances where the correct label is in the top 2 predictions. |
| **Top 3 Accuracy**    | Proportion of instances where the correct label is in the top 3 predictions. |

---

## Application Development

The web application is developed using **Flask** and deployed using **Heroku** for real-time diagnosis. It allows users to upload images of skin lesions, which are pre-processed and fed into the trained deep learning model for classification.

### Tools and Technologies Used:
| Tool       | Purpose                         |
|------------|----------------------------------|
| **Flask**  | Python-based web framework for building the app. |
| **Docker** | Containerization for seamless deployment across environments. |
| **Heroku** | Cloud platform for application deployment. |
| **Git**    | Version control for managing code. |

### Application Flow:
1. **Image Upload**: Users upload an image of a skin lesion.
2. **Preprocessing**: Image is resized and normalized to match the input requirements of the model.
3. **Prediction**: The model classifies the image into one of seven skin cancer types.
4. **Result Display**: The prediction result and confidence score are shown to the user.

### Flowchart:

```mermaid
graph TD;
    A[User Uploads Image] --> B[Image Processing];
    B --> C[Model Prediction];
    C --> D[Result Displayed with Confidence Score];
```

---

## Conclusion

This project showcases the effectiveness of **CNNs** for image classification tasks, particularly in the medical domain for skin cancer diagnosis. By combining deep learning techniques with a web-based platform, this tool demonstrates a potential real-world application for automated skin lesion diagnosis. The success of **EfficientNetV2** and other transfer learning models emphasizes the potential of **pre-trained models** in providing high accuracy with minimal training time. 

Further improvements in terms of dataset size, model optimization, and additional features like explainability could expand the scope and applicability of this solution.

---

## Deployment

The application is containerized using Docker and deployed on Heroku for scalability and ease of access.

---

## References

1. World Cancer Research Fund International. (2020). Skin Cancer Statistics.
2. Challenge ISIC Archive - https://challenge.isic-archive.com/
3. Tensorflow for Deep Learning (Models)

