"""Monomer/dimer classification of single-shot diffraction patterns.

A convolutional neural network labels each pnCCD frame as a single particle or
a pair, providing the morphology labels behind Fig. 3c and 3d and the confusion
matrix of Fig. S13.

Architecture (:func:`build_simplified_cnn`, the published model): a 7x7
convolution with 16 filters, batch normalization, ReLU, element-wise masking by
the beam stop, and 4x4 max-pooling; then a single residual block combining a
1x1 skip connection with a 5x5 convolution of 32 filters, batch normalization,
ReLU and spatial dropout; then global average pooling, a 64-unit dense layer
with L2 regularization, and a sigmoid output.

Note that :func:`build_masked_cnn`, elsewhere in the original notebook, is a
larger 3x3/32-64-128 variant that was trialled but not used for the published
classification.

Inputs are 200x200x2 tensors: the preprocessed frame, mirrored to a square
field and normalized to [0, 1], plus a fixed binary beam-stop mask so the
network learns only from informative pixels.

Evaluation uses dual decision thresholds (0.35 for monomer, 0.65 for dimer) to
reject ambiguous cases; see ``config/params.yaml`` for all hyperparameters.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def preprocess_image(image):
    """Mirror the top half of the image to create a complete diffraction pattern."""
    # The center of nanoparticle is at the center bottom, so we mirror the top half
    h, w = image.shape
    
    # Create a new square image (200×200)
    full_image = np.zeros((w, w), dtype=image.dtype)
    
    # Place the original image in the bottom half
    full_image[:w-h, :] = image
    
    # Mirror the bottom part to fill the top half
    # We only mirror up to half the height to avoid doubling the center
    full_image[w-h:, :] = np.flipud(image)
    
    return (full_image - np.min(full_image)) / (np.max(full_image) - np.min(full_image) + 1e-8)


def create_beamstop_mask(image):
    mask = np.ones(image.shape, dtype=int)
    
    x_start = 69
    x_end = 129
    y_start = 77
    y_end = 121
    
    mask[y_start:y_end, x_start:x_end] = 0
    return mask


def augment_img(image):
    
    flip_img = np.fliplr(image)
    
    intensities = np.random.uniform(0.4,2.5,4) 
    intensities_img = intensities[:, np.newaxis, np.newaxis] * [image]
    intensities_flip_img = intensities[:, np.newaxis, np.newaxis] * [flip_img]

    augmented_img = [image,flip_img]
    augmented_img.extend(intensities_img)
    augmented_img.extend(intensities_flip_img)

    return np.array(augmented_img)


def process_convert_augment(labels_trainIds, labeled_pnccd):

    augmented_images_cart = []
    # augmented_images_polar = []
    augmented_labels = []
    
    for label in labels_trainIds:
        
        img = labeled_pnccd[labeled_pnccd.trainId.isin(label.trainId)]
        full_img = preprocess_image(img.values[0])
        # polar_img = cartesian_projection(full_img)
    
        if label == 1:
            images_cart = augment_img(full_img)
            # images_polar = augment_img(polar_img)
            labels = [label] * len(images_cart)
            
        else:
            images_cart = np.array([full_img])
            # images_polar = np.array([polar_img])
            labels = [label]
    
        augmented_images_cart.extend(images_cart)
        # augmented_images_polar.extend(images_polar)
        augmented_labels.extend(labels)
    
    return np.array(augmented_images_cart), np.array(augmented_labels, dtype=int) #, np.array(augmented_images_polar)


def build_simplified_cnn(input_shape=(200, 200, 2)):
    
    inputs = layers.Input(shape=input_shape)
    image = layers.Lambda(lambda x: x[..., 0:1])(inputs)
    mask = layers.Lambda(lambda x: x[..., 1:2])(inputs)
    
    # Larger kernels capture more spatial context
    x = layers.Conv2D(16, (7, 7), padding='same')(image)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Multiply()([x, mask])
    x = layers.MaxPooling2D((4, 4))(x)  # Aggressive pooling
    
    # Single residual block
    residual = layers.Conv2D(32, (1, 1), padding='same')(x)
    x = layers.Conv2D(32, (5, 5), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(0.3)(x)
    x = layers.Add()([x, residual])
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Final layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs, outputs)


def evaluate_model_dual_threshold(model, X_test, y_test, class_names=['Monomer', 'Dimer'],
                                 n_examples=6, lower_threshold=0.45, upper_threshold=0.55):
    """
    Evaluate model with two thresholds, discard predictions in between.

    Args:
        model: Trained Keras model
        X_test: Test data, shape (N, H, W, 2)
        y_test: True labels, shape (N,)
        class_names: List of class names
        n_examples: Total number of examples to display
        lower_threshold: Lower classification threshold for 'monomer'
        upper_threshold: Upper classification threshold for 'dimer'
    """
    images = X_test[..., 0]
    masks = X_test[..., 1]
    y_pred_prob = model.predict(X_test).flatten()

    # Classify based on two thresholds
    y_pred = np.full_like(y_test, fill_value=-1)  # -1 means discard
    y_pred[y_pred_prob < lower_threshold] = 0  # Monomer
    y_pred[y_pred_prob > upper_threshold] = 1  # Dimer

    # Filter out discarded samples
    valid_indices = np.where(y_pred != -1)
    y_test_valid = y_test[valid_indices]
    y_pred_valid = y_pred[valid_indices]

    # Confusion matrix
    cm = confusion_matrix(y_test_valid, y_pred_valid)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(6, 6))
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix with Dual Thresholds')
    plt.show()

    print("\nClassification Report:")
    print(f"Lower Threshold: {lower_threshold:.2f}")
    print(f"Upper Threshold: {upper_threshold:.2f}")
    print(f"True Positives (Dimers): {cm[1,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"True Negatives (Monomers): {cm[0,0]}")

    # Find indices for each predicted class
    idx_pred_0 = np.where(y_pred == 0)
    idx_pred_1 = np.where(y_pred == 1)

    # Ensure at least one example from each class (if available)
    selected_indices = []
    if len(idx_pred_0) > 0:
        selected_indices.append(np.random.choice(idx_pred_0))
    if len(idx_pred_1) > 0:
        selected_indices.append(np.random.choice(idx_pred_1))

    # Fill up to n_examples with random choices from remaining valid indices
    remaining = list(set(valid_indices) - set(selected_indices))
    if len(remaining) > 0:
        n_to_pick = n_examples - len(selected_indices)
        if n_to_pick > 0:
            selected_indices += list(np.random.choice(remaining, size=min(n_to_pick, len(remaining)), replace=False))

    # Show the selected examples
    images_per_row = 5
    rows = n_examples // images_per_row
    plt.figure(figsize=(3*images_per_row, 4*rows))
    for i, idx in enumerate(selected_indices):
        img = images[idx]
        mask = masks[idx]
        true_label = class_names[y_test[idx]]
        pred_label = class_names[y_pred[idx]]
        prob = y_pred_prob[idx]
        masked_img = np.where(mask > 0.5, img, np.nan)

        plt.subplot(rows, images_per_row, i+1)
        plt.imshow(masked_img, cmap='viridis')
        plt.title(f"True: {true_label}\nPred: {pred_label}\n({prob:.2f})")
        plt.axis('off')
    plt.suptitle("Examples from Both Predicted Classes with Dual Thresholds")
    plt.tight_layout()
    plt.show()

    return cm, y_pred


def plot_threshold_analysis(y_test, y_pred_prob):

    prec, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    plt.figure()
    plt.plot(thresholds, prec[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    # plt.axvline(optimal_threshold, color='k', linestyle='--')
    plt.title("Precision-Recall vs Threshold")
    plt.legend()
    plt.show()
