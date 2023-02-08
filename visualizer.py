import os
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from metrics import segmentation_intersection_over_union


class Visualizer:
    def __init__(self, mode, task, class_probability_threshold, confidence_threshold, segmentation_background_class):
        self._task = task
        self._class_probability_threshold = class_probability_threshold
        self._confidence_threshold = confidence_threshold
        self._segmentation_background_class = segmentation_background_class

        base_path = os.path.dirname(__file__)
        self._learning_curves_path = os.path.join(base_path, 'figures', f'{mode}_{self._task}_learning_curves.png')
        self._prediction_path = os.path.join(base_path, 'figures', f'{mode}_{self._task}_prediction.png')

    def show_learning_curves(self, epochs_train_losses, epochs_validation_losses,
                             epochs_train_metrics, epochs_validation_metrics, metric_name):
        fig = plt.figure(figsize=(10, 5), dpi=300)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.plot(range(1, len(epochs_train_losses) + 1), epochs_train_losses, color='blue', label='Training',
                 linestyle=':')
        ax1.plot(range(1, len(epochs_validation_losses) + 1), epochs_validation_losses, color='red', label='Validation',
                 linestyle='-.')
        ax1.set_title(u'Loss')
        ax1.set_xlabel(u'Epoch')
        ax1.set_ylabel(u'Loss')
        ax1.legend()

        ax2.plot(range(1, len(epochs_train_metrics) + 1), epochs_train_metrics, color='blue', label='Training',
                 linestyle=':')
        ax2.plot(range(1, len(epochs_validation_metrics) + 1), epochs_validation_metrics, color='red',
                 label='Validation', linestyle='-.')
        ax2.set_title(metric_name)
        ax2.set_xlabel(u'Epoch')
        ax2.set_ylabel(metric_name)
        ax2.legend()

        fig.savefig(self._learning_curves_path)
        plt.close(fig)

    def show_prediction(self, image, prediction, segmentation_target, boxes, class_labels):
        image = image.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        segmentation_target = segmentation_target.cpu().detach().numpy()
        boxes = boxes.cpu().detach().numpy()
        class_labels = class_labels.cpu().detach().numpy()

        if self._task == 'classification':
            self.show_classification_prediction(image, prediction, class_labels)
        elif self._task == 'detection':
            self.show_detection_prediction(image, prediction, boxes)
        elif self._task == 'segmentation':
            self.show_segmentation_prediction(image, prediction, segmentation_target)
        else:
            raise ValueError('Not supported task')

    def show_classification_prediction(self, image, prediction, class_labels):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

        predicted_shapes = ''
        if prediction[0] >= self._class_probability_threshold:
            predicted_shapes += 'Circle '
        if prediction[1] >= self._class_probability_threshold:
            predicted_shapes += 'Triangle '
        if prediction[2] >= self._class_probability_threshold:
            predicted_shapes += 'Cross '

        target_shapes = ''
        if class_labels[0] >= self._class_probability_threshold:
            target_shapes += 'Circle '
        if class_labels[1] >= self._class_probability_threshold:
            target_shapes += 'Triangle '
        if class_labels[2] >= self._class_probability_threshold:
            target_shapes += 'Cross '

        ax.set_title(f'Prediction: {predicted_shapes}\nTarget: {target_shapes}')
        ax.imshow(image[0], cmap='gray', vmax=1)

        fig.savefig(self._prediction_path)
        plt.close(fig)

    def show_detection_prediction(self, image, prediction, target):
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.imshow(image[0], cmap='gray', vmax=1)
        ax1.set_title('Groundtruth')
        ax2.imshow(image[0], cmap='gray', vmax=1)
        ax2.set_title('Prediction')
        custom_lines = [Line2D([0], [0], color='r', lw=2),
                        Line2D([0], [0], color='g', lw=2),
                        Line2D([0], [0], color='b', lw=2)]
        ax2.legend(custom_lines, ['Circle', 'Triangle', 'Cross'], loc='center left', bbox_to_anchor=(1, 0.5))

        color = ['r', 'g', 'b']
        for i in range(target.shape[0]):
            pos_x = target[i, 1] * image[0].shape[0]
            pos_y = target[i, 2] * image[0].shape[0]
            size = target[i, 3] * int(0.75 * image[0].shape[0] / 2)
            class_index = int(target[i, 4])
            rec = patches.RegularPolygon((pos_x, pos_y), 4, orientation=0.78, radius=size, linewidth=2,
                                         edgecolor=color[class_index], facecolor='none')
            ax1.add_patch(rec)

        for i in range(prediction.shape[0]):
            if prediction[i, 0] > self._confidence_threshold:
                pos_x = prediction[i, 1] * image[0].shape[0]
                pos_y = prediction[i, 2] * image[0].shape[0]
                size = prediction[i, 3] * int(0.75 * image[0].shape[0] / 2)
                class_index = int(np.argmax(prediction[i, 4:]))
                rec = patches.RegularPolygon((pos_x, pos_y), 4, orientation=0.78, radius=size, linewidth=2,
                                             edgecolor=color[class_index], facecolor='none')
                ax2.add_patch(rec)

        fig.savefig(self._prediction_path)
        plt.close(fig)

    def show_segmentation_prediction(self, image, prediction, target):
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        epsilon = sys.float_info.epsilon
        intersection, union = segmentation_intersection_over_union(prediction[None, :, :, :],
                                                                   target[None, :, :],
                                                                   self._segmentation_background_class)
        intersection_over_union = (intersection + epsilon) / (union + epsilon)

        n_class = prediction.shape[0]
        prediction = np.argmax(prediction, axis=0)
        a = np.concatenate((prediction, target), axis=1)
        ax1.imshow(image[0], cmap='gray', vmax=1)
        ax1.set_title('Input image')
        ax2.imshow(a, vmax=n_class - 1, vmin=0)
        ax2.set_title('Prediction / Groundtruth')
        ax2.set_xlabel(f'IOU : {intersection_over_union}')

        fig.savefig(self._prediction_path)
        plt.close(fig)
