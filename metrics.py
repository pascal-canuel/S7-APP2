import sys

import numpy as np


class Metric:
    def get_name(self):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()

    def accumulate(self, prediction, target):
        raise NotImplementedError()

    def get_value(self):
        raise NotImplementedError()


class AccuracyMetric(Metric):
    def __init__(self, class_probability_threshold):
        self.class_probability_threshold = class_probability_threshold

        self._good_count = 0.0
        self._total_count = 0.0

    def get_name(self):
        return 'Accuracy'

    def clear(self):
        self._good_count = 0.0
        self._total_count = 0.0

    def accumulate(self, prediction, target):
        """
        Méthode qui accumule les métriques d'un lot de données.
        N: La taille du lot (batch size)

        :param prediction: Le tenseur PyTorch de prédiction des classes des images
            Dimensions : (N, 3)
                (i, 0) : probabilité qu'un cercle est présent dans l'image i
                (i, 1) : probabilité qu'un triangle est présent dans l'image i
                (i, 2) : probabilité qu'une croix est présente dans l'image i
        :param target: Le tenseur PyTorch cible pour la tâche de classification
            Dimensions : (N, 3)
                Si un 1 est présent à (i, 0), un cercle est présent dans l'image i.
                Si un 0 est présent à (i, 0), aucun cercle n'est présent dans l'image i.
                Si un 1 est présent à (i, 1), un triangle est présent dans l'image i.
                Si un 0 est présent à (i, 1), aucun triangle n'est présent dans l'image i.
                Si un 1 est présent à (i, 2), une croix est présente dans l'image i.
                Si un 0 est présent à (i, 2), aucune croix n'est présente dans l'image i.
        """
        prediction = prediction.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        prediction = (prediction > self.class_probability_threshold).astype(float)
        self._good_count += (prediction == target).sum()
        self._total_count += prediction.size

    def get_value(self):
        return self._good_count / self._total_count


class MeanAveragePrecisionMetric(Metric):
    def __init__(self, class_count, intersection_over_union_threshold):
        self._class_count = class_count
        self._intersection_over_union_threshold = intersection_over_union_threshold

        self._target_count_by_class = [0 for _ in range(self._class_count)]
        self._results_by_class = [[] for _ in range(self._class_count)]

    def get_name(self):
        return 'mAP'

    def clear(self):
        self._target_count_by_class = [0 for _ in range(self._class_count)]
        self._results_by_class = [[] for _ in range(self._class_count)]

    def accumulate(self, prediction, target):
        """
        Méthode qui accumule les métriques d'un lot de données.
        N: La taille du lot (batch size)
        M: Nombre de prédictions fait par le modèle

        :param prediction:
            Dimensions : (N, M, 7)
                (i, j, 0) indique le niveau de confiance entre 0 et 1 qu'un vrai objet est représenté par le vecteur (n, m, :)
                Si (i, j, 0) est plus grand qu'un seuil :
                    (i, j, 1) est la position x centrale normalisée de l'objet prédit j de l'image i
                    (i, j, 2) est la position y centrale normalisée de l'objet prédit j de l'image i
                    (i, j, 3) est la largeur normalisée et la hauteur normalisée de l'objet prédit j de l'image i
                    (i, j, 4) est le score pour la classe "cercle" de l'objet prédit j de l'image i
                    (i, j, 5) est le score pour la classe "triangle" de l'objet prédit j de l'image i
                    (i, j, 6) est le score pour la classe "croix" de l'objet prédit j de l'image i

        :param target: Le tenseur cible pour la tâche de détection:
            Dimensions : (N, 3, 5)
                Si un 1 est présent à (i, j, 0), le vecteur (i, j, 0:5) représente un objet
                Si un 0 est présent à (i, j, 0), le vecteur (i, j, 0:5) ne représente aucun objet
                Si le vecteur représente un objet (i, j, :):
                    (i, j, 1) est la position x centrale normalisée de l'objet j de l'image i
                    (i, j, 2) est la position y centrale normalisée de l'objet j de l'image i
                    (i, j, 3) est la largeur normalisée et la hauteur normalisée de l'objet j de l'image i
                    (i, j, 4) est l'indice de la classe de l'objet j de l'image i
        """
        prediction = prediction.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        N = prediction.shape[0]
        C = prediction.shape[1]

        for n in range(N):
            for c in range(C):
                if target[n, c, 0] == 1:
                    class_index = int(target[n, c, 4])
                    self._target_count_by_class[class_index] += 1

            found_target = set()
            for c in range(C):
                confidence = prediction[n, c, 0]

                iou, target_index = self._find_best_target(prediction[n, c], target[n])
                target_class = int(target[n, target_index, 4])
                predicted_class = np.argmax(prediction[n, c, 4:])

                true_positive = 0
                false_positive = 0
                if target_index in found_target or iou < self._intersection_over_union_threshold:
                    false_positive = 1
                elif iou > self._intersection_over_union_threshold and predicted_class == target_class:
                    true_positive = 1
                found_target.add(target_index)

                self._results_by_class[predicted_class].append({
                    'confidence': confidence,
                    'true_positive': true_positive,
                    'false_positive': false_positive,
                })

    def _find_best_target(self, prediction, target):
        C = target.shape[0]
        ious = np.zeros(C)
        for c in range(C):
            iou = detection_intersection_over_union(prediction[1:4], target[c, 1:4])
            ious[c] = iou

        target_index = np.argmax(ious)
        return ious[target_index], target_index

    def get_value(self):
        mean_average_precision = 0
        for class_index in range(self._class_count):
            mean_average_precision += self._calculate_average_precision(self._results_by_class[class_index],
                                                                        self._target_count_by_class[class_index])

        return mean_average_precision / self._class_count

    def _calculate_average_precision(self, results, target_count):
        sorted_results = sorted(results, key=lambda result: result['confidence'], reverse=True)

        recalls = [0]
        precisions = [1]

        true_positive = 0
        false_positive = 0
        for result in sorted_results:
            true_positive += result['true_positive']
            false_positive += result['false_positive']

            recalls.append(true_positive / target_count if target_count > 0 else 0)

            precision_denominator = true_positive + false_positive
            precisions.append(true_positive / precision_denominator if precision_denominator > 0 else 1)

        recalls = np.array(recalls)
        precisions = np.array(precisions)

        sorted_index = np.argsort(recalls)
        recalls = recalls[sorted_index]
        precisions = precisions[sorted_index]

        return np.trapz(y=precisions, x=recalls)


def detection_intersection_over_union(box_a, box_b):
    area_a = box_a[2] ** 2
    area_b = box_b[2] ** 2

    a_tl_x = box_a[0] - box_a[2] / 2
    a_tl_y = box_a[1] - box_a[2] / 2
    a_br_x = box_a[0] + box_a[2] / 2
    a_br_y = box_a[1] + box_a[2] / 2

    b_tl_x = box_b[0] - box_b[2] / 2
    b_tl_y = box_b[1] - box_b[2] / 2
    b_br_x = box_b[0] + box_b[2] / 2
    b_br_y = box_b[1] + box_b[2] / 2

    intersection_w = min(a_br_x, b_br_x) - max(a_tl_x, b_tl_x)
    intersection_h = min(a_br_y, b_br_y) - max(a_tl_y, b_tl_y)
    intersection_w = max(intersection_w, 0)
    intersection_h = max(intersection_h, 0)

    intersection_area = intersection_w * intersection_h

    return intersection_area / (area_a + area_b - intersection_area)


class SegmentationIntersectionOverUnionMetric(Metric):
    def __init__(self, background_class):
        self._background_class = background_class

        self._intersection = 0.0
        self._union = 0.0
        self._epsilon = sys.float_info.epsilon

    def get_name(self):
        return 'Intersection Over Union'

    def clear(self):
        self._intersection = 0.0
        self._union = 0.0

    def accumulate(self, prediction, target):
        """
        Méthode qui accumule les métriques d'un lot de données.
        N: La taille du lot (batch size)
        H: La hauteur des images
        W: La largeur des images

        :param prediction: Le tenseur PyTorch de prédiction des classes de chaque pixel des images
            Dimensions : (N, C, H, W) où C est le nombre de classes en incluant l'arrière-plan (4 dans la problématique)
            (n, c, h, w) : le score de la classe c du pixel (h,w) de l'image n

        :param target: Le tenseur PyTorch cible pour la tâche de segmentation qui contient l'indice de la classe pour chaque pixel
            Dimensions : (N, H, W)
        """
        prediction = prediction.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        intersection, union = segmentation_intersection_over_union(prediction, target, self._background_class)
        self._intersection += intersection
        self._union += union

    def get_value(self):
        return (self._intersection + self._epsilon) / (self._union + self._epsilon)


def segmentation_intersection_over_union(prediction, target, background_class):
    C = prediction.shape[1]
    intersection = 0.0
    union = 0.0
    predicted_segmentation = np.argmax(prediction, axis=1)

    for c in range(C):
        if c == background_class:
            continue

        predicted_mask = predicted_segmentation == c
        target_mask = target == c

        intersection += (predicted_mask * target_mask).sum()
        union += predicted_mask.sum() + target_mask.sum() - (predicted_mask * target_mask).sum()

    return intersection, union
