import numpy as np

class eval:
    def top3(self, probabilities, encoder, get_indices: bool = False):
        top3_predictions = []
        top3_indices = []
        
        for prob_row in probabilities:
            top3_indice = np.argsort(prob_row)[-3:][::-1]
            top3_indices.append(top3_indice)

            if not get_indices:
                top3_labels = encoder.inverse_transform(top3_indice)
                
                top3_string = " ".join(top3_labels)
                top3_predictions.append(top3_string)
        
        return top3_indices if get_indices else top3_predictions
    
    def map3(self, y_true, y_predicted, encoder):
        y_predicted = self.top3(y_predicted, encoder=encoder, get_indices=True)
        scores = []

        for preds, true in zip(y_predicted, y_true):
            position_scores = {preds[0]: 1.0, preds[1]: 1/2, preds[2]: 1/3}
            scores.append(position_scores.get(true, 0.0))
        
        return np.mean(scores)