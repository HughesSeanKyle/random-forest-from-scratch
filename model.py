"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    if len(labels) == 0:
        return 0.0
    counts = np.bincount(labels)
    probabilities = counts / len(labels)
    return 1.0 - np.sum(probabilities ** 2)

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    left_mask = features[:, feature_index] <= threshold
    right_mask = ~left_mask
    return features[left_mask], labels[left_mask], features[right_mask], labels[right_mask]

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    n_left = len(left_labels)
    n_right = len(right_labels)
    n_total = n_left + n_right
    parent_impurity = impurity(parent_labels)
    if n_total == 0:
        return 0.0
    
    return parent_impurity - ((n_left / n_total) * impurity(left_labels) + (n_right / n_total) * impurity(right_labels))

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    best_score = 0.0  # Must be a positive improvement (> 0.0)
    best_feat, best_thresh = None, None
    
    for feat in feature_indices:
        thresholds = np.unique(features[:, feat])
        for thresh in thresholds:
            features_l, labels_l, features_r, labels_r = split_dataset(features, labels, feat, thresh)
            if len(labels_l) == 0 or len(labels_r) == 0:
                continue
                
            score = split_score(labels, labels_l, labels_r)
            if score > best_score:
                best_score = score
                best_feat = feat
                best_thresh = thresh
                
    return {
        'feature_index': best_feat,
        'threshold': best_thresh,
        'score': best_score
    }

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    if depth >= max_depth:
        return True
    if len(labels) < min_samples_split:
        return True
    if len(np.unique(labels)) <= 1:
        return True
    return False

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    if len(labels) == 0:
        return 0
    return int(np.bincount(labels).argmax())

# Step 7 - build_tree
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0, random_state=None):
    """
    Recursively construct a decision tree mapping dictionary keys perfectly 
    to the platform validation harness requirements.
    """
    # 1. Check termination stopping conditions
    if should_stop(labels, depth, max_depth, min_samples_split):
        return {'leaf': True, 'prediction': leaf_prediction(labels)}
        
    n_features = features.shape[1]
    
    # 2. Advanced Feature Subset Parsing (Handles both integers and direct index lists)
    if feature_subset is not None:
        # If it's a list or array of specific indices (e.g., [2, 3])
        if isinstance(feature_subset, (list, np.ndarray)):
            feats = np.array(feature_subset)
        # If it's an integer specifying the subset size count
        elif isinstance(feature_subset, (int, np.integer)) and feature_subset < n_features:
            if random_state is not None:
                rng = np.random.RandomState(random_state + depth)
                feats = rng.choice(n_features, feature_subset, replace=False)
            else:
                feats = np.random.choice(n_features, feature_subset, replace=False)
        else:
            feats = np.arange(n_features)
    else:
        feats = np.arange(n_features)
        
    # 3. Query best_split dictionary
    split_info = best_split(features, labels, feats)
    
    # If no valid split improves impurity/gain, collapse to a leaf node
    if split_info['feature_index'] is None:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}
        
    best_feat = split_info['feature_index']
    best_thresh = split_info['threshold']
        
    features_l, labels_l, features_r, labels_r = split_dataset(features, labels, best_feat, best_thresh)
    
    # 4. Recursive branching pass-down
    left_child = build_tree(features_l, labels_l, max_depth, min_samples_split, feature_subset, depth + 1, random_state)
    right_child = build_tree(features_r, labels_r, max_depth, min_samples_split, feature_subset, depth + 1, random_state)
    
    # 5. Return node layout containing explicit 'feature_index' key string
    return {
        'leaf': False,
        'feature_index': best_feat,
        'threshold': best_thresh,
        'left': left_child,
        'right': right_child
    }

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    if tree['leaf']:
        return tree['prediction']
    if example[tree['feature_index']] <= tree['threshold']:
        return predict_example_tree(tree['left'], example)
    return predict_example_tree(tree['right'], example)

# Step 9 - predict_tree
def predict_tree(tree, features):
    """Predict class labels for every row of `features` using a fitted decision tree.

    tree: dict returned by build_tree
    features: np.ndarray of shape (n, d)
    returns: np.ndarray of shape (n,) with integer class labels
    """
    return np.array([predict_example_tree(tree, x) for x in features])

# Step 10 - bootstrap_sample
def bootstrap_sample(features, labels, rng):
    """Draw a bootstrap sample matching the size of the input features."""
    n_samples = features.shape[0]

    indices = rng.choice(n_samples, size=n_samples, replace=True)

    return features[indices], labels[indices]

# Step 11 - feature_subset
import numpy as np

def feature_subset(num_features, num_to_pick, rng):
    """
    Return num_to_pick distinct random feature indices from range(num_features) using rng.
    Returns a 1-D np.ndarray of shape (num_to_pick,) with no duplicates.
    """
    # Use the Generator object to sample unique indices without replacement
    return rng.choice(num_features, size=num_to_pick, replace=False)

# Step 12 - train_forest
import numpy as np

def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, random_state=0):
    forest = []
    n_samples, n_features = features.shape
    
    # Handle the adaptive baseline rule for feature calculation
    if num_features_per_split is None:
        num_to_pick = int(np.floor(np.sqrt(n_features)))
        num_to_pick = max(1, num_to_pick)  # Enforce a floor value of 1
    else:
        num_to_pick = num_features_per_split

    for i in range(num_trees):
        # Create a unique, deterministic seed value for each tree branch
        tree_seed = random_state + i
        rng = np.random.default_rng(tree_seed)
        
        # 1. Generate the bootstrap row sample (Pass tree_seed directly)
        features_b, labels_b = bootstrap_sample(features, labels, rng)
        
        # 2. Isolate the column indices subset allowed for this entire tree run
        chosen_cols = feature_subset(n_features, num_to_pick, rng)
        
        # 3. Fit the tree using the established feature list slice boundary
        custom_tree = build_tree(
            features=features_b, 
            labels=labels_b, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            feature_subset=chosen_cols, 
            depth=0
        )
        
        # 4. Pack the structural details exactly matching the target schema
        forest.append({
            'tree': custom_tree,
            'feature_indices': chosen_cols
        })
        
    return forest

# Step 13 - combine_predictions
def combine_predictions(tree_predictions):
    n_samples = tree_predictions.shape[1]
    combined = []
    
    for col in range(n_samples):
        column_votes = tree_predictions[:, col]
        # Find the most frequent integer label in the column vector
        combined.append(np.bincount(column_votes).argmax())
        
    return np.array(combined)

# Step 14 - predict_forest
def predict_forest(forest, features):
    # Extract the internal tree configuration from each dict item using the 'tree' key string
    preds = np.array([predict_tree(tree_info['tree'], features) for tree_info in forest])
    return combine_predictions(preds)

# Step 15 - accuracy (not yet solved)
# TODO: implement

