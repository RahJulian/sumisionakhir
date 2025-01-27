"""Transform module
"""
 
import tensorflow as tf
import tensorflow_transform as tft
 
CATEGORICAL_FEATURES = {
    "gender": 2,
    "hypertension": 2,
    "heart_disease": 2,
    "marital_status": 3,
    "work_type": 4,
    "residence_type": 2,
    "smoking_status": 3,
    "alcohol_intake": 4,
    "physical_activity": 3,
    "stroke_history": 2,
    "dietary_habits": 7
}
NUMERICAL_FEATURES = [
    "age",
    "average_glucose_level",
    "body_mass_index_(bmi)",
    "stress_levels"
]
LABEL_KEY = "diagnosis"
 
def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def convert_num_to_one_hot(label_tensor, num_labels=2):
   
    
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])
 
 
def preprocessing_fn(inputs):
    
   
    
    
    outputs = {}
    
    for key in CATEGORICAL_FEATURES:
        dim = CATEGORICAL_FEATURES[key]
        int_value = tft.compute_and_apply_vocabulary(
            inputs[key], top_k=dim + 1
        )
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1
        )
        tf.compat.v1.logging.info(f'Processing categorical feature: {key}')
        
    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])
        tf.compat.v1.logging.info(f'Processing numerical feature: {feature}')
    
    
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    return outputs


