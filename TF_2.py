import tensorflow as tf

DATASET_URL = “https://archive.ics.uci.edu/ml/machine-” \
              “learning-databases/covtype/covtype.data.gz”
DATASET_SIZE = 387698
dataset_path = tf.keras.utils.get_file(
 fname=DATASET_URL.split(’/’)[-1], origin=DATASET_URL)

COLUMN_NAMES = [
    'Elevation', 'Aspect’, ’Slope’,
    'Horizontal_Distance_To_Hydrology’,
    'Vertical_Distance_To_Hydrology’,
    'Horizontal_Distance_To_Roadways’,
    'Hillshade_9am’, ’Hillshade_Noon’, ’Hillshade_3pm’,
    'Horizontal_Distance_To_Fire_Points’, 'Soil_Type',
    'Cover_Type’]

def _parse_line(line):
    # Decode the line into values
 fields = tf.io.decode_csv(
 records=line, record_defaults=[0.0] * 54 + [0])
 # Pack the result into a dictionary
 features = dict(zip(COLUMN_NAMES,
 fields[:10] + [tf.stack(fields[14:54])] + [fields[-1]]))
 # Extract one-hot encoded class label from the features
 class_label = tf.argmax(fields[10:14], axis=0)
 return features, class_label
def csv_input_fn(csv_path, test=False,
 batch_size=DATASET_SIZE // 1000):
 # Create a dataset containing the csv lines
 dataset = tf.data.TextLineDataset(filenames=csv_path,
 compression_type=’GZIP’)

 'https: // www.python - course.eu / tensor_flow_introduction.php'

 # Parse each line
 dataset = dataset.map(_parse_line)
 # Shuffle, repeat, batch the examples for train and test
 dataset = dataset.shuffle(buffer_size=DATASET_SIZE,
 seed=42)
 TEST_SIZE = DATASET_SIZE // 10
 return dataset.take(TEST_SIZE).batch(TEST_SIZE) if test \
 else dataset.skip(TEST_SIZE).repeat().batch(batch_size)