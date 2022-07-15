from absl import app
from nasbench import api

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

def main(argv):
    del argv

    # Load the data from file (this will take some time)
    nasbench_obj = api.NASBench('./dataset/nasbench_full.tfrecord')

    # Create an Inception-like module (5x5 convolution replaced with two 3x3
    # convolutions).
    model_spec = api.ModelSpec(
        # Adjacency matrix of the module
        matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
                [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
                [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
                [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
                [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
                [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
                [0, 0, 0, 0, 0, 0, 0]],   # output layer
        # Operations at the vertices of the module, matches order of matrix
        ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])

    # Query this model from dataset, returns a dictionary containing the metrics
    # associated with this model.
    print(nasbench_obj.query(model_spec))


    nasbench_obj.evaluate(model_spec, './test_output')

if __name__ == "__main__":
    app.run(main)