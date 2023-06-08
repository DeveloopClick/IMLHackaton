from task_1 import *
from task_2 import *

if __name__ == '__main__':
    model1 = train_model_1('agoda_cancellation_train.csv')
    model2 = train_model_2('agoda_cancellation_train.csv')

    #second part
    results_first_model = test_model_1(model1, 'Agoda_Test_1.csv')
    results_second_model = test_model_2(model2, 'Agoda_Test_2.csv')
