import os
import argparse
import torch
import pandas as pd
import numpy as np
from evaluation_metric import range_lift_with_delay
from sklearn import metrics
from source import IntroVAE
from source import KPISeries, KpiFrameDataset, KpiFrameDataLoader

def arg_parse():
    """
    Parse arguments to the detect module
    """
    parser = argparse.ArgumentParser(description='KPI Anomaly Detection')

    parser.add_argument("--data", dest='data_path', type=str,
                        default='./data/series24_sr.csv', help='The dataset path')
    parser.add_argument("--max-epoch", dest='epochs', type=int, default=100, help="The random seed")
    parser.add_argument("--batch-size", dest='batch_size', type=int, default=256, help="The number of the batch size")
    parser.add_argument("--window-size", dest='window_size', type=int, default=120, help="The size the sliding window")
    parser.add_argument("--latent-size", dest='latent_size', type=int, default=3, help="The dimension of the latent variables")
    parser.add_argument("--results", dest='filename', type=str, default='result/series24.csv', help="The random seed")
    parser.add_argument("--positive-margin", dest='margin', type=float, default=15, help="The positive margin")
    parser.add_argument('--delay', dest='delay', type=float, default=7)
    return parser.parse_args()


if __name__=="__main__":
    args = arg_parse()
    print('-'*150)
    for arg in vars(args):
        print("{:15s}:{}".format(arg, getattr(args, arg)))
    print('-'*150)

    # set random seed
    seed = 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True

    # read args 
    epochs = args.epochs
    window_size = args.window_size
    latent_size = args.latent_size
    batch_size = args.batch_size
    margin = args.margin


    df = pd.read_csv(args.data_path, header=0, index_col=None)
    kpi = KPISeries(value=df.value, timestamp=df.timestamp, label=df.pred, truth=df.label)

    train_kpi, test_kpi = kpi.split((0.5, 0.5))
    train_kpi, train_kpi_mean, train_kpi_std = train_kpi.normalize(return_statistic=True)
    test_kpi = test_kpi.normalize(mean=train_kpi_mean, std=train_kpi_std)


    # Model
    model = IntroVAE(
        cuda=True,
        max_epoch=epochs,
        latent_dims=latent_size,
        window_size=window_size,
        batch_size=batch_size,
        margin=margin,
    )

    # Training 
    model.fit(train_kpi.label_sampling(1.0), train_kpi)

    # Testing 
    # compute the precision, recall and best F1-score of testing set
    y_prob_train = model.predict(test_kpi.label_sampling(0.))
    y_prob_train = range_lift_with_delay(y_prob_train, test_kpi.truth, delay=7)
    precisions, recalls, thresholds = metrics.precision_recall_curve(test_kpi.truth, y_prob_train)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    ind = np.argmax(f1_scores[np.isfinite(f1_scores)])
    precision = precisions[np.isfinite(precisions)][ind]
    recall = recalls[np.isfinite(recalls)][ind]
    best_f1 = np.max(f1_scores[np.isfinite(f1_scores)])

    print("The best F1 score:{}".format(best_f1))


    # save the results to .csv file
    cont_list = [{
        "Random-seed": seed,
        "Epoch":epochs, 
        "Latent-size":latent_size,
        "Window-size":window_size,
        "Margin":margin,
        "Precision": precision, 
        "Recall": recall, 
        "Best f1-score": best_f1
    }]


    result_df = pd.DataFrame(cont_list)
    filepath = args.filename

    if not os.path.exists(filepath):
        result_df.to_csv(filepath)
    else:
        result_df.to_csv(filepath, mode='a+', header=None)
